from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from codex_self_router.config import PROFILES, ProfileName, SwitchApproval
from codex_self_router.evaluation import analyze
from codex_self_router.proxy import (
    Bridge,
    RequestContext,
    ThreadState,
    _id_key,
    _is_async_question_item,
    inspect_model_catalog,
)
from codex_self_router.report import ReportStore


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)


def switch_request(request_id: str = "server-1") -> dict:
    return {
        "id": request_id,
        "method": "item/tool/call",
        "params": {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "callId": "call-1",
            "namespace": "self_router",
            "tool": "request_model_switch",
            "arguments": {
                "targetProfile": "astra",
                "reason": "architecture phase",
                "nextAction": "design the boundary",
                "estimatedFollowUpSteps": 4,
            },
        },
    }


@pytest.mark.asyncio
async def test_approved_agent_switch_updates_before_tool_result(tmp_path) -> None:
    websocket = FakeWebSocket()
    bridge = Bridge(websocket, codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    bridge.threads["thread-1"] = ThreadState(ProfileName.SOL, "turn-1")
    upstream: list[dict] = []
    order: list[str] = []

    async def approve(**_kwargs) -> bool:
        return True

    async def wait(*_args) -> None:
        order.append("usage")

    async def internal(method, params) -> dict:
        assert method == "turn/settings/update"
        assert params["model"] == PROFILES[ProfileName.ASTRA].model
        order.append("update")
        return {"result": {"status": "applied"}}

    async def send(message) -> None:
        order.append("tool-result")
        upstream.append(message)

    bridge._request_approval = approve
    bridge._wait_until_call_response_recorded = wait
    bridge._internal_request = internal
    bridge._send_upstream = send

    await bridge._handle_switch_call(switch_request())

    assert order == ["usage", "update", "tool-result"]
    assert bridge.threads["thread-1"].profile == ProfileName.ASTRA
    assert upstream[0]["result"]["success"] is True
    assert bridge.report.switches[0].outcome == "applied"
    notice = json.loads(websocket.sent[-1])
    assert notice["method"] == "warning"
    assert "astra/xhigh" in notice["params"]["message"]
    assert "model label may lag" in notice["params"]["message"]


@pytest.mark.asyncio
async def test_never_policy_auto_authorizes_agent_switch(tmp_path) -> None:
    bridge = Bridge(
        FakeWebSocket(),
        codex_bin=Path("codex"),
        report_store=ReportStore(tmp_path),
        switch_approval=SwitchApproval.NEVER,
    )
    bridge.threads["thread-1"] = ThreadState(ProfileName.SOL, "turn-1")
    upstream: list[dict] = []

    async def should_not_approve(**_kwargs) -> bool:
        raise AssertionError("approval should not be requested")

    async def wait(*_args) -> None:
        return None

    async def internal(_method, _params) -> dict:
        return {"result": {"status": "applied"}}

    async def send(message) -> None:
        upstream.append(message)

    bridge._request_approval = should_not_approve
    bridge._wait_until_call_response_recorded = wait
    bridge._internal_request = internal
    bridge._send_upstream = send

    await bridge._handle_switch_call(switch_request())

    switch = bridge.report.switches[0]
    assert switch.outcome == "applied"
    assert switch.approval_policy == "never"
    assert switch.approval_required is False
    assert switch.authorization == "router-policy"
    assert switch.approval_ms == 0
    assert bridge.threads["thread-1"].profile == ProfileName.ASTRA
    assert upstream[0]["result"]["success"] is True


@pytest.mark.asyncio
async def test_get_current_route_returns_exact_router_state(tmp_path) -> None:
    bridge = Bridge(
        FakeWebSocket(),
        codex_bin=Path("codex"),
        report_store=ReportStore(tmp_path),
        switch_approval=SwitchApproval.NEVER,
    )
    bridge.threads["thread-1"] = ThreadState(ProfileName.TERRA, "turn-1", effort="low")
    upstream: list[dict] = []

    async def send(message) -> None:
        upstream.append(message)

    bridge._send_upstream = send
    await bridge._handle_route_state_call(
        {
            "id": "route-state",
            "params": {"threadId": "thread-1", "turnId": "turn-1", "arguments": {}},
        }
    )

    result = upstream[0]["result"]
    assert result["success"] is True
    assert json.loads(result["contentItems"][0]["text"]) == {
        "profile": "terra",
        "model": PROFILES[ProfileName.TERRA].model,
        "reasoningEffort": "low",
        "approvalPolicy": "never",
    }


@pytest.mark.asyncio
async def test_client_settings_update_keeps_router_state_exact(tmp_path) -> None:
    websocket = FakeWebSocket()
    bridge = Bridge(websocket, codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    bridge.threads["thread-1"] = ThreadState(ProfileName.SOL, "turn-1", effort="medium")
    forwarded: list[dict] = []

    async def send(message) -> None:
        forwarded.append(message)

    bridge._send_upstream = send
    await bridge._handle_client_payload(
        json.dumps(
            {
                "id": 7,
                "method": "turn/settings/update",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "model": PROFILES[ProfileName.LUNA].model,
                    "effort": "low",
                },
            }
        )
    )
    assert forwarded[0]["method"] == "turn/settings/update"

    await bridge._handle_upstream_payload(json.dumps({"id": 7, "result": {"status": "applied"}}))
    assert bridge.threads["thread-1"].profile == ProfileName.LUNA
    assert bridge.threads["thread-1"].effort == "low"
    assert bridge.report.switches[0].source == "client-settings"


@pytest.mark.asyncio
async def test_stale_tui_route_does_not_undo_agent_switch(tmp_path) -> None:
    bridge = Bridge(FakeWebSocket(), codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    bridge.threads["thread-1"] = ThreadState(
        ProfileName.SOL,
        effort="xhigh",
        client_profile=ProfileName.LUNA,
        client_effort="high",
    )
    forwarded: list[dict] = []

    async def send(message) -> None:
        forwarded.append(message)

    bridge._send_upstream = send
    await bridge._handle_client_payload(
        json.dumps(
            {
                "id": 8,
                "method": "turn/start",
                "params": {
                    "threadId": "thread-1",
                    "model": PROFILES[ProfileName.LUNA].model,
                    "effort": "high",
                    "input": [{"type": "text", "text": "continue review"}],
                },
            }
        )
    )

    params = forwarded[0]["params"]
    assert params["model"] == PROFILES[ProfileName.SOL].model
    assert params["effort"] == "xhigh"
    assert params["input"] == [{"type": "text", "text": "continue review"}]
    await bridge._handle_upstream_payload(
        json.dumps({"id": 8, "result": {"turn": {"id": "turn-2"}}})
    )
    assert bridge.threads["thread-1"].profile == ProfileName.SOL
    assert bridge.threads["thread-1"].effort == "xhigh"
    assert bridge.threads["thread-1"].client_profile == ProfileName.LUNA


@pytest.mark.asyncio
async def test_changed_tui_route_is_an_intentional_override(tmp_path) -> None:
    bridge = Bridge(FakeWebSocket(), codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    bridge.threads["thread-1"] = ThreadState(
        ProfileName.SOL,
        effort="xhigh",
        client_profile=ProfileName.LUNA,
        client_effort="high",
    )
    forwarded: list[dict] = []

    async def send(message) -> None:
        forwarded.append(message)

    bridge._send_upstream = send
    await bridge._handle_client_payload(
        json.dumps(
            {
                "id": 9,
                "method": "turn/start",
                "params": {
                    "threadId": "thread-1",
                    "model": PROFILES[ProfileName.TERRA].model,
                    "effort": "medium",
                    "input": [{"type": "text", "text": "continue"}],
                },
            }
        )
    )
    assert forwarded[0]["params"]["model"] == PROFILES[ProfileName.TERRA].model
    await bridge._handle_upstream_payload(
        json.dumps({"id": 9, "result": {"turn": {"id": "turn-3"}}})
    )
    state = bridge.threads["thread-1"]
    assert (state.profile, state.effort) == (ProfileName.TERRA, "medium")
    assert (state.client_profile, state.client_effort) == (ProfileName.TERRA, "medium")


@pytest.mark.asyncio
async def test_incompatible_agent_switch_continues_in_new_turn(tmp_path) -> None:
    websocket = FakeWebSocket()
    bridge = Bridge(websocket, codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    bridge.measurements.start_turn("thread-1", "turn-1")
    bridge.threads["thread-1"] = ThreadState(
        ProfileName.LUNA,
        "turn-1",
        {
            "mode": "plan",
            "settings": {
                "model": PROFILES[ProfileName.LUNA].model,
                "reasoning_effort": "medium",
                "developer_instructions": "Plan carefully.",
            },
        },
    )
    calls: list[tuple[str, dict]] = []
    upstream: list[dict] = []

    async def approve(**_kwargs) -> bool:
        return True

    async def wait(*_args) -> None:
        return None

    async def internal(method, params) -> dict:
        calls.append((method, params))
        if method == "turn/settings/update":
            return {
                "error": {
                    "message": "the destination changes the admitted node REPL review requirement"
                }
            }
        if method == "turn/interrupt":
            await bridge._handle_upstream_payload(
                json.dumps(
                    {
                        "method": "turn/completed",
                        "params": {
                            "threadId": "thread-1",
                            "turn": {"id": "turn-1", "status": "interrupted"},
                        },
                    }
                )
            )
            return {"result": {}}
        if method == "turn/start":
            return {"result": {"turn": {"id": "turn-2", "status": "inProgress"}}}
        raise AssertionError(f"unexpected internal method: {method}")

    async def send(message) -> None:
        upstream.append(message)

    bridge._request_approval = approve
    bridge._wait_until_call_response_recorded = wait
    bridge._internal_request = internal
    bridge._send_upstream = send

    await bridge._handle_switch_call(switch_request())

    assert [method for method, _ in calls] == [
        "turn/settings/update",
        "turn/interrupt",
        "turn/start",
    ]
    continuation = calls[-1][1]
    assert continuation["input"] == []
    assert continuation["toolOutput"]["namespace"] == "self_router"
    continuation_output = continuation["toolOutput"]["output"]
    assert "automatic continuation of the existing user instruction" in continuation_output
    assert "not a user cancellation" in continuation_output
    assert "request_user_input_async" in continuation_output
    assert "Make and state reasonable assumptions" in continuation_output
    assert "design the boundary" in continuation_output
    assert continuation["model"] == PROFILES[ProfileName.ASTRA].model
    assert continuation["collaborationMode"]["settings"] == {
        "model": PROFILES[ProfileName.ASTRA].model,
        "reasoning_effort": "xhigh",
        "developer_instructions": "Plan carefully.",
    }
    assert bridge.threads["thread-1"].profile == ProfileName.ASTRA
    assert bridge.threads["thread-1"].active_turn_id == "turn-2"
    assert upstream == []
    assert bridge.report.switches[0].outcome == "applied"
    assert bridge.report.switches[0].turn_id == "turn-2"
    assert "fresh turn" in bridge.report.switches[0].detail
    switch = bridge.report.switches[0]
    assert switch.origin_turn_id == "turn-1"
    assert switch.mechanism == "continuation"
    assert switch.approval_ms >= 0
    assert switch.apply_ms >= 0
    assert switch.total_ms >= switch.apply_ms
    assert switch.response_index == 0
    await bridge._handle_upstream_payload(
        json.dumps(
            {
                "method": "rawResponse/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-2",
                    "responseId": "astra-response",
                    "usage": {"inputTokens": 100, "cachedInputTokens": 0, "outputTokens": 10},
                },
            }
        )
    )
    await bridge._handle_upstream_payload(
        json.dumps(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turn": {"id": "turn-2", "status": "completed"},
                },
            }
        )
    )
    report = bridge.report_store.latest()
    (task,) = analyze(report)["tasks"]
    assert task["taskId"] == "turn-1"
    assert task["turnIds"] == ["turn-1", "turn-2"]
    assert task["routerInterruptions"] == 1
    assert task["closed"]
    assert analyze(report)["phases"][0]["observedSteps"] == 1


@pytest.mark.asyncio
async def test_model_switched_continuation_suppresses_only_async_questions(tmp_path) -> None:
    websocket = FakeWebSocket()
    bridge = Bridge(websocket, codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    bridge.threads["thread-1"] = ThreadState(ProfileName.ASTRA, "turn-2")
    bridge.automatic_continuation_turns.add(("thread-1", "turn-2"))

    async_question = {
        "method": "item/completed",
        "params": {
            "threadId": "thread-1",
            "turnId": "turn-2",
            "item": {
                "type": "agentMessage",
                "id": "async-1",
                "text": "Which scope?",
                "phase": "final_answer",
                "delivery": "async",
                "questions": [{"title": "Which scope?", "options": None}],
            },
        },
    }
    assert _is_async_question_item(async_question)
    await bridge._handle_upstream_payload(json.dumps(async_question))
    assert websocket.sent == []

    blocking_question = {
        "method": "item/completed",
        "params": {
            "threadId": "thread-1",
            "turnId": "turn-2",
            "item": {"type": "toolCall", "tool": "request_user_input"},
        },
    }
    await bridge._handle_upstream_payload(json.dumps(blocking_question))
    assert len(websocket.sent) == 1

    completed = {
        "method": "turn/completed",
        "params": {"threadId": "thread-1", "turn": {"id": "turn-2", "status": "completed"}},
    }
    await bridge._handle_upstream_payload(json.dumps(completed))
    assert ("thread-1", "turn-2") not in bridge.automatic_continuation_turns


@pytest.mark.asyncio
async def test_denied_agent_switch_keeps_profile(tmp_path) -> None:
    bridge = Bridge(FakeWebSocket(), codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    bridge.threads["thread-1"] = ThreadState(ProfileName.SOL, "turn-1")
    upstream: list[dict] = []

    async def deny(**_kwargs) -> bool:
        return False

    async def send(message) -> None:
        upstream.append(message)

    bridge._request_approval = deny
    bridge._send_upstream = send
    await bridge._handle_switch_call(switch_request())

    assert bridge.threads["thread-1"].profile == ProfileName.SOL
    assert upstream[0]["result"]["success"] is False
    assert bridge.report.switches[0].outcome == "denied"
    assert bridge.report.switches[0].approval_ms >= 0
    assert bridge.report.switches[0].apply_ms is None


@pytest.mark.asyncio
async def test_same_profile_is_a_noop_without_approval(tmp_path) -> None:
    bridge = Bridge(FakeWebSocket(), codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    bridge.threads["thread-1"] = ThreadState(ProfileName.ASTRA, "turn-1")
    upstream: list[dict] = []

    async def should_not_approve(**_kwargs) -> bool:
        raise AssertionError("approval should not be requested")

    async def send(message) -> None:
        upstream.append(message)

    bridge._request_approval = should_not_approve
    bridge._send_upstream = send
    await bridge._handle_switch_call(switch_request())
    assert upstream[0]["result"]["success"] is True
    assert bridge.report.switches[0].outcome == "noop"


@pytest.mark.asyncio
async def test_agent_can_change_reasoning_effort_without_changing_model(tmp_path) -> None:
    bridge = Bridge(FakeWebSocket(), codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    bridge.threads["thread-1"] = ThreadState(ProfileName.SOL, "turn-1", effort="medium")
    request = switch_request()
    request["params"]["arguments"].pop("targetProfile")
    request["params"]["arguments"]["targetReasoningEffort"] = "low"
    calls = []
    upstream = []

    async def approve(**kwargs):
        assert kwargs["previous"] == kwargs["target"] == ProfileName.SOL
        assert kwargs["previous_effort"] == "medium"
        assert kwargs["target_effort"] == "low"
        return True

    async def wait(*args):
        return None

    async def internal(method, params):
        calls.append((method, params))
        return {"result": {"status": "applied"}}

    async def send(message):
        upstream.append(message)

    bridge._request_approval = approve
    bridge._wait_until_call_response_recorded = wait
    bridge._internal_request = internal
    bridge._send_upstream = send
    await bridge._handle_switch_call(request)
    assert calls[0][1]["model"] == PROFILES[ProfileName.SOL].model
    assert calls[0][1]["effort"] == "low"
    assert bridge.threads["thread-1"].profile == ProfileName.SOL
    assert bridge.threads["thread-1"].effort == "low"
    assert bridge.report.switches[0].from_effort == "medium"
    assert bridge.report.switches[0].to_effort == "low"
    assert upstream[0]["result"]["success"] is True


@pytest.mark.asyncio
async def test_duplicate_usage_and_untracked_models_are_not_fabricated(tmp_path):
    bridge = Bridge(FakeWebSocket(), codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    message = json.dumps(
        {
            "method": "rawResponse/completed",
            "params": {
                "threadId": "child",
                "turnId": "child-turn",
                "responseId": "same",
                "usage": {"inputTokens": 100, "cachedInputTokens": 0, "outputTokens": 10},
            },
        }
    )
    await bridge._handle_upstream_payload(message)
    await bridge._handle_upstream_payload(message)
    assert len(bridge.report.responses) == 1
    assert bridge.report.responses[0].model is None
    assert bridge.report.costs()["unpricedResponses"] == 1
    events = list(tmp_path.glob("*.events.jsonl"))
    assert len(events) == 1
    assert len(events[0].read_text().splitlines()) == 1


def test_model_catalog_validation() -> None:
    response = {
        "result": {
            "data": [
                {
                    "model": profile.model,
                    "supportedReasoningEfforts": [
                        {"reasoningEffort": effort, "description": "test"}
                        for effort in profile.allowed_efforts
                    ],
                }
                for profile in PROFILES.values()
            ]
        }
    }
    assert inspect_model_catalog(response) == []
    response["result"]["data"].pop()
    assert inspect_model_catalog(response) == ["model unavailable: gpt-6-astra"]


@pytest.mark.asyncio
async def test_approval_response_is_intercepted_and_resolved(tmp_path) -> None:
    websocket = FakeWebSocket()
    bridge = Bridge(websocket, codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    task = asyncio.create_task(
        bridge._request_approval(
            thread_id="thread-1",
            turn_id="turn-1",
            call_id="call-1",
            previous=ProfileName.SOL,
            previous_effort="medium",
            target=ProfileName.ASTRA,
            target_effort="xhigh",
            reason="hard problem",
            next_action="solve it",
            steps=3,
        )
    )
    await asyncio.sleep(0)
    prompt = json.loads(websocket.sent[0])
    await bridge._handle_client_payload(
        json.dumps(
            {
                "id": prompt["id"],
                "result": {"answers": {"model_switch": {"answers": ["Approve switch"]}}},
            }
        )
    )

    assert await task is True
    resolved = json.loads(websocket.sent[1])
    assert resolved["method"] == "serverRequest/resolved"
    assert resolved["params"]["requestId"] == prompt["id"]


@pytest.mark.asyncio
async def test_server_request_id_cannot_consume_client_response_context(tmp_path) -> None:
    websocket = FakeWebSocket()
    bridge = Bridge(websocket, codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    bridge.client_requests[_id_key(7)] = RequestContext("thread/start")
    server_request = {
        "id": 7,
        "method": "item/commandExecution/requestApproval",
        "params": {"threadId": "thread-1", "turnId": "turn-1", "itemId": "item-1"},
    }

    await bridge._handle_upstream_payload(json.dumps(server_request))

    assert _id_key(7) in bridge.client_requests
    assert json.loads(websocket.sent[0]) == server_request


@pytest.mark.asyncio
async def test_fork_is_rejected_before_reaching_upstream(tmp_path) -> None:
    websocket = FakeWebSocket()
    bridge = Bridge(websocket, codex_bin=Path("codex"), report_store=ReportStore(tmp_path))

    await bridge._handle_client_payload(
        json.dumps({"id": 9, "method": "thread/fork", "params": {"threadId": "thread-old"}})
    )

    response = json.loads(websocket.sent[0])
    assert response["id"] == 9
    assert response["error"]["code"] == -32601
    assert "fresh thread" in response["error"]["message"]


@pytest.mark.asyncio
async def test_explicit_steer_switches_without_approval_before_forwarding(tmp_path) -> None:
    bridge = Bridge(FakeWebSocket(), codex_bin=Path("codex"), report_store=ReportStore(tmp_path))
    bridge.threads["thread-1"] = ThreadState(ProfileName.SOL, "turn-1")
    order: list[str] = []
    forwarded: list[dict] = []

    async def internal(method, params) -> dict:
        assert method == "turn/settings/update"
        assert params["model"] == PROFILES[ProfileName.ASTRA].model
        order.append("update")
        return {"result": {"status": "applied"}}

    async def send(message) -> None:
        order.append("steer")
        forwarded.append(message)

    bridge._internal_request = internal
    bridge._send_upstream = send
    await bridge._apply_user_steer(
        {
            "id": 12,
            "method": "turn/steer",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "input": [{"type": "text", "text": "reconsider"}],
            },
        },
        ProfileName.ASTRA,
        "xhigh",
        "#3",
    )

    assert order == ["update", "steer"]
    assert bridge.threads["thread-1"].profile == ProfileName.ASTRA
    assert forwarded[0]["params"]["input"] == [{"type": "text", "text": "reconsider"}]
    assert bridge.report.switches[0].source == "user-directive"
