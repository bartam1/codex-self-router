from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from codex_self_router.config import PROFILES, ROUTING_POLICY, ROUTING_TOOL_SPEC, ProfileName
from codex_self_router.evaluation import aggregate, analyze
from codex_self_router.proxy import Bridge, ThreadState
from codex_self_router.report import ReportStore, SwitchRecord, utc_now
from codex_self_router.resume import RolloutReader, ThreadLease


class Socket:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(json.loads(message))


def append(path, *rows):
    with path.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def usage_row(response="new-response", turn="new-turn"):
    return {
        "type": "token_usage_record",
        "timestamp": utc_now(),
        "payload": {
            "thread_id": "thread-1",
            "turn_id": turn,
            "response_id": response,
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "output_tokens": 20,
                "reasoning_output_tokens": 10,
                "total_tokens": 120,
            },
        },
    }


def rollout(tmp_path, *, tools=True, tool_spec=None):
    path = tmp_path / "rollout.jsonl"
    append(
        path,
        {
            "type": "session_meta",
            "payload": {
                "id": "thread-1",
                "dynamic_tools": [tool_spec or ROUTING_TOOL_SPEC] if tools else [],
            },
        },
        usage_row("historical-response", "old-turn"),
    )
    return path


async def cleanup(bridge):
    bridge.closed = True
    tasks = list(bridge.background)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for lease in bridge.thread_leases.values():
        lease.close()


def test_rollout_only_counts_new_complete_records_and_verifies_tools(tmp_path):
    path = rollout(tmp_path)
    reader = RolloutReader(path, "thread-1")
    reader.prime()
    assert reader.events() == []
    append(
        path,
        {"type": "turn_context", "payload": {"turn_id": "new-turn"}},
        {"type": "response_item", "payload": {"type": "function_call", "call_id": "switch"}},
    )
    pending = json.dumps(usage_row())
    with path.open("a") as stream:
        stream.write(pending[:30])
    assert reader.events()[0]["method"] == "rawResponseItem/completed"
    with path.open("a") as stream:
        stream.write(pending[30:] + "\n")
    (event,) = reader.events()
    assert event["params"]["responseId"] == "new-response"
    assert event["params"]["usage"]["reasoningOutputTokens"] == 10
    assert reader.events() == []
    with pytest.raises(ValueError, match="belong"):
        RolloutReader(path, "wrong-thread").prime()
    path.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        reader.events()


def test_rollout_normalizes_naive_timestamps_to_utc(tmp_path):
    path = rollout(tmp_path)
    reader = RolloutReader(path, "thread-1")
    reader.prime()
    row = usage_row()
    row["timestamp"] = "2026-01-01T12:34:56"
    append(path, row)

    (event,) = reader.events()
    assert event["params"]["observedAt"] == "2026-01-01T12:34:56+00:00"


def test_rollout_detects_legacy_tool_schema(tmp_path):
    old_tool = copy.deepcopy(ROUTING_TOOL_SPEC)
    old_tool["tools"][0]["inputSchema"]["properties"].pop("targetReasoningEffort")
    reader = RolloutReader(rollout(tmp_path, tool_spec=old_tool), "thread-1")
    reader.prime()
    assert not reader.router_tool_supports_effort
    assert reader.router_tool_supports_route_state

    pre_state_tool = copy.deepcopy(ROUTING_TOOL_SPEC)
    pre_state_tool["tools"] = [pre_state_tool["tools"][0]]
    pre_state_dir = tmp_path / "pre-state"
    pre_state_dir.mkdir()
    reader = RolloutReader(rollout(pre_state_dir, tool_spec=pre_state_tool), "thread-1")
    reader.prime()
    assert not reader.router_tool_supports_route_state


@pytest.mark.asyncio
async def test_astra_survives_restart_with_tool_policy_and_measurement_lineage(tmp_path):
    store = ReportStore(tmp_path / "reports")
    path = rollout(tmp_path)
    old = Bridge(Socket(), codex_bin=Path("codex"), report_store=store)
    old.threads["thread-1"] = ThreadState(ProfileName.LUNA, "old-turn")
    old.measurements.start_turn("thread-1", "old-turn")
    old._checkpoint_thread("thread-1", "Keep project instructions.\n\n" + ROUTING_POLICY)

    async def approve(**kwargs):
        return True

    async def wait(*args):
        pass

    async def update(method, params):
        assert method == "turn/settings/update"
        return {"result": {"status": "applied"}}

    async def send(message):
        pass

    old._request_approval, old._wait_until_call_response_recorded = approve, wait
    old._internal_request, old._send_upstream = update, send
    await old._handle_switch_call(
        {
            "id": "switch",
            "params": {
                "threadId": "thread-1",
                "turnId": "old-turn",
                "callId": "call-1",
                "arguments": {
                    "targetProfile": "astra",
                    "reason": "design",
                    "nextAction": "plan",
                    "estimatedFollowUpSteps": 4,
                },
            },
        }
    )
    await old._observe_raw_response(
        {
            "params": {
                "threadId": "thread-1",
                "turnId": "old-turn",
                "responseId": "historical-response",
                "usage": {"inputTokens": 100, "cachedInputTokens": 0, "outputTokens": 20},
            }
        }
    )
    old.measurements.close()
    old._save_report()
    await cleanup(old)

    socket = Socket()
    resumed = Bridge(socket, codex_bin=Path("codex"), report_store=store)
    calls = []

    async def internal(method, params):
        calls.append((method, params))
        if method == "thread/read":
            return {"result": {"thread": {"id": "thread-1", "path": str(path)}}}
        assert method == "thread/resume"
        assert params["model"] == PROFILES[ProfileName.ASTRA].model
        assert params["config"]["model_reasoning_effort"] == "xhigh"
        assert "Keep project instructions." in params["developerInstructions"]
        assert params["developerInstructions"].count(ROUTING_POLICY) == 1
        assert "dynamicTools" not in params and "experimentalRawEvents" not in params
        return {
            "result": {
                "thread": {"id": "thread-1", "turns": [{"id": "old-turn"}]},
                "model": PROFILES[ProfileName.ASTRA].model,
                "reasoningEffort": "xhigh",
            }
        }

    resumed._internal_request = internal
    try:
        await resumed._resume_thread({"id": 1, "params": {"threadId": "thread-1"}})
        assert socket.sent[-1]["result"]["model"] == PROFILES[ProfileName.ASTRA].model
        assert resumed.threads["thread-1"].profile == ProfileName.ASTRA
        assert resumed.report.responses == []
        assert resumed.report.switches == []  # Old approvals are not replayed.
        assert resumed.report.metadata["resumedThreads"][0]["previousSessionIds"] == [
            old.report.session_id
        ]
        forwarded = []

        async def forward(message):
            forwarded.append(message)

        resumed._send_upstream = forward
        await resumed._handle_client_payload(
            json.dumps(
                {
                    "id": 2,
                    "method": "turn/start",
                    "params": {
                        "threadId": "thread-1",
                        "input": [{"type": "text", "text": "continue"}],
                    },
                }
            )
        )
        assert forwarded[-1]["params"]["model"] == PROFILES[ProfileName.ASTRA].model
        context = resumed.client_requests.pop("2")
        resumed._observe_client_response(context, {"result": {"turn": {"id": "new-turn"}}})
        append(path, usage_row("historical-response", "old-turn"), usage_row())
        await resumed._drain_rollout("thread-1")
        await resumed._drain_rollout("thread-1")
        assert len(resumed.report.responses) == 1
        assert resumed.report.responses[0].usage_source == "codex-rollout"
        assert resumed.report.responses[0].profile == "astra"
        await resumed._handle_upstream_payload(
            json.dumps(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {"id": "new-turn", "status": "completed"},
                    },
                }
            )
        )
        (task,) = analyze(resumed.report.to_dict())["tasks"]
        assert task["taskId"] == "old-turn"
        assert task["resumedFromSessionId"] == old.report.session_id
        (group,) = aggregate(store.all())["groups"]
        assert group["tasks"] == 1
        assert group["responses"] == 2
        assert group["medianTaskWallMs"] is None
    finally:
        await cleanup(resumed)


@pytest.mark.asyncio
async def test_resume_preserves_pending_temporary_route_restore(tmp_path):
    store = ReportStore(tmp_path / "reports")
    path = rollout(tmp_path)
    old = Bridge(Socket(), codex_bin=Path("codex"), report_store=store)
    old.threads["thread-1"] = ThreadState(
        ProfileName.ASTRA,
        "turn-1",
        effort="medium",
        temporary_restore_profile=ProfileName.LUNA,
        temporary_restore_effort="low",
        temporary_task_id="turn-1",
        explicit_route_task_id="turn-1",
    )
    old.measurements.start_turn("thread-1", "turn-1")
    old._checkpoint_thread("thread-1")
    await cleanup(old)

    resumed = Bridge(Socket(), codex_bin=Path("codex"), report_store=store)

    async def internal(method, params):
        if method == "thread/read":
            return {"result": {"thread": {"id": "thread-1", "path": str(path)}}}
        return {
            "result": {
                "thread": {"id": "thread-1"},
                "model": PROFILES[ProfileName.ASTRA].model,
                "reasoningEffort": "medium",
            }
        }

    resumed._internal_request = internal
    try:
        await resumed._resume_thread({"id": 1, "params": {"threadId": "thread-1"}})
        state = resumed.threads["thread-1"]
        assert state.temporary_restore_profile == ProfileName.LUNA
        assert state.temporary_restore_effort == "low"
        assert state.temporary_task_id == "turn-1"
        assert state.explicit_route_task_id == "turn-1"
    finally:
        await cleanup(resumed)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["ordinary", "upstream", "wrong-model", "override-history"])
async def test_resume_failures_are_explicit_and_do_not_create_state(tmp_path, failure):
    path = rollout(tmp_path, tools=failure != "ordinary")
    bridge = Bridge(
        Socket(), codex_bin=Path("codex"), report_store=ReportStore(tmp_path / "reports")
    )

    async def internal(method, params):
        if method == "thread/read":
            return {"result": {"thread": {"id": "thread-1", "path": str(path)}}}
        if failure == "upstream":
            return {"error": {"code": -32000, "message": "cannot read saved session"}}
        return {"result": {"thread": {"id": "thread-1"}, "model": "unknown-model"}}

    bridge._internal_request = internal
    params = {"threadId": "thread-1"}
    if failure == "override-history":
        params["history"] = []
    await bridge._resume_thread({"id": 1, "params": params})
    assert "error" in bridge.websocket.sent[-1]
    assert bridge.threads == {}
    assert bridge.thread_leases == {}
    assert bridge.report.thread_states == {}


def test_lease_prevents_two_router_owners_and_can_be_reacquired(tmp_path):
    path = tmp_path / "thread.lock"
    first = ThreadLease(path)
    try:
        with pytest.raises(ValueError, match="already attached"):
            ThreadLease(path)
    finally:
        first.close()
    second = ThreadLease(path)
    second.close()


@pytest.mark.asyncio
async def test_resume_switch_waits_for_durable_requesting_usage(tmp_path):
    path = rollout(tmp_path)
    reader = RolloutReader(path, "thread-1")
    reader.prime()
    bridge = Bridge(
        Socket(), codex_bin=Path("codex"), report_store=ReportStore(tmp_path / "reports")
    )
    bridge.rollouts["thread-1"] = reader
    bridge.threads["thread-1"] = ThreadState(ProfileName.ASTRA, "new-turn", resumed=True)
    append(
        path,
        {"type": "turn_context", "payload": {"turn_id": "new-turn"}},
        {"type": "response_item", "payload": {"type": "function_call", "call_id": "switch"}},
        usage_row(),
    )
    await bridge._wait_until_call_response_recorded("switch", "thread-1", "new-turn")
    assert bridge.response_sequence[("thread-1", "new-turn")] == 1
    assert bridge.report.responses[0].profile == "astra"


@pytest.mark.asyncio
async def test_nested_resume_call_does_not_require_nonexistent_parent_id(tmp_path):
    path = rollout(tmp_path)
    reader = RolloutReader(path, "thread-1")
    reader.prime()
    bridge = Bridge(
        Socket(), codex_bin=Path("codex"), report_store=ReportStore(tmp_path / "reports")
    )
    bridge.rollouts["thread-1"] = reader
    bridge.threads["thread-1"] = ThreadState(ProfileName.ASTRA, "new-turn", resumed=True)
    bridge.measurements.start_turn("thread-1", "new-turn")
    bridge._checkpoint_thread("thread-1")
    delayed_usage = usage_row()
    try:
        append(
            path,
            {"type": "turn_context", "payload": {"turn_id": "new-turn"}},
            {
                "type": "response_item",
                "payload": {"type": "custom_tool_call", "call_id": "outer-exec", "name": "exec"},
            },
        )
        waiting = asyncio.create_task(
            bridge._wait_until_call_response_recorded("exec-nested-uuid", "thread-1", "new-turn")
        )
        await asyncio.sleep(0.01)
        assert not waiting.done()  # Never interrupt before the outer response's usage is saved.
        append(path, delayed_usage)
        await asyncio.wait_for(
            waiting,
            timeout=1,
        )
        assert "exec-nested-uuid" not in bridge.raw_call_sequence
        late_astra = usage_row("late-astra-response")
        bridge.threads["thread-1"].profile = ProfileName.LUNA
        bridge.threads["thread-1"].effort = "medium"
        bridge._checkpoint_thread("thread-1")
        bridge.report.switches.append(
            SwitchRecord(
                utc_now(),
                "thread-1",
                "new-turn",
                "astra",
                "luna",
                "agent-tool",
                "applied",
                response_index=0,
            )
        )
        append(path, late_astra, usage_row("luna-response"))
        await bridge._drain_rollout("thread-1")
        assert bridge.report.responses[0].profile == "astra"
        assert bridge.report.responses[0].effort == "xhigh"
        assert bridge.report.responses[1].profile == "astra"
        assert bridge.report.responses[1].effort == "xhigh"
        assert bridge.report.responses[2].profile == "luna"
        (phase,) = analyze(bridge.report.to_dict())["phases"]
        assert phase["observedSteps"] == 1
        append(
            path,
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "outer-exec",
                    "output": "done",
                },
            },
        )
        await bridge._drain_rollout("thread-1")
        assert "outer-exec" not in bridge.open_raw_calls
    finally:
        await cleanup(bridge)


@pytest.mark.asyncio
async def test_resume_without_turn_keeps_unfinished_task_checkpoint(tmp_path):
    store = ReportStore(tmp_path / "reports")
    path = rollout(tmp_path)
    old = Bridge(Socket(), codex_bin=Path("codex"), report_store=store)
    old.threads["thread-1"] = ThreadState(ProfileName.ASTRA, "unfinished", effort="high")
    old.measurements.start_turn("thread-1", "unfinished")
    old._checkpoint_thread("thread-1")
    await cleanup(old)
    for _ in range(2):
        resumed = Bridge(Socket(), codex_bin=Path("codex"), report_store=store)

        async def internal(method, params):
            if method == "thread/read":
                return {"result": {"thread": {"id": "thread-1", "path": str(path)}}}
            return {
                "result": {
                    "thread": {"id": "thread-1"},
                    "model": PROFILES[ProfileName.ASTRA].model,
                    "reasoningEffort": "high",
                }
            }

        resumed._internal_request = internal
        try:
            await resumed._resume_thread({"id": 1, "params": {"threadId": "thread-1"}})
            assert "error" not in resumed.websocket.sent[-1]
            assert resumed.measurements.resumed_tasks["thread-1"]["task_id"] == "unfinished"
            assert store.thread_checkpoint("thread-1")["last_task"]["task_id"] == "unfinished"
        finally:
            await cleanup(resumed)
