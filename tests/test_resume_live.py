"""Opt-in protocol smoke test. Creates its own thread; never resumes a user's active thread."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from websockets.asyncio.client import connect

from codex_self_router.cli import make_server
from codex_self_router.config import PROFILES, ProfileName
from codex_self_router.evaluation import analyze
from codex_self_router.report import ReportStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROUTER_LIVE_CODEX"),
    reason="set ROUTER_LIVE_CODEX for a real inference test",
)
SWITCH_PROMPT = (
    "This is an authorized router smoke test. Call self_router.request_model_switch "
    "with targetProfile luna, targetReasoningEffort low, reason 'resume smoke test', "
    "nextAction 'Reply READY', "
    "and estimatedFollowUpSteps 1. After approval reply READY. Do nothing else."
)


async def response(ws, request_id):
    async with asyncio.timeout(45):
        while True:
            message = json.loads(await ws.recv())
            if message.get("id") == request_id and "method" not in message:
                assert "error" not in message, message
                return message["result"]


async def start_client(ws):
    await ws.send(
        json.dumps(
            {
                "id": 1,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "router_resume_smoke", "version": "1"},
                    "capabilities": {"experimentalApi": True},
                },
            }
        )
    )
    await response(ws, 1)
    await ws.send(json.dumps({"method": "initialized"}))


async def complete_turn(ws):
    approvals = 0
    async with asyncio.timeout(150):
        while True:
            message = json.loads(await ws.recv())
            if message.get("method") == "item/tool/requestUserInput":
                questions = message["params"]["questions"]
                assert questions[0]["id"] == "model_switch", message
                approvals += 1
                await ws.send(
                    json.dumps(
                        {
                            "id": message["id"],
                            "result": {
                                "answers": {"model_switch": {"answers": ["Approve switch"]}},
                            },
                        }
                    )
                )
            if message.get("method") == "turn/completed":
                status = message["params"]["turn"]["status"]
                if status != "interrupted":
                    assert status == "completed", message
                    return approvals


@pytest.mark.asyncio
async def test_real_fork_inherits_route_and_is_reported(tmp_path):
    binary = Path(os.environ["ROUTER_LIVE_CODEX"])
    store = ReportStore(tmp_path / "reports")
    server = await make_server(binary, "127.0.0.1", 0, store)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            await start_client(ws)
            await ws.send(
                json.dumps(
                    {
                        "id": 2,
                        "method": "thread/start",
                        "params": {
                            "model": PROFILES[ProfileName.LUNA].model,
                            "cwd": str(tmp_path),
                            "sandbox": "read-only",
                            "approvalPolicy": "never",
                        },
                    }
                )
            )
            started = await response(ws, 2)
            source_thread = started["thread"]["id"]
            source_effort = started["reasoningEffort"]
            await ws.send(
                json.dumps(
                    {
                        "id": 3,
                        "method": "turn/start",
                        "params": {
                            "threadId": source_thread,
                            "input": [{"type": "text", "text": "Reply READY. Use no tools."}],
                        },
                    }
                )
            )
            await response(ws, 3)
            assert await complete_turn(ws) == 0
            await ws.send(
                json.dumps(
                    {
                        "id": 4,
                        "method": "thread/fork",
                        "params": {"threadId": source_thread},
                    }
                )
            )
            forked = await response(ws, 4)
            fork_thread = forked["thread"]["id"]
            assert fork_thread != source_thread
            assert forked["model"] == PROFILES[ProfileName.LUNA].model
            assert forked["reasoningEffort"] == source_effort
    finally:
        server.close()
        await server.wait_closed()

    report = store.latest()
    assert report["threadStates"][source_thread]["profile"] == "luna"
    assert report["threadStates"][fork_thread]["profile"] == "luna"
    assert report["metadata"]["forkedThreads"] == [
        {
            "sourceThreadId": source_thread,
            "threadId": fork_thread,
            "inheritedProfile": "luna",
            "inheritedEffort": source_effort,
        }
    ]


@pytest.mark.asyncio
async def test_real_temporary_directive_restores_route_for_next_task(tmp_path):
    binary = Path(os.environ["ROUTER_LIVE_CODEX"])
    store = ReportStore(tmp_path / "reports")
    server = await make_server(binary, "127.0.0.1", 0, store)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            await start_client(ws)
            await ws.send(
                json.dumps(
                    {
                        "id": 2,
                        "method": "thread/start",
                        "params": {
                            "model": PROFILES[ProfileName.SOL].model,
                            "cwd": str(tmp_path),
                            "sandbox": "read-only",
                            "approvalPolicy": "never",
                        },
                    }
                )
            )
            started = await response(ws, 2)
            thread = started["thread"]["id"]
            original_effort = started["reasoningEffort"]
            for request_id, prompt in (
                (3, "l1~ Reply READY. Use no tools."),
                (4, "Reply READY again. Use no tools."),
            ):
                await ws.send(
                    json.dumps(
                        {
                            "id": request_id,
                            "method": "turn/start",
                            "params": {
                                "threadId": thread,
                                "input": [{"type": "text", "text": prompt}],
                            },
                        }
                    )
                )
                await response(ws, request_id)
                assert await complete_turn(ws) == 0
    finally:
        server.close()
        await server.wait_closed()

    report = store.latest()
    assert [row["profile"] for row in report["responses"]] == ["luna", "sol"]
    assert report["threadStates"][thread]["profile"] == "sol"
    assert report["threadStates"][thread]["effort"] == original_effort
    assert report["threadStates"][thread]["temporary_route"] is None
    assert report["switches"][-1]["source"] == "temporary-directive-restore"


@pytest.mark.asyncio
async def test_real_cold_resume_restores_astra_and_can_switch_again(tmp_path):
    binary = Path(os.environ["ROUTER_LIVE_CODEX"])
    store = ReportStore(tmp_path / "reports")
    server = await make_server(binary, "127.0.0.1", 0, store)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            await start_client(ws)
            await ws.send(
                json.dumps(
                    {
                        "id": 2,
                        "method": "thread/start",
                        "params": {
                            "model": PROFILES[ProfileName.LUNA].model,
                            "cwd": str(tmp_path),
                            "sandbox": "read-only",
                            "approvalPolicy": "never",
                        },
                    }
                )
            )
            thread = (await response(ws, 2))["thread"]["id"]
            for request_id, marker in [(3, "l2"), (4, "a3")]:
                await ws.send(
                    json.dumps(
                        {
                            "id": request_id,
                            "method": "turn/start",
                            "params": {
                                "threadId": thread,
                                "input": [
                                    {
                                        "type": "text",
                                        "text": f"{marker} Reply READY. Use no tools.",
                                    }
                                ],
                            },
                        }
                    )
                )
                await response(ws, request_id)
                await complete_turn(ws)
    finally:
        server.close()
        await server.wait_closed()
    prior = store.latest()
    assert prior["threadStates"][thread]["profile"] == "astra"
    print(f"Saved smoke-test thread {thread}; restarting app-server.", flush=True)
    server = await make_server(binary, "127.0.0.1", 0, store)
    port = server.sockets[0].getsockname()[1]
    try:
        async with connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
            await start_client(ws)
            await ws.send(
                json.dumps({"id": 2, "method": "thread/resume", "params": {"threadId": thread}})
            )
            resumed = await response(ws, 2)
            assert resumed["model"] == PROFILES[ProfileName.ASTRA].model
            assert resumed["reasoningEffort"] == "xhigh"
            print(
                "Cold resume restored Astra xhigh; checking the persisted router tool.", flush=True
            )
            await ws.send(
                json.dumps(
                    {
                        "id": 3,
                        "method": "turn/start",
                        "params": {
                            "threadId": thread,
                            "input": [
                                {
                                    "type": "text",
                                    "text": SWITCH_PROMPT,
                                }
                            ],
                        },
                    }
                )
            )
            await response(ws, 3)
            assert await complete_turn(ws) == 1
    finally:
        server.close()
        await server.wait_closed()
    reports = store.all()
    final = next(r for r in reports if r["sessionId"] != prior["sessionId"])
    assert final["threadStates"][thread]["profile"] == "luna", final["switches"]
    assert final["switches"][-1]["outcome"] == "applied"
    assert final["responses"]
    old_ids = {r["response_id"] for r in prior["responses"]}
    assert not old_ids.intersection(r["response_id"] for r in final["responses"])
    assert {r["profile"] for r in final["responses"]} == {"astra", "luna"}
    assert all(r["usage_source"] == "codex-rollout" for r in final["responses"])
    assert all(
        r["effort"] == ("xhigh" if r["profile"] == "astra" else "low") for r in final["responses"]
    )
    (phase,) = analyze(final)["phases"]
    assert phase["observedSteps"] == sum(r["profile"] == "luna" for r in final["responses"])
