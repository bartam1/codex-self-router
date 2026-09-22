from __future__ import annotations

import json

import pytest
from websockets.asyncio.client import connect

import codex_self_router.proxy as proxy_module
from codex_self_router.cli import make_server
from codex_self_router.config import PROFILES, ProfileName
from codex_self_router.evaluation import analyze
from codex_self_router.report import ReportStore

FAKE_APP_SERVER = r"""#!/usr/bin/env python3
import json
import sys

models = [
    ("gpt-5.6-luna", "medium"),
    ("gpt-5.6-terra", "medium"),
    ("gpt-5.6-sol", "medium"),
    ("gpt-6-astra", "xhigh"),
]

for line in sys.stdin:
    message = json.loads(line)
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params", {})
    if request_id is None:
        continue
    if method == "initialize":
        result = {"serverInfo": {"name": "fake", "version": "1"}}
    elif method == "model/list":
        result = {
            "data": [
                {
                    "id": model,
                    "model": model,
                    "description": "x" * 100_000,
                    "supportedReasoningEfforts": [{"reasoningEffort": effort}],
                }
                for model, effort in models
            ],
            "nextCursor": None,
        }
    elif method == "thread/start":
        result = {
            "thread": {"id": "thread-1"},
            "seen": {
                "model": params.get("model"),
                "experimentalRawEvents": params.get("experimentalRawEvents"),
                "dynamicTools": params.get("dynamicTools"),
            },
        }
    elif method == "turn/start":
        result = {
            "turn": {"id": "turn-1"},
            "seen": {
                "model": params.get("model"),
                "effort": params.get("effort"),
                "text": params["input"][0]["text"],
            },
        }
    else:
        result = {}
    print(json.dumps({"id": request_id, "result": result}), flush=True)
    if method == "model/list":
        print(json.dumps({"method": "test/catalogValidated", "params": {}}), flush=True)
    if method == "turn/start":
        for event in [
            {"method": "turn/started", "params": {
                "threadId": "thread-1", "turn": {"id": "turn-1", "status": "inProgress"}}},
            {"method": "rawResponse/completed", "params": {
                "threadId": "thread-1", "turnId": "turn-1", "responseId": "response-1",
                "usage": {"inputTokens": 1000, "cachedInputTokens": 200,
                          "outputTokens": 80, "reasoningOutputTokens": 50}}},
            {"method": "item/completed", "params": {
                "threadId": "thread-1", "turnId": "turn-1",
                "item": {"type": "commandExecution", "id": "command-1", "status": "completed",
                         "exitCode": 0, "durationMs": 10, "aggregatedOutput": "not logged"}}},
            {"method": "item/completed", "params": {
                "threadId": "thread-1", "turnId": "turn-1",
                "item": {"type": "collabToolCall", "id": "collab-1", "tool": "spawn_agent",
                         "status": "completed", "senderThreadId": "thread-1",
                         "newThreadId": "child-thread", "prompt": "not logged"}}},
            {"method": "turn/completed", "params": {
                "threadId": "thread-1", "turn": {"id": "turn-1", "status": "completed"}}},
        ]:
            print(json.dumps(event), flush=True)
"""

OVERFLOW_APP_SERVER = r"""#!/usr/bin/env python3
print("x" * 1024, flush=True)
"""


async def receive_id(websocket, request_id: int) -> dict:
    while True:
        message = json.loads(await websocket.recv())
        if message.get("id") == request_id:
            return message


async def receive_method(websocket, method: str) -> dict:
    while True:
        message = json.loads(await websocket.recv())
        if message.get("method") == method:
            return message


@pytest.mark.asyncio
async def test_upstream_read_error_is_reported_before_disconnect(tmp_path, monkeypatch) -> None:
    fake = tmp_path / "overflow-codex"
    fake.write_text(OVERFLOW_APP_SERVER)
    fake.chmod(0o755)
    monkeypatch.setattr(proxy_module, "APP_SERVER_STREAM_LIMIT", 128)
    store = ReportStore(tmp_path / "reports")
    server = await make_server(fake, "127.0.0.1", 0, store)
    host, port = server.sockets[0].getsockname()[:2]

    try:
        async with connect(f"ws://{host}:{port}") as websocket:
            message = json.loads(await websocket.recv())
            assert message["method"] == "error"
            assert "could not read Codex app-server output" in message["params"]["message"]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("fixed_profile", [None, ProfileName.SOL])
async def test_websocket_proxy_injects_tools_and_applies_directive_to_first_inference(
    tmp_path,
    fixed_profile,
) -> None:
    fake = tmp_path / "fake-codex"
    fake.write_text(FAKE_APP_SERVER)
    fake.chmod(0o755)
    store = ReportStore(tmp_path / "reports")
    server = await make_server(
        fake, "127.0.0.1", 0, store, metadata={"label": "integration"}, fixed_profile=fixed_profile
    )
    host, port = server.sockets[0].getsockname()[:2]

    try:
        async with connect(f"ws://{host}:{port}") as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "clientInfo": {"name": "test", "version": "1"},
                            "capabilities": {},
                        },
                    }
                )
            )
            assert (await receive_id(websocket, 1))["result"]["serverInfo"]["name"] == "fake"
            await websocket.send(json.dumps({"method": "initialized", "params": {}}))
            await receive_method(websocket, "test/catalogValidated")

            await websocket.send(json.dumps({"id": 2, "method": "thread/start", "params": {}}))
            thread = await receive_id(websocket, 2)
            assert thread["result"]["seen"]["model"] == PROFILES[ProfileName.SOL].model
            assert thread["result"]["seen"]["experimentalRawEvents"] is True
            if fixed_profile is None:
                assert thread["result"]["seen"]["dynamicTools"][-1]["name"] == "self_router"
            else:
                assert thread["result"]["seen"]["dynamicTools"] == []

            await websocket.send(
                json.dumps(
                    {
                        "id": 3,
                        "method": "turn/start",
                        "params": {
                            "threadId": "thread-1",
                            "model": PROFILES[ProfileName.ASTRA].model,
                            "input": [
                                {
                                    "type": "text",
                                    "text": "#3 design it"
                                    if fixed_profile is None
                                    else "design it",
                                }
                            ],
                        },
                    }
                )
            )
            turn = await receive_id(websocket, 3)
            expected = fixed_profile or ProfileName.ASTRA
            assert turn["result"]["seen"] == {
                "model": PROFILES[expected].model,
                "effort": PROFILES[expected].effort,
                "text": "design it",
            }
            await receive_method(websocket, "turn/completed")
    finally:
        server.close()
        await server.wait_closed()
    report = store.latest()
    assert report["endedAt"] is not None
    collab = next(item for item in report["measurements"]["items"] if item["item_id"] == "collab-1")
    assert collab["is_subagent_delegation"] is True
    assert collab["new_thread_id"] == "child-thread"
    assert "not logged" not in json.dumps(report["measurements"])
    (task,) = analyze(report)["tasks"]
    assert task["routingDelegationClass"] == ("both" if fixed_profile is None else "subagent-only")
    assert report["schemaVersion"] == 2
    assert report["metadata"]["label"] == "integration"
    assert report["responses"][0]["profile"] == expected.value
    assert report["responses"][0]["usage"]["reasoningOutputTokens"] == 50
    assert report["measurements"]["turns"][0]["status"] == "completed"
    assert report["measurements"]["items"][0]["reported_duration_ms"] == 10
    assert "not logged" not in json.dumps(report)
    assert len(list((tmp_path / "reports").glob("*.events.jsonl"))) == 1
