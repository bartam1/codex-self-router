"""Read-only app-server diagnostics and shared model-catalog validation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from . import __version__
from .config import PROFILES, get_default_profile

# Newline-delimited model/list messages can exceed asyncio's default 64 KiB limit.
APP_SERVER_STREAM_LIMIT = 16 * 1024 * 1024
PROBE_THREAD_ID = "00000000-0000-0000-0000-000000000000"


async def read_model_catalog(
    request: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    entries = []
    cursors: set[str] = set()
    params: dict[str, Any] = {"limit": 100, "includeHidden": True}
    while True:
        response = await request("model/list", params)
        if "error" in response:
            raise RuntimeError(response["error"].get("message", "model/list failed"))
        result = response.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise RuntimeError("model/list returned an invalid catalog")
        if any(not isinstance(entry, dict) for entry in result["data"]):
            raise RuntimeError("model/list returned an invalid model entry")
        entries.extend(result["data"])
        cursor = result.get("nextCursor")
        if cursor is None:
            return {"result": {"data": entries, "nextCursor": None}}
        if not isinstance(cursor, str) or not cursor or cursor in cursors:
            raise RuntimeError("model/list returned an invalid or repeated cursor")
        cursors.add(cursor)
        params = {**params, "cursor": cursor}


def inspect_model_catalog(response: dict[str, Any]) -> list[str]:
    """Return problems with the configured models and advertised reasoning efforts."""
    entries = response.get("result", {}).get("data", [])
    by_model = {item.get("model") or item.get("id"): item for item in entries}
    problems: list[str] = []
    for profile in PROFILES.values():
        item = by_model.get(profile.model)
        if item is None:
            problems.append(f"model unavailable: {profile.model}")
            continue
        efforts = item.get("supportedReasoningEfforts") or []
        effort_names = {
            effort.get("reasoningEffort") if isinstance(effort, dict) else effort
            for effort in efforts
        }
        for effort in profile.allowed_efforts:
            if effort not in effort_names:
                problems.append(f"{profile.model} does not advertise effort {effort}")
    return problems


async def probe_app_server(codex_bin: Path) -> dict[str, Any]:
    """Exercise the handshake, settings API, and catalog without creating a thread or turn."""
    process = await asyncio.create_subprocess_exec(
        str(codex_bin),
        "--enable",
        "step_model_switching",
        "app-server",
        "--listen",
        "stdio://",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=APP_SERVER_STREAM_LIMIT,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        raise RuntimeError("could not open app-server protocol streams")

    async def drain_stderr() -> None:
        while await process.stderr.read(4096):
            pass

    stderr_task = asyncio.create_task(drain_stderr(), name="doctor-stderr")
    request_id = 0

    async def request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal request_id
        request_id += 1
        payload = {"id": request_id, "method": method, "params": params}
        process.stdin.write(json.dumps(payload).encode() + b"\n")
        await process.stdin.drain()
        while line := await process.stdout.readline():
            response = json.loads(line)
            if not isinstance(response, dict):
                raise RuntimeError("app-server returned an invalid protocol message")
            if response.get("id") == request_id and "method" not in response:
                return response
        raise RuntimeError("app-server exited before completing the capability probe")

    try:
        async with asyncio.timeout(30):
            initialized = await request(
                "initialize",
                {
                    "clientInfo": {"name": "codex_self_router_doctor", "version": __version__},
                    "capabilities": {"experimentalApi": True},
                },
            )
            if "error" in initialized or "result" not in initialized:
                raise RuntimeError("app-server initialization failed")
            process.stdin.write(b'{"method":"initialized"}\n')
            await process.stdin.drain()
            profile = PROFILES[get_default_profile()]
            settings = await request(
                "turn/settings/update",
                {
                    "threadId": PROBE_THREAD_ID,
                    "turnId": PROBE_THREAD_ID,
                    "model": profile.model,
                    "effort": profile.effort,
                },
            )
            # A missing-thread response proves that dispatch and the enabled feature accepted
            # the method. An unknown method/disabled feature must not pass as "compatible".
            error = settings.get("error") or {}
            if not str(error.get("message", "")).startswith("thread not found:"):
                raise RuntimeError(
                    "app-server settings-update probe failed: "
                    + str(error.get("message", "unexpected response"))
                )
            return await read_model_catalog(request)
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
        await stderr_task
