"""Read-only access to Codex's persisted tool definitions and response usage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import ROUTER_NAMESPACE, ROUTER_STATE_TOOL, ROUTER_TOOL
from .report import normalize_timestamp

TOKEN_NAMES = {
    "input_tokens": "inputTokens",
    "cached_input_tokens": "cachedInputTokens",
    "cache_write_input_tokens": "cacheWriteInputTokens",
    "output_tokens": "outputTokens",
    "reasoning_output_tokens": "reasoningOutputTokens",
    "total_tokens": "totalTokens",
}


class ThreadLease:
    """OS locks prevent concurrent routers from checkpointing the same thread."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.stream.write(b"\0")
                self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise ValueError(
                "thread is already attached to another router; close that session first"
            ) from exc

    def close(self) -> None:
        self.stream.close()


class RolloutReader:
    def __init__(self, path: Path, thread_id: str) -> None:
        self.path = path
        self.thread_id = thread_id
        self.offset = 0
        self.turn_id: str | None = None
        self.identity: tuple[int, int] | None = None
        self.router_tool_supports_effort = False
        self.router_tool_supports_route_state = False

    def read_new(self) -> list[dict[str, Any]]:
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
            raise ValueError(
                "Codex rollout was replaced or truncated; restart the router to resume"
            )
        self.identity = identity
        rows = []
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break  # The writer may still be appending this record.
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("invalid Codex rollout record")
                rows.append(row)
                self.offset = stream.tell()
        return rows

    def prime(self, require_router: bool = True) -> None:
        rows = self.read_new()
        meta = next((r.get("payload", {}) for r in rows if r.get("type") == "session_meta"), {})
        if meta.get("id") != self.thread_id:
            raise ValueError("Codex rollout does not belong to the requested thread")
        if require_router:
            tools = meta.get("dynamic_tools") or []
            router_tools = [
                nested
                for tool in tools
                if tool.get("type") == "namespace" and tool.get("name") == ROUTER_NAMESPACE
                for nested in tool.get("tools", [])
                if nested.get("name") == ROUTER_TOOL
            ]
            found = bool(router_tools)
            if not found:
                raise ValueError(
                    "this thread has no persisted self_router tool; "
                    "resume requires a thread created through codex-self-router"
                )
            schema = router_tools[0].get("inputSchema") or {}
            self.router_tool_supports_effort = "targetReasoningEffort" in (
                schema.get("properties") or {}
            )
            self.router_tool_supports_route_state = any(
                nested.get("name") == ROUTER_STATE_TOOL
                for tool in tools
                if tool.get("type") == "namespace" and tool.get("name") == ROUTER_NAMESPACE
                for nested in tool.get("tools", [])
            )
        # Historical records establish identity/tools only. They are never billed again.
        self.turn_id = None

    def events(self) -> list[dict[str, Any]]:
        events = []
        for row in self.read_new():
            payload = row.get("payload") or {}
            kind = row.get("type")
            if kind == "turn_context" or (
                kind == "event_msg" and payload.get("type") == "task_started"
            ):
                self.turn_id = payload.get("turn_id")
            elif (
                kind == "response_item"
                and payload.get("type")
                in {
                    "function_call",
                    "custom_tool_call",
                    "function_call_output",
                    "custom_tool_call_output",
                }
                and payload.get("call_id")
                and self.turn_id
            ):
                events.append(
                    {
                        "method": "rawResponseItem/completed",
                        "params": {
                            "threadId": self.thread_id,
                            "turnId": self.turn_id,
                            "item": payload,
                        },
                    }
                )
            elif kind == "token_usage_record" and payload.get("thread_id") == self.thread_id:
                usage = payload.get("usage")
                observed_at = row.get("timestamp")
                if observed_at is not None:
                    if not isinstance(observed_at, str):
                        raise ValueError("rollout timestamp must be an ISO string")
                    observed_at = normalize_timestamp(observed_at)
                events.append(
                    {
                        "method": "rawResponse/completed",
                        "params": {
                            "threadId": self.thread_id,
                            "turnId": payload.get("turn_id"),
                            "responseId": payload.get("response_id"),
                            "usage": {
                                target: usage[source]
                                for source, target in TOKEN_NAMES.items()
                                if source in usage
                            }
                            if isinstance(usage, dict)
                            else None,
                            "usageSource": "codex-rollout",
                            "observedAt": observed_at,
                        },
                    }
                )
        return events
