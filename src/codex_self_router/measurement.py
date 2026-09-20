"""Protocol observations only: no prompt text, command output, or inferred task success."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .report import utc_now

TOOL_ITEMS = {
    "commandExecution",
    "mcpToolCall",
    "dynamicToolCall",
    "fileChange",
    "webSearch",
    "collabAgentToolCall",
}


class Measurements:
    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self.emit = emit
        self.origin = time.monotonic()
        self.sequence = 0
        self.turns: dict[tuple[str, str], dict[str, Any]] = {}
        self.items: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.waits: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.continuations: dict[str, str] = {}
        self.interruptions: dict[tuple[str, str], str] = {}
        self.resumed_tasks: dict[str, dict[str, Any]] = {}

    def event(self, kind: str, **fields: Any) -> dict[str, Any]:
        self.sequence += 1
        event = {
            "sequence": self.sequence,
            "timestamp": utc_now(),
            "elapsed_ms": round((time.monotonic() - self.origin) * 1000, 3),
            "kind": kind,
            **fields,
        }
        self.events.append(event)
        self.emit(event)
        return event

    def start_turn(self, thread_id: str, turn_id: str) -> dict[str, Any]:
        key = (thread_id, turn_id)
        if key not in self.turns:
            parent = self.continuations.pop(thread_id, None)
            resumed = self.resumed_tasks.pop(thread_id, None) if parent is None else None
            previous = self.turns.get((thread_id, parent or ""), {})
            event = self.event("turn_started", thread_id=thread_id, turn_id=turn_id)
            self.turns[key] = {
                "thread_id": thread_id,
                "turn_id": turn_id,
                "task_id": previous.get("task_id", parent or turn_id),
                "parent_turn_id": parent,
                "source": "router-continuation" if parent else "user-or-upstream",
                "started_at": event["timestamp"],
                "start_ms": event["elapsed_ms"],
                "ended_at": None,
                "end_ms": None,
                "status": "inProgress",
                "interruption_source": None,
            }
            if resumed:
                self.turns[key].update(
                    task_id=resumed["task_id"],
                    parent_turn_id=resumed["turn_id"],
                    source="session-resume",
                    resumed_from_session_id=resumed["session_id"],
                )
        return self.turns[key]

    def observe(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        params = message.get("params") or {}
        thread_id = str(params.get("threadId", ""))
        turn_id = str(params.get("turnId") or params.get("turn", {}).get("id", ""))
        if method == "turn/started":
            self.start_turn(thread_id, turn_id)
        elif method == "turn/completed":
            # A missing start is unknown, not a zero-duration successful task.
            turn = self.turns.get((thread_id, turn_id))
            event = self.event(
                "turn_completed",
                thread_id=thread_id,
                turn_id=turn_id,
                status=params.get("turn", {}).get("status"),
            )
            if turn is not None:
                turn.update(
                    ended_at=event["timestamp"],
                    end_ms=event["elapsed_ms"],
                    status=event["status"],
                    interruption_source=self.interruptions.pop((thread_id, turn_id), None),
                )
        elif method in {"item/started", "item/completed"}:
            item = params.get("item") or {}
            kind = item.get("type")
            if kind not in TOOL_ITEMS and kind != "contextCompaction":
                return
            key = (thread_id, turn_id, str(item.get("id", "")))
            if not key[2]:
                return
            record = self.items.get(key)
            if record is not None and record.get("end_ms") is not None:
                return
            event = self.event(
                "item_started" if method == "item/started" else "item_completed",
                thread_id=thread_id,
                turn_id=turn_id,
                item_id=key[2],
                item_type=kind,
                status=item.get("status"),
                exit_code=item.get("exitCode"),
                success=item.get("success"),
                duration_ms=item.get("durationMs"),
            )
            if record is None:
                record = {
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "item_id": key[2],
                    "item_type": kind,
                    "start_ms": None,
                    "end_ms": None,
                }
                self.items[key] = record
            if method == "item/started":
                record["start_ms"] = event["elapsed_ms"]
            else:
                record.update(
                    end_ms=event["elapsed_ms"],
                    status=item.get("status"),
                    exit_code=item.get("exitCode"),
                    success=item.get("success"),
                    reported_duration_ms=item.get("durationMs"),
                    failed=(
                        item.get("status") in {"failed", "declined"}
                        or item.get("success") is False
                        or item.get("exitCode") not in {None, 0}
                    ),
                )
        elif method == "error":
            self.event(
                "upstream_error",
                thread_id=thread_id,
                turn_id=turn_id,
                will_retry=params.get("willRetry"),
                error_code=(params.get("error") or {}).get("codexErrorInfo"),
            )
        elif method == "serverRequest/resolved":
            self.resolve_wait(str(params.get("requestId")))
        elif (
            "id" in message
            and method
            and params.get("isBlocking") is not False
            and (
                "requestApproval" in method
                or method
                in {
                    "item/tool/requestUserInput",
                    "item/permissions/requestApproval",
                    "mcpServer/elicitation/request",
                }
            )
        ):
            self.start_wait(str(message["id"]), thread_id, turn_id, method)

    def start_wait(self, request_id: str, thread_id: str, turn_id: str, kind: str) -> None:
        key = f"{kind}:{request_id}"
        if key in self.waits:
            return
        event = self.event(
            "wait_started",
            request_id=request_id,
            thread_id=thread_id,
            turn_id=turn_id,
            request_kind=kind,
        )
        self.waits[key] = {
            "request_id": request_id,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "kind": kind,
            "start_ms": event["elapsed_ms"],
            "end_ms": None,
        }

    def resolve_wait(self, request_id: str) -> None:
        for wait in self.waits.values():
            if wait["request_id"] == request_id and wait["end_ms"] is None:
                event = self.event("wait_resolved", request_id=request_id)
                wait["end_ms"] = event["elapsed_ms"]

    def close(self) -> None:
        self.event("session_closed")
        for turn in self.turns.values():
            if turn["end_ms"] is None:
                turn["status"] = "disconnected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": list(self.turns.values()),
            "items": list(self.items.values()),
            "waits": list(self.waits.values()),
            "events": self.events,
        }
