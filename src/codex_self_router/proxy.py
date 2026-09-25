from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection

from . import __version__
from .config import (
    PROFILES,
    ProfileName,
    SwitchApproval,
    config_snapshot,
    get_agent_switching_enabled,
    get_default_profile,
    get_switch_approval,
    merge_routing_policy,
    profile_for_model,
    routing_policy,
    switch_requires_approval,
)
from .measurement import Measurements
from .report import ReportStore, SessionReport, SwitchRecord, UsageRecord, parse_timestamp, utc_now
from .resume import RolloutReader, ThreadLease
from .transform import (
    TurnRoute,
    enable_experimental_api,
    is_route_state_tool_call,
    is_router_tool_call,
    parse_switch_arguments,
    prepare_thread_start,
    prepare_turn_start,
    prepare_turn_steer,
)

LOG = logging.getLogger("codex_self_router")

# App-server speaks newline-delimited JSON, and model/list can exceed asyncio's
# default 64 KiB StreamReader limit as the model catalog grows.
APP_SERVER_STREAM_LIMIT = 16 * 1024 * 1024
STEP_UPDATE_COMPATIBILITY_PREFIX = "the destination changes "


def _continuation_tool_output(
    previous: ProfileName,
    previous_effort: str,
    target: ProfileName,
    target_effort: str,
) -> str:
    return (
        "Model/effort switch authorized and applied: "
        f"{previous.value}/{previous_effort} to {target.value}/{target_effort}. "
        "This is an automatic continuation of the existing user instruction, not a new request. "
        "The preceding interruption was only the router creating a safety-compatible turn "
        "boundary; it was not a user cancellation. Preserve and use the full conversation "
        "context, then continue immediately.\n\n"
        "Do not call request_user_input or request_user_input_async for optional clarification "
        "in this continuation. Make and state reasonable assumptions, then complete the requested "
        "work. Ask a question only if a missing answer genuinely prevents safe or correct progress."
    )


def _id_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _rpc_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"id": request_id, "result": result}


def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"id": request_id, "error": {"code": code, "message": message}}


def _tool_result(request_id: Any, text: str, *, success: bool) -> dict[str, Any]:
    return _rpc_result(
        request_id,
        {"contentItems": [{"type": "inputText", "text": text}], "success": success},
    )


def _is_async_question_item(message: dict[str, Any]) -> bool:
    if message.get("method") not in {"item/started", "item/completed"}:
        return False
    item = message.get("params", {}).get("item", {})
    return (
        isinstance(item, dict)
        and item.get("type") == "agentMessage"
        and item.get("delivery") == "async"
        and bool(item.get("questions"))
    )


@dataclass(slots=True)
class ThreadState:
    profile: ProfileName
    active_turn_id: str | None = None
    collaboration_mode: dict[str, Any] | None = None
    resumed: bool = False
    effort: str | None = None
    client_profile: ProfileName | None = None
    client_effort: str | None = None
    temporary_restore_profile: ProfileName | None = None
    temporary_restore_effort: str | None = None
    temporary_task_id: str | None = None
    explicit_route_task_id: str | None = None


@dataclass(slots=True)
class RequestContext:
    method: str
    thread_id: str | None = None
    profile: ProfileName | None = None
    previous_profile: ProfileName | None = None
    previous_effort: str | None = None
    route: TurnRoute | None = None
    submitted_event: dict[str, Any] | None = None
    developer_instructions: str | None = None
    collaboration_mode: dict[str, Any] | None = None
    effort: str | None = None
    client_route_changed: bool = False


class Bridge:
    """Bidirectional JSON-RPC bridge with model-routing interception."""

    def __init__(
        self,
        websocket: ServerConnection,
        *,
        codex_bin: Path,
        report_store: ReportStore | None = None,
        metadata: dict[str, Any] | None = None,
        fixed_profile: ProfileName | None = None,
        switch_approval: SwitchApproval | None = None,
    ) -> None:
        self.websocket = websocket
        self.codex_bin = codex_bin
        self.report_store = report_store or ReportStore()
        self.report = SessionReport(session_id=str(uuid.uuid4()))
        self.fixed_profile = fixed_profile
        self.switch_approval = switch_approval or get_switch_approval()
        self.agent_switching_enabled = get_agent_switching_enabled()
        self.report.metadata = {
            "routerVersion": __version__,
            "codexBinary": str(codex_bin),
            "policySha256": hashlib.sha256(routing_policy().encode()).hexdigest(),
            "routingMode": f"fixed-{fixed_profile.value}" if fixed_profile else "auto",
            "pricingSource": "configured API-equivalent rates; not provider billing data",
            "routerConfig": config_snapshot(),
            **(metadata or {}),
        }
        self.measurements = Measurements(
            lambda event: self.report_store.append_event(self.report.session_id, event)
        )
        self.response_ids: set[tuple[str, str, str]] = set()
        self.switch_clocks: dict[str, dict[str, float]] = {}
        self.rollouts: dict[str, RolloutReader] = {}
        self.resume_locks: dict[str, asyncio.Lock] = {}
        self.rollout_failed: set[str] = set()
        self.thread_leases: dict[str, ThreadLease] = {}
        self.profile_history: dict[tuple[str, str], list[tuple[str, ProfileName, str]]] = {}
        self.process: asyncio.subprocess.Process | None = None
        self.client_send_lock = asyncio.Lock()
        self.upstream_send_lock = asyncio.Lock()
        self.client_requests: dict[str, RequestContext] = {}
        self.internal_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.client_prompts: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.threads: dict[str, ThreadState] = {}
        self.turn_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self.response_sequence: dict[tuple[str, str], int] = {}
        self.raw_call_sequence: dict[str, tuple[tuple[str, str], int]] = {}
        self.open_raw_calls: set[str] = set()
        self.response_condition = asyncio.Condition()
        self.turn_completion_waiters: dict[tuple[str, str], asyncio.Future[dict[str, Any]]] = {}
        self.automatic_continuation_turns: set[tuple[str, str]] = set()
        self.background: set[asyncio.Task[Any]] = set()
        self.catalog_validation_started = False
        self.closed = False

    async def run(self) -> None:
        self.measurements.event("session_started", metadata=self.report.metadata)
        self._save_report()
        self.process = await asyncio.create_subprocess_exec(
            str(self.codex_bin),
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
        assert self.process.stdout is not None
        stderr_task = asyncio.create_task(self._drain_stderr(), name="app-server-stderr")
        client_read = asyncio.create_task(self.websocket.recv(), name="client-read")
        upstream_read = asyncio.create_task(self.process.stdout.readline(), name="upstream-read")
        try:
            while True:
                done, _ = await asyncio.wait(
                    {client_read, upstream_read}, return_when=asyncio.FIRST_COMPLETED
                )
                if client_read in done:
                    try:
                        payload = client_read.result()
                    except Exception:
                        break
                    await self._handle_client_payload(payload)
                    client_read = asyncio.create_task(self.websocket.recv(), name="client-read")

                if upstream_read in done:
                    try:
                        line = upstream_read.result()
                    except Exception as exc:
                        with contextlib.suppress(Exception):
                            await self._send_client(
                                {
                                    "method": "error",
                                    "params": {
                                        "message": (
                                            "could not read Codex app-server output: "
                                            f"{type(exc).__name__}: {exc}"
                                        )
                                    },
                                }
                            )
                        break
                    if not line:
                        code = await self.process.wait()
                        if code != 0:
                            await self._send_client(
                                {
                                    "method": "error",
                                    "params": {
                                        "message": f"codex app-server exited with status {code}"
                                    },
                                }
                            )
                        break
                    await self._handle_upstream_payload(line.decode("utf-8"))
                    upstream_read = asyncio.create_task(
                        self.process.stdout.readline(), name="upstream-read"
                    )
        finally:
            self.closed = True
            client_read.cancel()
            upstream_read.cancel()
            stderr_task.cancel()
            for task in self.background:
                task.cancel()
            for future in [*self.internal_requests.values(), *self.client_prompts.values()]:
                if not future.done():
                    future.cancel()
            for future in self.turn_completion_waiters.values():
                if not future.done():
                    future.cancel()
            await self._stop_process()
            # The writer flushes on shutdown. Capture its final records before closing the report.
            for thread_id in self.rollouts:
                await self._drain_rollout(thread_id)
            for switch in self.report.switches:
                if switch.outcome == "pending":
                    self._record_agent_switch(
                        switch.thread_id,
                        switch.turn_id or "",
                        ProfileName(switch.from_profile),
                        ProfileName(switch.to_profile),
                        "cancelled",
                        detail="Session ended before the routing decision completed.",
                        origin_turn_id=switch.origin_turn_id,
                    )
            self.report.ended_at = utc_now()
            self.measurements.close()
            self._save_report()
            for lease in self.thread_leases.values():
                lease.close()

    def _save_report(self) -> None:
        self.report.measurements = self.measurements.to_dict()
        for thread_id, snapshot in self.report.thread_states.items():
            turns = [t for t in self.measurements.turns.values() if t["thread_id"] == thread_id]
            if turns:
                last = max(turns, key=lambda t: t["start_ms"])
                snapshot["last_task"] = {
                    key: last.get(key)
                    for key in (
                        "task_id",
                        "turn_id",
                        "status",
                        "interruption_source",
                    )
                }
            snapshot["updated_at"] = utc_now()
        self.report_store.save(self.report)

    def _checkpoint_thread(self, thread_id: str, instructions: str | None = None) -> None:
        self._claim_thread(thread_id)
        state = self.threads[thread_id]
        effort = state.effort or PROFILES[state.profile].effort
        if state.active_turn_id:
            history = self.profile_history.setdefault((thread_id, state.active_turn_id), [])
            if not history or history[-1][1:] != (state.profile, effort):
                history.append((utc_now(), state.profile, effort))
        prior = self.report.thread_states.get(thread_id, {})
        self.report.thread_states[thread_id] = {
            **prior,
            "profile": state.profile.value,
            "effort": effort,
            "collaboration_mode": state.collaboration_mode,
            "developer_instructions": instructions
            if instructions is not None
            else prior.get("developer_instructions"),
            "temporary_route": (
                {
                    "restore_profile": state.temporary_restore_profile.value,
                    "restore_effort": state.temporary_restore_effort,
                    "task_id": state.temporary_task_id,
                }
                if state.temporary_restore_profile is not None
                else None
            ),
            "explicit_route_task_id": state.explicit_route_task_id,
            "routing_mode": self.report.metadata["routingMode"],
            "updated_at": utc_now(),
        }
        self._save_report()

    def _claim_thread(self, thread_id: str) -> None:
        if thread_id not in self.thread_leases:
            filename = self.report_store.path_for(thread_id).with_suffix(".lock").name
            self.thread_leases[thread_id] = ThreadLease(
                self.report_store.report_dir / "threads" / filename
            )

    async def _drain_rollout(self, thread_id: str) -> None:
        if thread_id in self.rollout_failed:
            return
        try:
            for event in self.rollouts[thread_id].events():
                if event["method"] == "rawResponseItem/completed":
                    await self._observe_raw_item(event)
                else:
                    await self._observe_raw_response(event)
        except (OSError, TypeError, ValueError) as exc:
            self.rollout_failed.add(thread_id)
            note = f"Resume usage capture failed for {thread_id}: {exc}"
            self.report.notes.append(note)
            self.measurements.event("resume_usage_error", thread_id=thread_id)
            self._save_report()
            if not self.closed:
                await self._send_client({"method": "warning", "params": {"message": note}})

    async def _watch_rollout(self, thread_id: str) -> None:
        while not self.closed and thread_id not in self.rollout_failed:
            await self._drain_rollout(thread_id)
            await asyncio.sleep(0.1)

    async def _resume_thread(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        params = copy.deepcopy(message.get("params") or {})
        thread_id = str(params.get("threadId", ""))
        lock = self.resume_locks.setdefault(thread_id, asyncio.Lock())
        async with lock:
            legacy_tool_warning = False
            try:
                if params.get("history") is not None or params.get("path"):
                    raise ValueError(
                        "resume by threadId; history/path overrides cannot restore router state"
                    )
                self.report_store.path_for(thread_id)  # Validate before any filesystem lookup.
                self._claim_thread(thread_id)
                existing = self.threads.get(thread_id)
                checkpoint = (
                    self.report_store.thread_checkpoint(thread_id) if not existing else None
                )
                if (
                    checkpoint
                    and checkpoint.get("routing_mode") != self.report.metadata["routingMode"]
                ):
                    raise ValueError(
                        "resume with the same --fixed-profile mode as the original thread"
                    )
                if self.fixed_profile and not checkpoint and not existing:
                    raise ValueError("fixed-profile resume requires a saved router session")
                if not existing:
                    read = await self._internal_request("thread/read", {"threadId": thread_id})
                    if "error" in read:
                        await self._send_client({"id": request_id, "error": read["error"]})
                        return
                    thread = read.get("result", {}).get("thread", {})
                    if thread.get("id") != thread_id or not thread.get("path"):
                        raise ValueError(
                            "resume requires a durable local Codex thread with a rollout path"
                        )
                    reader = RolloutReader(Path(thread["path"]), thread_id)
                    reader.prime(require_router=self.fixed_profile is None)
                explicit = profile_for_model(params.get("model"))
                if params.get("model") and explicit is None:
                    raise ValueError(f"unsupported resume model: {params['model']}")
                restored = ProfileName(checkpoint["profile"]) if checkpoint else None
                selected = (
                    self.fixed_profile or explicit or (existing.profile if existing else restored)
                )
                if selected:
                    params["model"] = PROFILES[selected].model
                    effort = PROFILES[selected].effort
                    if explicit is None:
                        if existing:
                            effort = existing.effort or effort
                        elif checkpoint and selected == restored:
                            effort = checkpoint.get("effort") or effort
                    params.setdefault("config", {})
                    if params["config"] is None:
                        params["config"] = {}
                    if self.fixed_profile:
                        params["config"]["model_reasoning_effort"] = effort
                    else:
                        params["config"].setdefault("model_reasoning_effort", effort)
                instructions = params.get("developerInstructions")
                if instructions is None and checkpoint:
                    instructions = checkpoint.get("developer_instructions")
                if self.fixed_profile is None:
                    instructions = merge_routing_policy(instructions)
                if instructions is not None:
                    params["developerInstructions"] = instructions
                response = await self._internal_request("thread/resume", params)
                if "error" in response:
                    await self._send_client({"id": request_id, "error": response["error"]})
                    return
                result = response.get("result", {})
                profile = profile_for_model(result.get("model"))
                if result.get("thread", {}).get("id") != thread_id or profile is None:
                    raise ValueError("app-server resumed an unexpected thread or unsupported model")
                if selected and profile != selected:
                    raise ValueError("app-server did not restore the requested model profile")
                state = ThreadState(
                    profile,
                    resumed=True,
                    effort=result.get("reasoningEffort"),
                    client_profile=profile,
                    client_effort=result.get("reasoningEffort"),
                )
                state.collaboration_mode = copy.deepcopy(
                    existing.collaboration_mode
                    if existing
                    else (checkpoint or {}).get("collaboration_mode")
                )
                if existing:
                    state.active_turn_id = existing.active_turn_id
                    state.temporary_restore_profile = existing.temporary_restore_profile
                    state.temporary_restore_effort = existing.temporary_restore_effort
                    state.temporary_task_id = existing.temporary_task_id
                    state.explicit_route_task_id = existing.explicit_route_task_id
                else:
                    temporary = (checkpoint or {}).get("temporary_route")
                    if temporary:
                        state.temporary_restore_profile = ProfileName(temporary["restore_profile"])
                        state.temporary_restore_effort = str(temporary["restore_effort"])
                        state.temporary_task_id = temporary.get("task_id")
                    state.explicit_route_task_id = (checkpoint or {}).get("explicit_route_task_id")
                self.threads[thread_id] = state
                if not existing:
                    identities, sessions = self.report_store.thread_history(thread_id)
                    self.response_ids.update(identities)
                    link = {
                        "threadId": thread_id,
                        "previousSessionIds": sessions,
                        "restoredProfile": profile.value,
                        "previousTask": (checkpoint or {}).get("last_task"),
                    }
                    self.report.metadata.setdefault("resumedThreads", []).append(link)
                    if checkpoint and not self.report.metadata.get("label"):
                        self.report.metadata["label"] = checkpoint.get("metadata", {}).get("label")
                    last = (checkpoint or {}).get("last_task")
                    if last and (
                        last.get("status") in {"inProgress", "disconnected"}
                        or last.get("interruption_source") == "router"
                    ):
                        self.measurements.resumed_tasks[thread_id] = {
                            **last,
                            "session_id": checkpoint["previous_session_id"],
                        }
                    if last:
                        # Closing without starting a turn must not erase the task
                        # linkage needed by a subsequent resume.
                        self.report.thread_states[thread_id] = {"last_task": last}
                    self.rollouts[thread_id] = reader
                    self._start_background(self._watch_rollout(thread_id), "resume-usage")
                    if not reader.router_tool_supports_effort and self.fixed_profile is None:
                        legacy_tool_warning = True
                        self.report.metadata.setdefault("legacyToolSchemaThreads", []).append(
                            thread_id
                        )
                        self.report.notes.append(
                            "Resumed thread uses a pre-0.4 router tool schema; explicit directives "
                            "support model+effort, but agent effort-only requests are unavailable."
                        )
                self.measurements.event(
                    "thread_resumed", thread_id=thread_id, profile=profile.value
                )
                self._checkpoint_thread(thread_id, instructions)
                # History stays inside the response, rather than becoming new usage events.
                await self._send_client({"id": request_id, "result": result})
                if legacy_tool_warning:
                    await self._send_client(
                        {
                            "method": "warning",
                            "params": {
                                "message": (
                                    "This resumed thread has the pre-0.4 router tool schema. "
                                    "New explicit directives work, but start a new router thread "
                                    "for agent-requested effort-only changes."
                                )
                            },
                        }
                    )
            except (ValueError, OSError, TimeoutError, RuntimeError) as exc:
                await self._send_client(
                    _rpc_error(request_id, -32602, f"cannot resume thread: {exc}")
                )
            finally:
                if thread_id not in self.threads and thread_id in self.thread_leases:
                    self.thread_leases.pop(thread_id).close()

    async def _handle_client_payload(self, payload: str | bytes) -> None:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        try:
            message = json.loads(payload)
        except json.JSONDecodeError as exc:
            await self._send_client(_rpc_error(None, -32700, f"invalid JSON: {exc}"))
            return
        if not isinstance(message, dict):
            await self._send_client(_rpc_error(None, -32600, "JSON-RPC message must be an object"))
            return

        request_id = message.get("id")
        submitted_event = None
        request_key = _id_key(request_id) if "id" in message else None
        if method := message.get("method"):
            method = str(method)
        if method is None and request_key is not None:
            self.measurements.resolve_wait(str(request_id))
            self._save_report()
        if method in {"turn/interrupt", "turn/steer", "turn/start"}:
            params = message.get("params", {})
            state = self.threads.get(str(params.get("threadId", "")))
            measured_turn = params.get("turnId") or params.get("expectedTurnId")
            if method != "turn/start" and not measured_turn and state:
                measured_turn = state.active_turn_id
            submitted_event = self.measurements.event(
                "client_request",
                method=method,
                thread_id=params.get("threadId"),
                turn_id=measured_turn,
            )
            if method == "turn/interrupt":
                self.measurements.interruptions[
                    (str(params.get("threadId", "")), str(measured_turn or ""))
                ] = "user"
            self._save_report()
        if method is None and request_key is not None and request_key in self.client_prompts:
            future = self.client_prompts.pop(request_key)
            if not future.done():
                future.set_result(message)
            return

        if method == "thread/resume" and "id" in message:
            self._start_background(self._resume_thread(message), "thread-resume")
            return

        if self.fixed_profile and method == "turn/settings/update":
            await self._send_client(
                _rpc_error(
                    request_id,
                    -32602,
                    "model settings are locked for this fixed-profile evaluation run",
                )
            )
            return

        context = RequestContext(method=str(method or ""), submitted_event=submitted_event)
        if method == "initialize":
            message = enable_experimental_api(message)
        elif method == "thread/start":
            try:
                message, context.profile = prepare_thread_start(message, self.fixed_profile)
                context.developer_instructions = message["params"].get("developerInstructions")
            except ValueError as exc:
                await self._send_client(_rpc_error(request_id, -32602, str(exc)))
                return
        elif method == "thread/fork":
            message = copy.deepcopy(message)
            params = message.setdefault("params", {})
            source_thread_id = str(params.get("threadId", ""))
            source = self.threads.get(source_thread_id)
            if source is None:
                await self._send_client(
                    _rpc_error(request_id, -32602, "router has no state for the source thread")
                )
                return
            profile = self.fixed_profile or source.profile
            effort = (
                PROFILES[self.fixed_profile].effort
                if self.fixed_profile
                else source.effort or PROFILES[source.profile].effort
            )
            params["model"] = PROFILES[profile].model
            config = params.get("config")
            if not isinstance(config, dict):
                config = {}
                params["config"] = config
            config["model_reasoning_effort"] = effort
            instructions = params.get("developerInstructions")
            if instructions is not None and self.fixed_profile is None:
                instructions = merge_routing_policy(instructions)
                params["developerInstructions"] = instructions
            if instructions is None:
                instructions = self.report.thread_states.get(source_thread_id, {}).get(
                    "developer_instructions"
                )
            context.thread_id = source_thread_id
            context.profile = profile
            context.effort = effort
            context.developer_instructions = instructions
            context.collaboration_mode = copy.deepcopy(source.collaboration_mode)
        elif method == "turn/start":
            thread_id = str(message.get("params", {}).get("threadId", ""))
            context.thread_id = thread_id
            context.previous_profile = self.threads.get(
                thread_id, ThreadState(get_default_profile())
            ).profile
            previous_state = self.threads.get(thread_id)
            context.previous_effort = (
                previous_state.effort or PROFILES[context.previous_profile].effort
                if previous_state
                else PROFILES[context.previous_profile].effort
            )
            try:
                current = self.threads.get(thread_id)
                if current is not None:
                    params = message.setdefault("params", {})
                    advertised_profile = profile_for_model(params.get("model"))
                    advertised_effort = (
                        str(params["effort"]) if params.get("effort") is not None else None
                    )
                    baseline_profile = current.client_profile or current.profile
                    baseline_effort = (
                        current.client_effort or current.effort or PROFILES[baseline_profile].effort
                    )
                    context.client_route_changed = bool(
                        (advertised_profile and advertised_profile != baseline_profile)
                        or (advertised_effort and advertised_effort != baseline_effort)
                    )
                    if not context.client_route_changed:
                        params["model"] = PROFILES[current.profile].model
                        params["effort"] = current.effort or PROFILES[current.profile].effort
                message, context.route = prepare_turn_start(
                    message,
                    self.fixed_profile,
                    current.profile if current else None,
                    current.effort if current else None,
                )
                context.effort = message["params"]["effort"]
            except ValueError as exc:
                await self._send_client(_rpc_error(request_id, -32602, str(exc)))
                return
            context.profile = context.route.profile
            state = self.threads.setdefault(thread_id, ThreadState(context.profile))
            state.profile = context.profile
            if (
                state.resumed
                and state.collaboration_mode
                and not message["params"].get("collaborationMode")
            ):
                mode = copy.deepcopy(state.collaboration_mode)
                mode.setdefault("settings", {})["model"] = PROFILES[state.profile].model
                mode["settings"]["reasoning_effort"] = message["params"]["effort"]
                message["params"]["collaborationMode"] = mode
            collaboration_mode = message.get("params", {}).get("collaborationMode")
            if isinstance(collaboration_mode, dict):
                state.collaboration_mode = copy.deepcopy(collaboration_mode)
        elif method == "turn/settings/update":
            params = message.get("params", {})
            thread_id = str(params.get("threadId", ""))
            state = self.threads.get(thread_id)
            if state is None:
                await self._send_client(
                    _rpc_error(request_id, -32602, "router has no state for this thread")
                )
                return
            requested_profile = profile_for_model(params.get("model"))
            if params.get("model") and requested_profile is None:
                await self._send_client(
                    _rpc_error(request_id, -32602, f"unsupported model override: {params['model']}")
                )
                return
            context.thread_id = thread_id
            context.previous_profile = state.profile
            context.previous_effort = state.effort or PROFILES[state.profile].effort
            context.profile = requested_profile or state.profile
            context.effort = str(params.get("effort") or context.previous_effort)
        elif method == "turn/steer":
            message, target, effort, marker, temporary = prepare_turn_steer(message)
            if (
                self.fixed_profile
                and target
                and (target != self.fixed_profile or effort != PROFILES[self.fixed_profile].effort)
            ):
                await self._send_client(
                    _rpc_error(
                        request_id,
                        -32602,
                        "model directives cannot change a fixed-profile evaluation run",
                    )
                )
                return
            if target is not None and marker is not None:
                self._start_background(
                    self._apply_user_steer(
                        message, target, str(effort), marker, temporary=temporary
                    ),
                    "user-directive-steer",
                )
                return

        if request_key is not None and method:
            self.client_requests[request_key] = context
        await self._send_upstream(message)
        if method == "initialized" and not self.catalog_validation_started:
            self.catalog_validation_started = True
            self._start_background(self._validate_model_catalog(), "model-catalog-validation")

    async def _handle_upstream_payload(self, payload: str) -> None:
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            LOG.warning("ignoring non-JSON app-server output: %s", payload.rstrip())
            return
        if not isinstance(message, dict):
            return

        old_sequence = self.measurements.sequence
        self.measurements.observe(message)
        if self.measurements.sequence != old_sequence:
            self._save_report()

        if "id" in message and "method" not in message:
            key = _id_key(message["id"])
            internal = self.internal_requests.pop(key, None)
            if internal is not None:
                if not internal.done():
                    internal.set_result(message)
                return

            context = self.client_requests.pop(key, None)
            if context is not None:
                self._observe_client_response(context, message)

        method = message.get("method")
        params = message.get("params", {})
        restored_route: tuple[str, ProfileName, str] | None = None
        event_turn = (str(params.get("threadId", "")), str(params.get("turnId", "")))
        if event_turn in self.automatic_continuation_turns and _is_async_question_item(message):
            if method == "item/completed":
                self.measurements.event(
                    "async_question_suppressed", thread_id=event_turn[0], turn_id=event_turn[1]
                )
                self._save_report()
            LOG.info("suppressing optional async question at model-switch continuation boundary")
            return
        if method == "rawResponseItem/completed":
            await self._observe_raw_item(message)
            return
        if method == "rawResponse/completed":
            await self._observe_raw_response(message)
            return
        if method == "turn/started":
            params = message.get("params", {})
            turn = params.get("turn", {})
            thread_id = str(params.get("threadId", ""))
            turn_id = str(turn.get("id", ""))
            if thread_id in self.threads:
                self.threads[thread_id].active_turn_id = turn_id
        elif method == "turn/completed":
            thread_id = str(params.get("threadId", ""))
            turn_id = str(params.get("turn", {}).get("id") or params.get("turnId", ""))
            self.automatic_continuation_turns.discard((thread_id, turn_id))
            if thread_id in self.threads and self.threads[thread_id].active_turn_id == turn_id:
                self.threads[thread_id].active_turn_id = None
            waiter = self.turn_completion_waiters.get((thread_id, turn_id))
            if waiter is not None and not waiter.done():
                waiter.set_result(message)
            directive_lock_released = self._release_explicit_route_lock(thread_id, turn_id)
            restored = self._restore_temporary_route(thread_id, turn_id)
            if restored is not None:
                restored_route = (thread_id, *restored)
            elif directive_lock_released:
                self._checkpoint_thread(thread_id)

        if is_route_state_tool_call(message):
            await self._handle_route_state_call(message)
            return
        if is_router_tool_call(message):
            if self.fixed_profile:
                await self._send_upstream(
                    _tool_result(
                        message.get("id"),
                        "Model switching is disabled for this fixed-profile evaluation run.",
                        success=False,
                    )
                )
                return
            self._start_background(self._handle_switch_call(message), "model-switch")
            return
        await self._send_client(message)
        if restored_route is not None:
            await self._notify_restored_route(*restored_route)

    async def _handle_route_state_call(self, message: dict[str, Any]) -> None:
        params = message.get("params", {})
        thread_id = str(params.get("threadId", ""))
        turn_id = str(params.get("turnId", ""))
        state = self.threads.get(thread_id)
        if state is None:
            await self._send_upstream(
                _tool_result(
                    message.get("id"), "Router has no state for this thread.", success=False
                )
            )
            return
        effort = state.effort or PROFILES[state.profile].effort
        route = {
            "profile": state.profile.value,
            "model": PROFILES[state.profile].model,
            "reasoningEffort": effort,
            "agentSwitchingEnabled": self.agent_switching_enabled,
            "approvalPolicy": self.switch_approval.value,
        }
        self.measurements.event(
            "route_state_queried",
            thread_id=thread_id,
            turn_id=turn_id,
            profile=state.profile.value,
            effort=effort,
        )
        self._save_report()
        await self._send_upstream(
            _tool_result(message.get("id"), json.dumps(route, separators=(",", ":")), success=True)
        )

    async def _notify_active_route(self, thread_id: str, profile: ProfileName, effort: str) -> None:
        await self._send_client(
            {
                "method": "warning",
                "params": {
                    "threadId": thread_id,
                    "message": (
                        f"Self-router active route: {profile.value}/{effort} "
                        f"({PROFILES[profile].model}). The Codex model label may lag until the "
                        "current turn completes."
                    ),
                },
            }
        )

    @staticmethod
    def _clear_temporary_route(state: ThreadState) -> None:
        state.temporary_restore_profile = None
        state.temporary_restore_effort = None
        state.temporary_task_id = None

    def _release_explicit_route_lock(self, thread_id: str, turn_id: str) -> bool:
        state = self.threads.get(thread_id)
        if state is None or state.explicit_route_task_id is None:
            return False
        turn = self.measurements.turns.get((thread_id, turn_id), {})
        task_id = str(turn.get("task_id") or turn_id)
        if state.explicit_route_task_id != task_id:
            return False
        if turn.get("interruption_source") == "router":
            return False
        state.explicit_route_task_id = None
        self.measurements.event(
            "explicit_route_task_unlocked",
            thread_id=thread_id,
            turn_id=turn_id,
            task_id=task_id,
        )
        return True

    def _explicit_route_is_locked(self, state: ThreadState, thread_id: str, turn_id: str) -> bool:
        if state.explicit_route_task_id is None:
            return False
        turn = self.measurements.turns.get((thread_id, turn_id), {})
        task_id = str(turn.get("task_id") or turn_id)
        return state.explicit_route_task_id == task_id

    def _restore_temporary_route(
        self, thread_id: str, turn_id: str
    ) -> tuple[ProfileName, str] | None:
        state = self.threads.get(thread_id)
        if state is None or state.temporary_restore_profile is None:
            return None
        turn = self.measurements.turns.get((thread_id, turn_id), {})
        task_id = str(turn.get("task_id") or turn_id)
        if state.temporary_task_id not in {None, task_id}:
            return None
        if turn.get("interruption_source") == "router":
            return None

        previous = state.profile
        previous_effort = state.effort or PROFILES[previous].effort
        target = state.temporary_restore_profile
        target_effort = state.temporary_restore_effort or PROFILES[target].effort
        state.profile = target
        state.effort = target_effort
        if state.collaboration_mode is not None:
            settings = state.collaboration_mode.setdefault("settings", {})
            settings["model"] = PROFILES[target].model
            settings["reasoning_effort"] = target_effort
        self._clear_temporary_route(state)
        self.report.switches.append(
            SwitchRecord(
                timestamp=utc_now(),
                thread_id=thread_id,
                turn_id=turn_id,
                from_profile=previous.value,
                to_profile=target.value,
                source="temporary-directive-restore",
                outcome="applied",
                detail="Restored the route active before the task-scoped directive.",
                response_index=len(self.report.responses),
                from_effort=previous_effort,
                to_effort=target_effort,
            )
        )
        self.measurements.event(
            "temporary_route_restored",
            thread_id=thread_id,
            turn_id=turn_id,
            task_id=task_id,
            profile=target.value,
            effort=target_effort,
        )
        self._checkpoint_thread(thread_id)
        return target, target_effort

    async def _notify_restored_route(
        self, thread_id: str, profile: ProfileName, effort: str
    ) -> None:
        await self._send_client(
            {
                "method": "warning",
                "params": {
                    "threadId": thread_id,
                    "message": (
                        f"Self-router restored route: {profile.value}/{effort} "
                        f"({PROFILES[profile].model})."
                    ),
                },
            }
        )

    def _start_background(self, coroutine: Any, name: str) -> None:
        task = asyncio.create_task(coroutine, name=name)
        self.background.add(task)

        def complete(done: asyncio.Task[Any]) -> None:
            self.background.discard(done)
            if done.cancelled():
                return
            exception = done.exception()
            if exception is not None:
                self.measurements.event(
                    "router_error", operation=name, exception_type=type(exception).__name__
                )
                self._save_report()
                LOG.error("%s failed: %s", name, exception)

        task.add_done_callback(complete)

    async def _validate_model_catalog(self) -> None:
        try:
            response = await self._internal_request(
                "model/list", {"limit": 100, "includeHidden": True}
            )
            if "error" in response:
                raise RuntimeError(response["error"].get("message", "model/list failed"))
            problems = inspect_model_catalog(response)
        except Exception as exc:
            problems = [f"could not validate model catalog: {exc}"]
        if not problems:
            return
        message = "codex-self-router profile validation: " + "; ".join(problems)
        self.report.notes.append(message)
        self._save_report()
        await self._send_client({"method": "warning", "params": {"message": message}})

    async def _apply_user_steer(
        self,
        message: dict[str, Any],
        target: ProfileName,
        effort: str,
        marker: str,
        *,
        temporary: bool,
    ) -> None:
        params = message.get("params", {})
        request_id = message.get("id")
        thread_id = str(params.get("threadId", ""))
        turn_id = str(params.get("turnId", ""))
        state = self.threads.get(thread_id)
        if state is None:
            await self._send_client(
                _rpc_error(request_id, -32602, "router has no state for this thread")
            )
            return
        previous = state.profile
        previous_effort = state.effort or PROFILES[previous].effort
        route_changed = target != previous or effort != previous_effort
        if route_changed:
            profile = PROFILES[target]
            try:
                response = await self._internal_request(
                    "turn/settings/update",
                    {
                        "threadId": thread_id,
                        "turnId": turn_id,
                        "model": profile.model,
                        "effort": effort,
                    },
                )
            except Exception as exc:
                await self._send_client(
                    _rpc_error(request_id, -32000, f"explicit model switch failed: {exc}")
                )
                return
            if "error" in response:
                await self._send_client(
                    _rpc_error(
                        request_id,
                        -32000,
                        "explicit model switch failed: "
                        + response["error"].get("message", "unknown app-server error"),
                    )
                )
                return
            status = response.get("result", {}).get("status")
            if status != "applied":
                await self._send_client(
                    _rpc_error(
                        request_id,
                        -32000,
                        f"explicit model switch was not applied: {status!r}",
                    )
                )
                return
            state.profile = target
            state.effort = effort
        if temporary:
            if state.temporary_restore_profile is None:
                state.temporary_restore_profile = previous
                state.temporary_restore_effort = previous_effort
            turn = self.measurements.turns.get((thread_id, turn_id), {})
            state.temporary_task_id = str(turn.get("task_id") or turn_id)
        else:
            self._clear_temporary_route(state)
        turn = self.measurements.turns.get((thread_id, turn_id), {})
        state.explicit_route_task_id = str(turn.get("task_id") or turn_id)
        self.report.add_directive(
            thread_id=thread_id,
            turn_id=turn_id,
            previous=previous,
            target=target,
            previous_effort=previous_effort,
            target_effort=effort,
            marker=marker,
        )
        state.effort = effort
        self._checkpoint_thread(thread_id)
        self._save_report()
        if route_changed:
            await self._notify_active_route(thread_id, target, effort)
        await self._send_upstream(message)

    def _observe_client_response(self, context: RequestContext, response: dict[str, Any]) -> None:
        if "error" in response:
            if context.method == "turn/start" and context.thread_id and context.previous_profile:
                state = self.threads.get(context.thread_id)
                if state:
                    state.profile = context.previous_profile
                    state.effort = context.previous_effort
            return
        if context.method == "thread/start":
            thread_id = response.get("result", {}).get("thread", {}).get("id")
            if thread_id and context.profile:
                self.threads[str(thread_id)] = ThreadState(
                    context.profile,
                    effort=response.get("result", {}).get("reasoningEffort"),
                    client_profile=context.profile,
                    client_effort=response.get("result", {}).get("reasoningEffort"),
                )
                self._checkpoint_thread(str(thread_id), context.developer_instructions)
        elif context.method == "thread/fork" and context.thread_id and context.profile:
            result = response.get("result", {})
            thread_id = str(result.get("thread", {}).get("id", ""))
            result_profile = profile_for_model(result.get("model"))
            result_effort = result.get("reasoningEffort") or context.effort
            if (
                not thread_id
                or result_profile != context.profile
                or result_effort != context.effort
            ):
                request_id = response.get("id")
                response.clear()
                response.update(
                    _rpc_error(
                        request_id,
                        -32602,
                        "app-server forked an unexpected thread or did not preserve its route",
                    )
                )
                return
            state = ThreadState(
                result_profile,
                collaboration_mode=copy.deepcopy(context.collaboration_mode),
                effort=str(result_effort),
                client_profile=result_profile,
                client_effort=str(result_effort),
            )
            self.threads[thread_id] = state
            try:
                self._checkpoint_thread(thread_id, context.developer_instructions)
            except (OSError, ValueError) as exc:
                self.threads.pop(thread_id, None)
                lease = self.thread_leases.pop(thread_id, None)
                if lease is not None:
                    lease.close()
                request_id = response.get("id")
                response.clear()
                response.update(_rpc_error(request_id, -32602, f"cannot track fork: {exc}"))
                return
            self.report.metadata.setdefault("forkedThreads", []).append(
                {
                    "sourceThreadId": context.thread_id,
                    "threadId": thread_id,
                    "inheritedProfile": result_profile.value,
                    "inheritedEffort": str(result_effort),
                }
            )
            self.measurements.event(
                "thread_forked",
                thread_id=thread_id,
                source_thread_id=context.thread_id,
                profile=result_profile.value,
                effort=str(result_effort),
            )
            self._save_report()
        elif context.method == "turn/start" and context.thread_id and context.route:
            turn_id = response.get("result", {}).get("turn", {}).get("id")
            if turn_id:
                self.threads[context.thread_id].active_turn_id = str(turn_id)
                measured = self.measurements.start_turn(context.thread_id, str(turn_id))
                if context.submitted_event:
                    measured["start_ms"] = context.submitted_event["elapsed_ms"]
                    measured["started_at"] = context.submitted_event["timestamp"]
                measured["requested_profile"] = context.profile
                measured["requested_effort"] = context.route.effort
                state = self.threads[context.thread_id]
                state.effort = context.effort or PROFILES[context.profile].effort
                if context.route.source == "user-directive":
                    state.explicit_route_task_id = str(measured["task_id"])
                    if context.route.temporary:
                        if state.temporary_restore_profile is None:
                            state.temporary_restore_profile = context.previous_profile
                            state.temporary_restore_effort = context.previous_effort
                        state.temporary_task_id = str(measured["task_id"])
                    else:
                        self._clear_temporary_route(state)
                if context.client_route_changed and context.route.source == "client-override":
                    state.client_profile = context.profile
                    state.client_effort = context.effort
                    self._clear_temporary_route(state)
                self._checkpoint_thread(context.thread_id)
            if context.route.source == "user-directive" and context.route.marker:
                self.report.add_directive(
                    thread_id=context.thread_id,
                    turn_id=str(turn_id) if turn_id else None,
                    previous=context.previous_profile,
                    target=context.route.profile,
                    previous_effort=context.previous_effort,
                    target_effort=context.route.effort,
                    marker=context.route.marker,
                )
            elif (
                context.previous_profile != context.route.profile
                or context.previous_effort != context.route.effort
            ):
                self.report.switches.append(
                    SwitchRecord(
                        timestamp=utc_now(),
                        thread_id=context.thread_id,
                        turn_id=str(turn_id) if turn_id else None,
                        from_profile=(
                            context.previous_profile.value if context.previous_profile else None
                        ),
                        to_profile=context.route.profile.value,
                        source=context.route.source,
                        outcome="applied",
                        response_index=len(self.report.responses),
                        from_effort=context.previous_effort,
                        to_effort=context.route.effort,
                    )
                )
                self._save_report()
        elif (
            context.method == "turn/settings/update"
            and context.thread_id
            and context.profile
            and response.get("result", {}).get("status") == "applied"
        ):
            state = self.threads.get(context.thread_id)
            if state is not None:
                state.profile = context.profile
                state.effort = context.effort or PROFILES[context.profile].effort
                state.client_profile = state.profile
                state.client_effort = state.effort
                self._clear_temporary_route(state)
                state.explicit_route_task_id = None
                if state.collaboration_mode is not None:
                    settings = state.collaboration_mode.setdefault("settings", {})
                    settings["model"] = PROFILES[state.profile].model
                    settings["reasoning_effort"] = state.effort
                if (
                    context.previous_profile != state.profile
                    or context.previous_effort != state.effort
                ):
                    self.report.switches.append(
                        SwitchRecord(
                            timestamp=utc_now(),
                            thread_id=context.thread_id,
                            turn_id=state.active_turn_id,
                            from_profile=(
                                context.previous_profile.value if context.previous_profile else None
                            ),
                            to_profile=state.profile.value,
                            source="client-settings",
                            outcome="applied",
                            detail="Applied directly through turn/settings/update.",
                            response_index=len(self.report.responses),
                            from_effort=context.previous_effort,
                            to_effort=state.effort,
                        )
                    )
                self._checkpoint_thread(context.thread_id)
                self._save_report()

    async def _observe_raw_item(self, message: dict[str, Any]) -> None:
        params = message.get("params", {})
        item = params.get("item", {})
        call_id = item.get("call_id") or item.get("callId")
        if not call_id:
            return
        if item.get("type") in {"function_call_output", "custom_tool_call_output"}:
            self.open_raw_calls.discard(str(call_id))
            return
        thread_turn = (str(params.get("threadId", "")), str(params.get("turnId", "")))
        next_sequence = self.response_sequence.get(thread_turn, 0) + 1
        self.raw_call_sequence[str(call_id)] = (thread_turn, next_sequence)
        self.open_raw_calls.add(str(call_id))

    async def _observe_raw_response(self, message: dict[str, Any]) -> None:
        params = message.get("params", {})
        thread_id = str(params.get("threadId", ""))
        turn_id = str(params.get("turnId", ""))
        key = (thread_id, turn_id)
        response_id = str(params.get("responseId", ""))
        identity = (thread_id, turn_id, response_id)
        if response_id and identity in self.response_ids:
            return
        if response_id:
            self.response_ids.add(identity)
        state = self.threads.get(thread_id)
        profile_name = state.profile if state else None
        effort = (state.effort or PROFILES[state.profile].effort) if state else None
        if params.get("usageSource") == "codex-rollout":
            history = self.profile_history.get(key, [])
            observed_at = params.get("observedAt")
            if history:
                # A file flush can arrive after the next turn or model update.
                profile_name, effort = history[0][1:]
                if observed_at:
                    for changed_at, candidate, candidate_effort in history:
                        if parse_timestamp(changed_at) <= parse_timestamp(observed_at):
                            profile_name = candidate
                            effort = candidate_effort
        usage = params.get("usage")
        normalized_usage = dict(usage) if isinstance(usage, dict) else None
        self.report.responses.append(
            UsageRecord(
                timestamp=params.get("observedAt") or utc_now(),
                thread_id=thread_id,
                turn_id=turn_id,
                response_id=response_id,
                profile=profile_name.value if profile_name else None,
                model=PROFILES[profile_name].model if profile_name else None,
                effort=effort,
                usage=normalized_usage,
                usage_metadata=params.get("usageMetadata"),
                usage_source=params.get("usageSource", "raw-response-event"),
            )
        )
        self.measurements.event(
            "response_completed",
            thread_id=thread_id,
            turn_id=turn_id,
            response_id=response_id,
            profile=profile_name,
            usage=normalized_usage,
        )
        self._save_report()
        async with self.response_condition:
            self.response_sequence[key] = self.response_sequence.get(key, 0) + 1
            self.response_condition.notify_all()

    async def _handle_switch_call(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        params = message.get("params", {})
        thread_id = str(params.get("threadId", ""))
        turn_id = str(params.get("turnId", ""))
        call_id = str(params.get("callId", ""))
        if not self.agent_switching_enabled:
            self.measurements.event(
                "agent_switch_rejected",
                thread_id=thread_id,
                turn_id=turn_id,
                reason="disabled-by-config",
            )
            self._save_report()
            await self._send_upstream(
                _tool_result(
                    request_id,
                    "Agent-requested model switching is disabled by router configuration. "
                    "Only explicit user directives or manual model changes can change the route.",
                    success=False,
                )
            )
            return
        lock = self.turn_locks.setdefault((thread_id, turn_id), asyncio.Lock())
        async with lock:
            state = self.threads.get(thread_id)
            if state is None:
                await self._send_upstream(
                    _tool_result(request_id, "Router has no state for this thread.", success=False)
                )
                return
            if self._explicit_route_is_locked(state, thread_id, turn_id):
                self.measurements.event(
                    "agent_switch_rejected",
                    thread_id=thread_id,
                    turn_id=turn_id,
                    task_id=state.explicit_route_task_id,
                    reason="explicit-user-route",
                )
                self._save_report()
                await self._send_upstream(
                    _tool_result(
                        request_id,
                        "The user explicitly selected the model and reasoning effort for this "
                        "task. Keep the current route until the task ends.",
                        success=False,
                    )
                )
                return
            try:
                previous = state.profile
                previous_effort = state.effort or PROFILES[previous].effort
                target, target_effort, reason = parse_switch_arguments(
                    params, previous, previous_effort
                )
            except ValueError as exc:
                await self._send_upstream(_tool_result(request_id, str(exc), success=False))
                return

            switch = SwitchRecord(
                timestamp=utc_now(),
                thread_id=thread_id,
                turn_id=turn_id,
                origin_turn_id=turn_id,
                from_profile=previous.value,
                to_profile=target.value,
                source="agent-tool",
                outcome="pending",
                reason=reason,
                switch_id=str(uuid.uuid4()),
                requested_at=utc_now(),
                from_effort=previous_effort,
                to_effort=target_effort,
            )
            approval_required = switch_requires_approval(
                previous,
                previous_effort,
                target,
                target_effort,
                self.switch_approval,
            )
            switch.approval_policy = self.switch_approval.value
            switch.approval_required = approval_required
            self.report.switches.append(switch)
            self.switch_clocks[switch.switch_id] = {"started": time.monotonic()}
            self.measurements.event(
                "switch_requested",
                switch_id=switch.switch_id,
                thread_id=thread_id,
                turn_id=turn_id,
                from_profile=previous.value,
                to_profile=target.value,
                from_effort=previous_effort,
                to_effort=target_effort,
                approval_policy=self.switch_approval.value,
                approval_required=approval_required,
            )
            self._save_report()
            if target == previous and target_effort == previous_effort:
                switch.approval_required = False
                switch.authorization = "not-needed"
                switch.approval_ms = 0.0
                self._record_agent_switch(thread_id, turn_id, previous, target, "noop")
                await self._send_upstream(
                    _tool_result(
                        request_id,
                        f"Already using {target.value}/{target_effort}; no approval was needed.",
                        success=True,
                    )
                )
                return

            if approval_required:
                approval_started = time.monotonic()
                approved = await self._request_approval(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    call_id=call_id,
                    previous=previous,
                    previous_effort=previous_effort,
                    target=target,
                    target_effort=target_effort,
                    reason=reason,
                )
                switch.approval_ms = round((time.monotonic() - approval_started) * 1000, 3)
                switch.authorization = "user-approved" if approved else "user-denied"
            else:
                approved = True
                switch.approval_ms = 0.0
                switch.authorization = "router-policy"
            if not approved:
                self._record_agent_switch(
                    thread_id,
                    turn_id,
                    previous,
                    target,
                    "denied",
                )
                await self._send_upstream(
                    _tool_result(
                        request_id,
                        f"User kept the current {previous.value}/{previous_effort} route.",
                        success=False,
                    )
                )
                return

            usage_wait_started = time.monotonic()
            try:
                await self._wait_until_call_response_recorded(call_id, thread_id, turn_id)
            except RuntimeError as exc:
                self._record_agent_switch(
                    thread_id,
                    turn_id,
                    previous,
                    target,
                    "failed",
                    detail=str(exc),
                )
                await self._send_upstream(_tool_result(request_id, str(exc), success=False))
                return
            switch.usage_wait_ms = round((time.monotonic() - usage_wait_started) * 1000, 3)
            association = self.raw_call_sequence.get(call_id)
            candidates = [
                r
                for r in self.report.responses
                if r.thread_id == thread_id and r.turn_id == turn_id
            ]
            if association and association[0] == (thread_id, turn_id):
                ordinal = association[1] - 1
                if 0 <= ordinal < len(candidates):
                    switch.decision_response_id = candidates[ordinal].response_id
            self.switch_clocks[switch.switch_id]["apply_started"] = time.monotonic()
            switch.response_index = len(self.report.responses)
            switch.mechanism = "settings-update"
            profile = PROFILES[target]
            try:
                response = await self._internal_request(
                    "turn/settings/update",
                    {
                        "threadId": thread_id,
                        "turnId": turn_id,
                        "model": profile.model,
                        "effort": target_effort,
                    },
                )
                if "error" in response:
                    detail = response["error"].get("message", "unknown app-server error")
                    if str(detail).startswith(STEP_UPDATE_COMPATIBILITY_PREFIX):
                        switch.mechanism = "continuation"
                        try:
                            continuation_turn_id = await self._continue_on_new_turn(
                                thread_id=thread_id,
                                turn_id=turn_id,
                                previous=previous,
                                previous_effort=previous_effort,
                                target=target,
                                target_effort=target_effort,
                            )
                        except Exception as exc:
                            self._record_agent_switch(
                                thread_id,
                                turn_id,
                                previous,
                                target,
                                "failed",
                                detail=str(exc),
                            )
                            await self._send_client(
                                {
                                    "method": "warning",
                                    "params": {
                                        "message": f"Model-switched continuation failed: {exc}"
                                    },
                                }
                            )
                            return
                        self._record_agent_switch(
                            thread_id,
                            continuation_turn_id,
                            previous,
                            target,
                            "applied",
                            detail=(
                                "Continued in a fresh turn because Codex rejected an in-turn "
                                f"settings update: {detail}"
                            ),
                            origin_turn_id=turn_id,
                        )
                        await self._notify_active_route(thread_id, target, target_effort)
                        return
                    raise RuntimeError(detail)
                status = response.get("result", {}).get("status")
                if status != "applied":
                    raise RuntimeError(f"app-server returned status {status!r}")
            except Exception as exc:
                self._record_agent_switch(
                    thread_id,
                    turn_id,
                    previous,
                    target,
                    "failed",
                    detail=str(exc),
                )
                await self._send_upstream(
                    _tool_result(request_id, f"Model switch failed: {exc}", success=False)
                )
                return

            state.profile = target
            state.effort = target_effort
            self._record_agent_switch(
                thread_id,
                turn_id,
                previous,
                target,
                "applied",
            )
            await self._notify_active_route(thread_id, target, target_effort)
            await self._send_upstream(
                _tool_result(
                    request_id,
                    (
                        f"Switched from {previous.value}/{previous_effort} "
                        f"to {target.value}/{target_effort}. Continue the current task."
                    ),
                    success=True,
                )
            )

    async def _continue_on_new_turn(
        self,
        *,
        thread_id: str,
        turn_id: str,
        previous: ProfileName,
        previous_effort: str,
        target: ProfileName,
        target_effort: str,
        reason: str | None = None,
    ) -> str:
        key = (thread_id, turn_id)
        if key in self.turn_completion_waiters:
            raise RuntimeError("a continuation is already waiting for this turn")
        completion = asyncio.get_running_loop().create_future()
        self.turn_completion_waiters[key] = completion
        self.measurements.interruptions[key] = "router"
        try:
            interrupted = await self._internal_request(
                "turn/interrupt", {"threadId": thread_id, "turnId": turn_id}
            )
            if "error" in interrupted:
                raise RuntimeError(
                    "could not interrupt incompatible turn: "
                    + interrupted["error"].get("message", "unknown app-server error")
                )
            await asyncio.wait_for(asyncio.shield(completion), timeout=30)
        finally:
            self.turn_completion_waiters.pop(key, None)

        state = self.threads[thread_id]
        profile = PROFILES[target]
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [],
            "toolOutput": {
                "name": "request_model_switch",
                "namespace": "self_router",
                "output": _continuation_tool_output(
                    previous, previous_effort, target, target_effort
                ),
            },
            "model": profile.model,
            "effort": target_effort,
        }
        if state.collaboration_mode is not None:
            collaboration_mode = copy.deepcopy(state.collaboration_mode)
            settings = collaboration_mode.setdefault("settings", {})
            settings["model"] = profile.model
            settings["reasoning_effort"] = target_effort
            params["collaborationMode"] = collaboration_mode

        state.profile = target
        state.effort = target_effort
        self.measurements.continuations[thread_id] = turn_id
        try:
            started = await self._internal_request("turn/start", params)
            if "error" in started:
                raise RuntimeError(
                    "could not start model-switched continuation: "
                    + started["error"].get("message", "unknown app-server error")
                )
            continuation_turn_id = str(started.get("result", {}).get("turn", {}).get("id", ""))
            if not continuation_turn_id:
                raise RuntimeError("model-switched continuation returned no turn id")
            state.active_turn_id = continuation_turn_id
            self.measurements.start_turn(thread_id, continuation_turn_id)
            state.collaboration_mode = copy.deepcopy(params.get("collaborationMode"))
            self.automatic_continuation_turns.add((thread_id, continuation_turn_id))
            return continuation_turn_id
        except Exception:
            self.measurements.continuations.pop(thread_id, None)
            state.profile = previous
            state.effort = previous_effort
            raise

    async def _request_approval(
        self,
        *,
        thread_id: str,
        turn_id: str,
        call_id: str,
        previous: ProfileName,
        previous_effort: str,
        target: ProfileName,
        target_effort: str,
        reason: str | None = None,
    ) -> bool:
        approval_id = f"self-router-approval-{uuid.uuid4()}"
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.client_prompts[_id_key(approval_id)] = future
        self.measurements.start_wait(approval_id, thread_id, turn_id, "model-switch-approval")
        self._save_report()
        await self._send_client(
            {
                "id": approval_id,
                "method": "item/tool/requestUserInput",
                "params": {
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "itemId": call_id,
                    "questions": [
                        {
                            "id": "model_switch",
                            "header": "Model switch",
                            "question": (
                                f"Switch {previous.value}/{previous_effort} → "
                                f"{target.value}/{target_effort}?"
                                + (f"\nReason: {reason}" if reason else "")
                            ),
                            "isOther": False,
                            "isSecret": False,
                            "options": [
                                {
                                    "label": "Approve switch",
                                    "description": (
                                        f"Continue with {target.value}/{target_effort}."
                                    ),
                                },
                                {
                                    "label": "Stay on current",
                                    "description": (
                                        f"Keep using {previous.value}/{previous_effort}."
                                    ),
                                },
                            ],
                        }
                    ],
                    "isBlocking": True,
                    "autoResolutionMs": None,
                },
            }
        )
        response = await future
        self.measurements.resolve_wait(approval_id)
        self._save_report()
        await self._send_client(
            {
                "method": "serverRequest/resolved",
                "params": {"threadId": thread_id, "requestId": approval_id},
            }
        )
        if "error" in response:
            return False
        answers = (
            response.get("result", {}).get("answers", {}).get("model_switch", {}).get("answers", [])
        )
        return "Approve switch" in answers

    async def _wait_until_call_response_recorded(
        self, call_id: str, thread_id: str, turn_id: str
    ) -> None:
        if thread_id in self.rollouts:
            # Code-mode assigns nested calls fresh exec-UUID IDs, unrelated to the
            # enclosing response's call ID. Wait for ALL observed open outer-call
            # responses to finish, but do not claim an exact parent association.
            # Interrupting before response completion can discard its usage record.
            nested = call_id.startswith("exec-")
            if nested:
                self.measurements.event(
                    "decision_response_unavailable",
                    thread_id=thread_id,
                    turn_id=turn_id,
                    reason="nested-code-mode-call",
                )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                await self._drain_rollout(thread_id)
                association = self.raw_call_sequence.get(call_id)
                if nested:
                    candidates = [
                        self.raw_call_sequence[outer]
                        for outer in self.open_raw_calls
                        if self.raw_call_sequence[outer][0] == (thread_id, turn_id)
                    ]
                    association = max(candidates, key=lambda item: item[1], default=None)
                if (
                    association
                    and association[0] == (thread_id, turn_id)
                    and (self.response_sequence.get((thread_id, turn_id), 0) >= association[1])
                ):
                    return
                if thread_id in self.rollout_failed:
                    break
                await asyncio.sleep(0.05)
            raise RuntimeError(
                "could not verify the requesting response in the resumed rollout; model unchanged"
            )
        # Raw response items arrive before rawResponse/completed and let us associate this
        # particular tool call with the correct sampling response. A short fallback prevents an
        # older/changed app-server from deadlocking routing if it omits raw item events.
        await asyncio.sleep(0)
        key = (thread_id, turn_id)
        association = self.raw_call_sequence.get(call_id)
        if association is None or association[0] != key:
            return
        required_sequence = association[1]

        async def wait() -> None:
            async with self.response_condition:
                await self.response_condition.wait_for(
                    lambda: self.response_sequence.get(key, 0) >= required_sequence
                )

        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(wait(), timeout=10)
        if self.response_sequence.get(key, 0) < required_sequence:
            self.measurements.event("usage_boundary_timeout", thread_id=thread_id, turn_id=turn_id)
            self._save_report()

    async def _internal_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = f"self-router-internal-{uuid.uuid4()}"
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.internal_requests[_id_key(request_id)] = future
        await self._send_upstream({"id": request_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(future, timeout=30)
        finally:
            self.internal_requests.pop(_id_key(request_id), None)

    def _record_agent_switch(
        self,
        thread_id: str,
        turn_id: str,
        previous: ProfileName,
        target: ProfileName,
        outcome: str,
        *,
        detail: str | None = None,
        origin_turn_id: str | None = None,
    ) -> None:
        record = next(
            (
                s
                for s in reversed(self.report.switches)
                if s.thread_id == thread_id
                and s.outcome == "pending"
                and s.origin_turn_id == (origin_turn_id or turn_id)
            ),
            None,
        )
        if record is None:
            current_effort = (
                self.threads.get(thread_id).effort if thread_id in self.threads else None
            )
            record = SwitchRecord(
                timestamp=utc_now(),
                thread_id=thread_id,
                turn_id=turn_id,
                from_profile=previous.value,
                to_profile=target.value,
                source="agent-tool",
                outcome=outcome,
                detail=detail,
                from_effort=current_effort,
                to_effort=PROFILES[target].effort,
            )
            self.report.switches.append(record)
        record.timestamp = utc_now()
        record.turn_id = turn_id
        record.outcome = outcome
        record.detail = detail
        clock = self.switch_clocks.pop(record.switch_id, {})
        if "started" in clock:
            record.total_ms = round((time.monotonic() - clock["started"]) * 1000, 3)
        if "apply_started" in clock:
            record.apply_ms = round((time.monotonic() - clock["apply_started"]) * 1000, 3)
        self.measurements.event(
            "switch_resolved",
            switch_id=record.switch_id,
            thread_id=thread_id,
            turn_id=turn_id,
            outcome=outcome,
            approval_ms=record.approval_ms,
            approval_policy=record.approval_policy,
            approval_required=record.approval_required,
            authorization=record.authorization,
            apply_ms=record.apply_ms,
            total_ms=record.total_ms,
            mechanism=record.mechanism,
            from_effort=record.from_effort,
            to_effort=record.to_effort,
        )
        if outcome == "applied":
            self.threads[thread_id].effort = record.to_effort or PROFILES[target].effort
            self._checkpoint_thread(thread_id)
        self._save_report()

    async def _send_client(self, message: dict[str, Any]) -> None:
        if "error" in message:
            self.measurements.event("rpc_error", code=message["error"].get("code"))
            self._save_report()
        async with self.client_send_lock:
            await self.websocket.send(json.dumps(message, separators=(",", ":")))

    async def _send_upstream(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("app-server is not running")
        wire = json.dumps(message, separators=(",", ":")).encode() + b"\n"
        async with self.upstream_send_lock:
            self.process.stdin.write(wire)
            await self.process.stdin.drain()

    async def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while line := await self.process.stderr.readline():
            LOG.info("app-server: %s", line.decode(errors="replace").rstrip())

    async def _stop_process(self) -> None:
        if self.process is None or self.process.returncode is not None:
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()


def inspect_model_catalog(response: dict[str, Any]) -> list[str]:
    """Return validation problems for the configured profile catalog."""

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
        if effort_names:
            for effort in profile.allowed_efforts:
                if effort not in effort_names:
                    problems.append(f"{profile.model} does not advertise effort {effort}")
    return problems
