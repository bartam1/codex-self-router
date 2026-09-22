from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import PROFILES, ProfileName


def pricing_snapshot() -> dict[str, Any]:
    return {name.value: asdict(profile.price) for name, profile in PROFILES.items()}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp and normalize missing offsets to UTC."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def normalize_timestamp(value: str) -> str:
    return parse_timestamp(value).isoformat()


def default_report_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "codex-self-router" / "reports"
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "codex-self-router" / "reports"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "codex-self-router" / "reports"


@dataclass(slots=True)
class UsageRecord:
    timestamp: str
    thread_id: str
    turn_id: str
    response_id: str
    profile: str | None
    model: str | None
    usage: dict[str, int] | None
    usage_metadata: dict[str, Any] | None = None
    effort: str | None = None
    model_attribution: str = "router-state; not independently reported by upstream"
    usage_source: str = "raw-response-event"


@dataclass(slots=True)
class SwitchRecord:
    timestamp: str
    thread_id: str
    turn_id: str | None
    from_profile: str | None
    to_profile: str
    source: str
    outcome: str
    reason: str | None = None
    next_action: str | None = None
    estimated_follow_up_steps: int | None = None
    marker: str | None = None
    detail: str | None = None
    switch_id: str | None = None
    requested_at: str | None = None
    origin_turn_id: str | None = None
    approval_ms: float | None = None
    usage_wait_ms: float | None = None
    apply_ms: float | None = None
    total_ms: float | None = None
    response_index: int | None = None
    decision_response_id: str | None = None
    mechanism: str | None = None
    from_effort: str | None = None
    to_effort: str | None = None
    approval_policy: str | None = None
    approval_required: bool | None = None
    authorization: str | None = None


@dataclass(slots=True)
class SessionReport:
    session_id: str
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    switches: list[SwitchRecord] = field(default_factory=list)
    responses: list[UsageRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    prices: dict[str, Any] = field(default_factory=pricing_snapshot)
    thread_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(
        default_factory=lambda: [
            "Costs are API-equivalent estimates, not ChatGPT subscription charges.",
            "All-model comparisons reuse observed tokens and are not true counterfactual runs.",
            "Reasoning tokens are part of output tokens, not an additional charge.",
            "Turn completion and command exit codes do not establish task quality.",
            "Response timestamps include tool and user waits; inference latency is not exposed.",
        ]
    )

    def add_directive(
        self,
        *,
        thread_id: str,
        turn_id: str | None,
        previous: ProfileName | None,
        target: ProfileName,
        previous_effort: str | None,
        target_effort: str,
        marker: str,
    ) -> None:
        self.switches.append(
            SwitchRecord(
                timestamp=utc_now(),
                thread_id=thread_id,
                turn_id=turn_id,
                from_profile=previous.value if previous else None,
                to_profile=target.value,
                source="user-directive",
                outcome="applied",
                marker=marker,
                detail="Explicit user directive; approval intentionally bypassed.",
                response_index=len(self.responses),
                from_effort=previous_effort,
                to_effort=target_effort,
            )
        )

    def costs(self) -> dict[str, Any]:
        known = [
            item
            for item in self.responses
            if record_cost(item, item.profile, self.prices) is not None
        ]
        routed = sum(record_cost(item, item.profile, self.prices) or 0 for item in known)
        all_sol = sum(record_cost(item, ProfileName.SOL, self.prices) or 0 for item in known)
        all_astra = sum(record_cost(item, ProfileName.ASTRA, self.prices) or 0 for item in known)
        return {
            "routedApiEquivalentUsd": round(routed, 8),
            "sameObservedUsageAllSolUsd": round(all_sol, 8),
            "sameObservedUsageAllAstraUsd": round(all_astra, 8),
            "estimatedSavingsVsAllSolUsd": round(all_sol - routed, 8),
            "estimatedSavingsVsAllAstraUsd": round(all_astra - routed, 8),
            "pricedResponses": len(known),
            "unpricedResponses": len(self.responses) - len(known),
            "complete": len(known) == len(self.responses),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 2,
            "sessionId": self.session_id,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "switches": [asdict(item) for item in self.switches],
            "responses": [asdict(item) for item in self.responses],
            "costs": self.costs(),
            "pricingUsdPerMillionTokens": self.prices,
            "metadata": self.metadata,
            "measurements": self.measurements,
            "threadStates": self.thread_states,
            "notes": self.notes,
        }


def record_cost(
    record: UsageRecord,
    profile_name: str | None,
    prices: dict[str, Any] | None = None,
) -> float | None:
    return usage_cost(record.usage, (prices or pricing_snapshot()).get(profile_name))


def usage_cost(usage: dict[str, Any] | None, price: dict[str, Any] | None) -> float | None:
    if not usage or not price:
        return None
    required = ("inputTokens", "cachedInputTokens", "outputTokens")
    if any(type(usage.get(name)) is not int or usage[name] < 0 for name in required):
        return None
    if any(type(value) is not int or value < 0 for value in usage.values()):
        return None
    cached = usage["cachedInputTokens"]
    cache_write = usage.get("cacheWriteInputTokens", 0)
    total_input = usage["inputTokens"]
    if cached + cache_write > total_input:
        return None
    if usage.get("reasoningOutputTokens", 0) > usage["outputTokens"]:
        return None
    uncached = max(0, total_input - cached - cache_write)
    output = usage["outputTokens"]
    return (
        uncached * price["input"]
        + cached * price["cached_input"]
        + cache_write * price["cache_write_input"]
        + output * price["output"]
    ) / 1_000_000


class ReportStore:
    def __init__(self, report_dir: Path | None = None) -> None:
        self.report_dir = report_dir or default_report_dir()

    def path_for(self, session_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError("invalid session id")
        return self.report_dir / f"{session_id}.json"

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        path = self.path_for(session_id).with_suffix(".events.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")

    def read(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        feedback = self.report_dir / "feedback" / path.name
        data["feedback"] = (
            json.loads(feedback.read_text(encoding="utf-8")) if feedback.exists() else {}
        )
        return data

    def all(self) -> list[dict[str, Any]]:
        # Never silently omit damaged reports from an evaluation.
        return [self.read(path) for path in sorted(self.report_dir.glob("*.json"))]

    def thread_checkpoint(self, thread_id: str) -> dict[str, Any] | None:
        # State lives in its owning session's atomic report, so pricing/lineage and
        # routing cannot be restored from two independently committed snapshots.
        matches = []
        reports = self.all()
        for report in reports:
            state = report.get("threadStates", {}).get(thread_id)
            if state:
                matches.append((state["updated_at"], report, state))
        if not matches:
            # Upgrade pre-checkpoint router sessions from their last observed/confirmed route.
            candidates = []
            for report in reports:
                for row in report.get("responses", []):
                    if row["thread_id"] == thread_id and row.get("profile") in PROFILES:
                        candidates.append((row["timestamp"], row["profile"], report))
                for row in report.get("switches", []):
                    if row["thread_id"] == thread_id and row.get("outcome") == "applied":
                        candidates.append((row["timestamp"], row["to_profile"], report))
            if not candidates:
                return None
            timestamp, profile, report = max(candidates, key=lambda row: row[0])
            return {
                "profile": profile,
                "effort": PROFILES[ProfileName(profile)].effort,
                "updated_at": timestamp,
                "previous_session_id": report["sessionId"],
                "routing_mode": report.get("metadata", {}).get("routingMode", "auto"),
                "metadata": report.get("metadata", {}),
                "legacy_checkpoint": True,
            }
        _, report, state = max(matches, key=lambda match: match[0])
        return {
            **state,
            "previous_session_id": report["sessionId"],
            "metadata": report.get("metadata", {}),
        }

    def thread_history(self, thread_id: str) -> tuple[set[tuple[str, str, str]], list[str]]:
        identities = set()
        sessions = []
        for report in self.all():
            responses = [r for r in report.get("responses", []) if r["thread_id"] == thread_id]
            if responses or thread_id in report.get("threadStates", {}):
                sessions.append(report["sessionId"])
            identities.update(
                (thread_id, r["turn_id"], r["response_id"])
                for r in responses
                if r.get("response_id")
            )
        return identities, sessions

    def feedback(self, session_id: str, task_id: str, values: dict[str, Any]) -> None:
        data = self.read(self.path_for(session_id))
        tasks = {turn["task_id"] for turn in data.get("measurements", {}).get("turns", [])}
        if task_id not in tasks:
            raise ValueError("task id is not present in this session's measurements")
        destination = self.report_dir / "feedback" / self.path_for(session_id).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        feedback = data["feedback"]
        feedback[task_id] = {"timestamp": utc_now(), **values}
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(feedback, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def save(self, report: SessionReport) -> Path:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        destination = self.path_for(report.session_id)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
        return destination

    def latest(self) -> dict[str, Any] | None:
        if not self.report_dir.exists():
            return None
        candidates = sorted(self.report_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            return None
        return self.read(candidates[-1])
