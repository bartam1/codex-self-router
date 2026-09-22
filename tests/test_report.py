from __future__ import annotations

import json

from codex_self_router.config import ProfileName
from codex_self_router.evaluation import analyze
from codex_self_router.report import (
    ReportStore,
    SessionReport,
    SwitchRecord,
    UsageRecord,
    record_cost,
)


def usage_record(profile: str = "luna") -> UsageRecord:
    return UsageRecord(
        timestamp="2026-01-01T00:00:00+00:00",
        thread_id="thread-1",
        turn_id="turn-1",
        response_id="response-1",
        profile=profile,
        model="test-model",
        usage={
            "inputTokens": 1_000_000,
            "cachedInputTokens": 200_000,
            "cacheWriteInputTokens": 100_000,
            "outputTokens": 10_000,
            "reasoningOutputTokens": 5_000,
            "totalTokens": 1_010_000,
        },
    )


def test_cost_uses_separate_cache_buckets_without_double_counting() -> None:
    # Luna: 700k regular input + 200k cache hit + 100k cache write + 10k output.
    assert record_cost(usage_record(), ProfileName.LUNA) == 0.181


def test_report_compares_same_observed_usage(tmp_path) -> None:
    report = SessionReport("session-1")
    report.responses.append(usage_record())
    costs = report.costs()
    assert costs["routedApiEquivalentUsd"] < costs["sameObservedUsageAllSolUsd"]
    assert costs["sameObservedUsageAllSolUsd"] < costs["sameObservedUsageAllAstraUsd"]

    store = ReportStore(tmp_path)
    path = store.save(report)
    assert path.exists()
    assert json.loads(path.read_text())["sessionId"] == "session-1"
    assert store.latest()["sessionId"] == "session-1"


def test_analysis_accepts_legacy_naive_rollout_timestamps() -> None:
    report = SessionReport("session-1")
    response = usage_record()
    response.timestamp = "2026-01-01T00:00:01"
    response.usage_source = "codex-rollout"
    report.responses.append(response)
    report.switches.append(
        SwitchRecord(
            timestamp="2026-01-01T00:00:00+00:00",
            thread_id="thread-1",
            turn_id="turn-1",
            from_profile="sol",
            to_profile="luna",
            source="agent-tool",
            outcome="applied",
            response_index=0,
        )
    )
    report.measurements = {
        "turns": [
            {
                "thread_id": "thread-1",
                "turn_id": "turn-1",
                "task_id": "turn-1",
                "start_ms": 0,
                "end_ms": 1,
                "status": "completed",
                "interruption_source": None,
            }
        ]
    }

    assert analyze(report.to_dict())["phases"][0]["observedSteps"] == 1
