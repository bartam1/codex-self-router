from __future__ import annotations

import asyncio
import copy
import json
import threading

import pytest

from codex_self_router.config import ProfileName
from codex_self_router.evaluation import analyze
from codex_self_router.report import (
    ReportStore,
    ReportWriter,
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
    assert record_cost(usage_record(), ProfileName.LUNA) == 0.0895


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


@pytest.mark.asyncio
async def test_report_writer_coalesces_updates_and_flushes_on_close(tmp_path):
    report = SessionReport("session-1")
    saved = []

    class Store(ReportStore):
        def save(self, snapshot):
            saved.append(snapshot)
            return super().save(snapshot)

    store = Store(tmp_path)
    writer = ReportWriter(store, lambda: copy.deepcopy(report), interval=60)
    writer.start()
    for index in range(100):
        report.metadata["revision"] = index
        writer.request_save()
    await writer.close()
    assert len(saved) == 1
    assert store.latest()["metadata"]["revision"] == 99


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_write_and_preserves_newer_snapshot(tmp_path):
    report = SessionReport("session-1", metadata={"revision": 1})
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    saved = []

    class Store(ReportStore):
        def save(self, snapshot):
            saved.append(snapshot)
            if len(saved) == 1:
                loop.call_soon_threadsafe(started.set)
                assert release.wait(timeout=5)
            return super().save(snapshot)

    store = Store(tmp_path)
    writer = ReportWriter(store, lambda: copy.deepcopy(report), interval=0)
    writer.start()
    writer.request_save()
    closing = None
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        # The event loop remains responsive while the disk worker is held in save().
        report.metadata["revision"] = 2
        writer.request_save()
        closing = asyncio.create_task(writer.close())
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
        await writer.close()
    if closing is not None:
        await closing
    assert [snapshot.metadata["revision"] for snapshot in saved] == [1, 2]
    assert store.latest()["metadata"]["revision"] == 2


@pytest.mark.asyncio
async def test_report_writer_failure_is_surfaced(tmp_path):
    class Store(ReportStore):
        def save(self, snapshot):
            raise OSError("disk is full")

    writer = ReportWriter(Store(tmp_path), lambda: SessionReport("session-1"))
    writer.start()
    writer.request_save()
    with pytest.raises(OSError, match="disk is full"):
        await writer.close()
    with pytest.raises(OSError, match="disk is full"):
        writer.request_save()
