from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from codex_self_router import measurement
from codex_self_router.cli import async_main, build_parser
from codex_self_router.evaluation import aggregate, analyze
from codex_self_router.measurement import Measurements
from codex_self_router.report import ReportStore, SessionReport, SwitchRecord, UsageRecord


def usage(profile="astra", turn="turn-2", response="response-1", tokens=None):
    return UsageRecord(
        "2026-01-01T00:00:00+00:00",
        "thread-1",
        turn,
        response,
        profile,
        "model",
        tokens
        if tokens is not None
        else {
            "inputTokens": 1000,
            "cachedInputTokens": 500,
            "outputTokens": 100,
            "reasoningOutputTokens": 80,
        },
    )


@pytest.fixture
def measured(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(measurement, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    events = []
    observer = Measurements(events.append)
    report = SessionReport("session-1")
    observer.start_turn("thread-1", "turn-1")
    clock[0] = 1
    observer.start_wait("approval", "thread-1", "turn-1", "model-switch-approval")
    clock[0] = 2
    observer.start_wait("overlap", "thread-1", "turn-1", "tool-approval")
    clock[0] = 4
    observer.resolve_wait("approval")
    observer.resolve_wait("overlap")
    observer.interruptions[("thread-1", "turn-1")] = "router"
    observer.observe(
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread-1",
                "turn": {"id": "turn-1", "status": "interrupted"},
            },
        }
    )
    observer.continuations["thread-1"] = "turn-1"
    clock[0] = 5
    observer.start_turn("thread-1", "turn-2")
    report.responses = [usage(profile="sol", turn="turn-1", response="decision"), usage()]
    report.switches = [
        SwitchRecord(
            "now",
            "thread-1",
            "turn-2",
            "sol",
            "astra",
            "agent-tool",
            "applied",
            estimated_follow_up_steps=2,
            response_index=1,
            origin_turn_id="turn-1",
            decision_response_id="decision",
            approval_ms=3000,
            apply_ms=1000,
        )
    ]
    clock[0] = 6
    params = {
        "threadId": "thread-1",
        "turnId": "turn-2",
        "item": {
            "type": "commandExecution",
            "id": "cmd",
            "command": "secret command",
            "aggregatedOutput": "secret output",
            "status": "completed",
            "exitCode": 1,
            "durationMs": 300,
        },
    }
    observer.observe({"method": "item/completed", "params": params})
    observer.observe({"method": "item/completed", "params": params})  # duplicate lifecycle event
    observer.observe(
        {
            "method": "error",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-2",
                "willRetry": True,
                "error": {"message": "secret provider diagnostic", "codexErrorInfo": "streamError"},
            },
        }
    )
    clock[0] = 8
    observer.observe(
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread-1",
                "turn": {"id": "turn-2", "status": "completed"},
            },
        }
    )
    report.measurements = observer.to_dict()
    return report, observer, events


def test_continuations_form_one_task_with_union_of_approval_waits(measured):
    report, _, events = measured
    result = analyze(report.to_dict())
    (task,) = result["tasks"]
    assert task["taskId"] == "turn-1"
    assert task["turnIds"] == ["turn-1", "turn-2"]
    assert task["wallMs"] == 8000
    assert task["userWaitMs"] == 3000  # overlapping waits must not be added twice
    assert task["wallExcludingUserWaitMs"] == 5000
    assert task["routerInterruptions"] == 1
    assert task["userInterruptions"] == 0
    assert task["failedToolItems"] == 1
    assert task["announcedRetries"] == 1
    assert task["feedback"] is None  # a completed turn is not a success rating
    (phase,) = result["phases"]
    assert phase["observedSteps"] == 1  # exclude the switch-requesting response
    assert phase["stepEstimateError"] == -1
    assert phase["decisionResponseCostUsd"] == pytest.approx(0.0042)
    assert phase["tokens"]["reasoningOutputTokens"] == 80
    assert phase["knownCostUsd"] == pytest.approx(0.0105)
    assert "secret" not in json.dumps(events)


def test_open_phase_and_following_user_task_do_not_contaminate_estimate(measured):
    report, observer, _ = measured
    observer.turns[("thread-1", "turn-2")].update(status="disconnected", end_ms=None)
    observer.start_turn("thread-1", "turn-3")
    report.responses.append(usage(turn="turn-3", response="next-task"))
    report.measurements = observer.to_dict()
    result = analyze(report.to_dict())
    assert len(result["tasks"]) == 2
    (phase,) = result["phases"]
    assert phase["observedSteps"] == 1
    assert phase["stepEstimateError"] is None
    assert not phase["closed"]
    assert all(task["wallMs"] is None for task in result["tasks"])


def test_missing_usage_unknown_models_and_invalid_buckets_remain_unpriced():
    report = SessionReport("s")
    absent = usage()
    absent.usage = None
    unknown = usage(profile=None)
    invalid = usage(tokens={"inputTokens": 100, "cachedInputTokens": 200, "outputTokens": 5})
    incomplete = usage(tokens={"outputTokens": 5})
    report.responses = [usage(), absent, unknown, invalid, incomplete]
    costs = report.costs()
    assert not costs["complete"]
    assert costs["pricedResponses"] == 1
    assert costs["unpricedResponses"] == 4
    assert costs["routedApiEquivalentUsd"] == pytest.approx(0.0105)
    result = analyze(report.to_dict())
    assert result["usage"]["unpricedResponses"] == 4
    assert result["coverage"]["responsesWithoutTask"] == 5


def test_aggregate_uses_original_prices_and_separates_control_cohorts(measured):
    report, _, _ = measured
    report.metadata = {"label": "coding", "routingMode": "auto"}
    first = report.to_dict()
    first["feedback"] = {"turn-1": {"outcome": "rework", "tests": "failed"}}
    second = copy.deepcopy(first)
    second["metadata"]["routingMode"] = "fixed-sol"
    for price in second["pricingUsdPerMillionTokens"].values():
        for field in price:
            price[field] *= 2
    result = aggregate([first, second])
    auto, fixed = result["groups"]
    assert auto["taskOutcomes"] == {"rework": 1}
    assert auto["ratedTasks"] == 1
    assert fixed["knownCostUsd"] == pytest.approx(auto["knownCostUsd"] * 2)
    assert auto["meanAbsoluteStepEstimateError"] == 1


@pytest.mark.asyncio
async def test_feedback_survives_live_report_save_and_cli_reports_it(measured, tmp_path, capsys):
    report, _, _ = measured
    store = ReportStore(tmp_path)
    store.save(report)
    parser = build_parser()
    await async_main(
        parser.parse_args(
            [
                "--report-dir",
                str(tmp_path),
                "feedback",
                "--session",
                "session-1",
                "--task",
                "turn-1",
                "--outcome",
                "success",
                "--tests",
                "passed",
            ]
        )
    )
    store.save(report)  # live router writes must not erase a user's assessment
    assert store.latest()["feedback"]["turn-1"]["outcome"] == "success"
    capsys.readouterr()
    await async_main(
        parser.parse_args(["--report-dir", str(tmp_path), "report", "--all", "--json"])
    )
    result = json.loads(capsys.readouterr().out)
    assert result["groups"][0]["ratedTasks"] == 1
    with pytest.raises(ValueError, match="task id"):
        store.feedback("session-1", "wrong-task", {"outcome": "success"})
    with pytest.raises(ValueError, match="session id"):
        store.path_for("../outside")


def test_legacy_reports_have_no_invented_tasks():
    result = aggregate([{"responses": [], "switches": []}])
    assert result["groups"][0]["routingMode"] == "legacy"
    assert result["groups"][0]["tasks"] == 0
    assert result["groups"][0]["medianTaskWallMs"] is None


def test_run_parses_switch_approval_override():
    args = build_parser().parse_args(["run", "--switch-approval", "never", "--", "--full-auto"])
    assert args.switch_approval == "never"
    assert args.codex_args == ["--", "--full-auto"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--disable-agent-switching", "run"],
        ["run", "--disable-agent-switching"],
        ["serve", "--disable-agent-switching"],
        ["doctor", "--disable-agent-switching"],
    ],
)
def test_disable_agent_switching_override_parses_before_or_after_command(arguments):
    args = build_parser().parse_args(arguments)
    assert args.disable_agent_switching is True
