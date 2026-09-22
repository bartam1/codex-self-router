"""Summaries of observed runs; repricing is never represented as a control run."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from .report import parse_timestamp, usage_cost

ROUTING_DELEGATION_CLASSES = ("neither", "router-only", "subagent-only", "both")


def routing_delegation_class(routing: bool, delegation: bool) -> str:
    if routing and delegation:
        return "both"
    if routing:
        return "router-only"
    if delegation:
        return "subagent-only"
    return "neither"


def routing_delegation_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [task for task in tasks if not task.get("isSubagentTask")]
    classified = [
        task for task in primary if task.get("delegationObservationCoverage") != "unavailable"
    ]
    classes = Counter(task["routingDelegationClass"] for task in classified)
    total = len(classified)
    routing = sum(bool(task["routerParticipationObserved"]) for task in classified)
    delegation = sum(bool(task["subagentDelegationObserved"]) for task in classified)
    observation_coverage = (
        "not-applicable"
        if not primary
        else (
            "unavailable"
            if not classified
            else (
                "complete"
                if all(task["delegationObservationCoverage"] == "complete" for task in classified)
                and len(classified) == len(primary)
                else "partial"
            )
        )
    )
    return {
        "tasks": len(primary),
        "classifiedTasks": total,
        "unclassifiedLegacyTasks": len(primary) - total,
        "delegationObservationCoverage": observation_coverage,
        "classes": {name: classes[name] for name in ROUTING_DELEGATION_CLASSES},
        "routerParticipationTasks": routing,
        "routerParticipationFraction": routing / total if total else None,
        "subagentDelegationTasks": delegation,
        "subagentDelegationFraction": delegation / total if total else None,
        "subagentDelegations": sum(task["subagentDelegations"] for task in classified),
        "subagentResponsesObserved": sum(task["subagentResponsesObserved"] for task in classified),
        "pricedSubagentResponses": sum(task["pricedSubagentResponses"] for task in classified),
        "delegationIntentAttribution": "not-observable-without-prompt-inspection",
        "subagentCostCoverage": (
            "not-applicable"
            if not delegation
            else (
                "partial"
                if any(task["pricedSubagentResponses"] for task in classified)
                else "unavailable"
            )
        ),
    }


def supports_delegation_observation(data: dict[str, Any]) -> bool:
    version = str(data.get("metadata", {}).get("routerVersion", ""))
    numbers = [int(part) for part in re.findall(r"\d+", version)[:3]]
    return tuple(numbers + [0] * (3 - len(numbers))) >= (0, 7, 0)


def interval_union_ms(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = float("-inf")
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return round(total, 3)


def response_summary(responses: list[dict], prices: dict) -> dict[str, Any]:
    costs = [usage_cost(row.get("usage"), prices.get(row.get("profile"))) for row in responses]
    tokens: Counter = Counter()
    reasoning_known = 0
    valid_usage = 0
    for row in responses:
        usage = row.get("usage") or {}
        # Token coverage is independent of whether the model has a known price.
        if (
            usage_cost(usage, {"input": 1, "cached_input": 1, "cache_write_input": 1, "output": 1})
            is None
        ):
            continue
        valid_usage += 1
        if type(usage.get("reasoningOutputTokens")) is int:
            reasoning_known += 1
        tokens.update(
            {key: value for key, value in usage.items() if type(value) is int and value >= 0}
        )
    return {
        "responses": len(responses),
        "tokens": dict(tokens),
        "reasoningUsageReportedResponses": reasoning_known,
        "validUsageResponses": valid_usage,
        "pricedResponses": sum(cost is not None for cost in costs),
        "unpricedResponses": sum(cost is None for cost in costs),
        "knownCostUsd": round(sum(cost for cost in costs if cost is not None), 8),
        "cacheHitFraction": (
            tokens["cachedInputTokens"] / tokens["inputTokens"] if tokens["inputTokens"] else None
        ),
    }


def analyze(data: dict[str, Any]) -> dict[str, Any]:
    measurements = data.get("measurements", {})
    turns = measurements.get("turns", [])
    turn_map = {(t["thread_id"], t["turn_id"]): t for t in turns}
    responses = data.get("responses", [])
    prices = data.get("pricingUsdPerMillionTokens", {})
    switches = data.get("switches", [])
    delegation_observation_supported = supports_delegation_observation(data)
    measurement_items = measurements.get("items", [])

    def is_delegation(item: dict) -> bool:
        if item.get("is_subagent_delegation") is not None:
            return item.get("is_subagent_delegation") is True
        return item.get("item_type") == "collabAgentToolCall"

    delegation_items = [item for item in measurement_items if is_delegation(item)]
    child_thread_ids = {
        str(thread_id)
        for item in delegation_items
        for thread_id in (item.get("new_thread_id"), item.get("receiver_thread_id"))
        if thread_id
    }
    tasks: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for turn in turns:
        tasks[(turn["thread_id"], turn["task_id"])].append(turn)

    def task_for(row: dict) -> tuple[str, str | None]:
        turn = turn_map.get((row.get("thread_id"), row.get("turn_id")), {})
        return row.get("thread_id", ""), turn.get("task_id")

    task_summaries = []
    for task_key, task_turns in tasks.items():
        ordered = sorted(task_turns, key=lambda t: t["start_ms"])
        last = ordered[-1]
        terminal = last.get("status") in {"completed", "failed", "interrupted"}
        complete = terminal and last.get("interruption_source") != "router"
        end = last.get("end_ms") if complete else None
        start = ordered[0]["start_ms"]
        relevant = [r for r in responses if task_for(r) == task_key]
        task_waits = [w for w in measurements.get("waits", []) if task_for(w) == task_key]
        # Only closed tasks get duration statistics; ongoing waits must not appear as zero.
        intervals = (
            [
                (
                    max(start, w["start_ms"]),
                    min(end, w["end_ms"] if w["end_ms"] is not None else end),
                )
                for w in task_waits
            ]
            if end is not None
            else []
        )
        wait_ms = interval_union_ms(intervals) if end is not None else None
        items = [i for i in measurement_items if task_for(i) == task_key]
        delegations = [item for item in items if is_delegation(item)]
        task_child_threads = sorted(
            {
                str(thread_id)
                for item in delegations
                for thread_id in (item.get("new_thread_id"), item.get("receiver_thread_id"))
                if thread_id
            }
        )
        subagent_responses = [
            response for response in responses if response.get("thread_id") in task_child_threads
        ]
        priced_subagent_responses = sum(
            usage_cost(response.get("usage"), prices.get(response.get("profile"))) is not None
            for response in subagent_responses
        )
        router_participation = any(
            switch.get("outcome") == "applied" and task_for(switch) == task_key
            for switch in switches
        )
        delegation_observed = bool(delegations)
        errors = [
            e
            for e in measurements.get("events", [])
            if e.get("kind") == "upstream_error" and task_for(e) == task_key
        ]
        interactions = [
            e
            for e in measurements.get("events", [])
            if e.get("kind") == "client_request" and task_for(e) == task_key
        ]
        completed_responses = [
            e["elapsed_ms"]
            for e in measurements.get("events", [])
            if e.get("kind") == "response_completed" and task_for(e) == task_key
        ]
        task_summaries.append(
            {
                "threadId": task_key[0],
                "taskId": task_key[1],
                "resumedFromSessionId": ordered[0].get("resumed_from_session_id"),
                "durationScope": "session-fragment"
                if ordered[0].get("resumed_from_session_id")
                else "task",
                "turnIds": [t["turn_id"] for t in ordered],
                "status": last.get("status"),
                "closed": complete,
                "wallMs": round(end - start, 3) if end is not None else None,
                "userWaitMs": wait_ms,
                "wallExcludingUserWaitMs": round(end - start - wait_ms, 3)
                if end is not None
                else None,
                "toolItems": len(items),
                "failedToolItems": sum(bool(i.get("failed")) for i in items),
                "isSubagentTask": task_key[0] in child_thread_ids,
                "delegationObservationCoverage": (
                    "complete" if delegation_observation_supported else "unavailable"
                ),
                "routerParticipationObserved": router_participation,
                "subagentDelegationObserved": delegation_observed,
                "routingDelegationClass": routing_delegation_class(
                    router_participation, delegation_observed
                ),
                "subagentDelegations": len(delegations),
                "subagentChildThreads": task_child_threads,
                "subagentResponsesObserved": len(subagent_responses),
                "pricedSubagentResponses": priced_subagent_responses,
                "subagentCostCoverage": (
                    "not-applicable"
                    if not delegations
                    else ("partial" if priced_subagent_responses else "unavailable")
                ),
                "upstreamErrors": len(errors),
                "announcedRetries": sum(e.get("will_retry") is True for e in errors),
                "userSteers": sum(e.get("method") == "turn/steer" for e in interactions),
                "timeToFirstCompletedResponseMs": round(min(completed_responses) - start, 3)
                if completed_responses
                else None,
                "routerInterruptions": sum(
                    t.get("interruption_source") == "router" for t in ordered
                ),
                "userInterruptions": sum(t.get("interruption_source") == "user" for t in ordered),
                "feedback": data.get("feedback", {}).get(task_key[1]),
                **response_summary(relevant, prices),
            }
        )
    task_summary_map = {(t["threadId"], t["taskId"]): t for t in task_summaries}
    phases = []
    applied = [
        s
        for s in switches
        if s.get("outcome") == "applied" and type(s.get("response_index")) is int
    ]
    for index, switch in enumerate(applied):
        task_key = task_for(switch)
        if task_key[1] is None:
            continue
        next_switch = next((s for s in applied[index + 1 :] if task_for(s) == task_key), None)
        stop = next_switch["response_index"] if next_switch else len(responses)
        phase = [r for r in responses[switch["response_index"] : stop] if task_for(r) == task_key]
        task_responses = [r for r in responses if task_for(r) == task_key]
        if any(r.get("usage_source") == "codex-rollout" for r in task_responses):
            # A rollout can flush the requesting response after a switch was
            # recorded. File-arrival indexes are not sampling boundaries.
            start_at = parse_timestamp(switch["timestamp"])
            stop_at = parse_timestamp(next_switch["timestamp"]) if next_switch else None
            phase = [
                r
                for r in task_responses
                if parse_timestamp(r["timestamp"]) >= start_at
                and (stop_at is None or parse_timestamp(r["timestamp"]) < stop_at)
            ]
        summary = response_summary(phase, prices)
        previous_price = prices.get(switch.get("from_profile"))
        previous_costs = [usage_cost(r.get("usage"), previous_price) for r in phase]
        known_pairs = (
            all(c is not None for c in previous_costs) and not summary["unpricedResponses"]
        )
        estimated = switch.get("estimated_follow_up_steps")
        closed = next_switch is not None or task_summary_map[task_key]["closed"]
        decision = next(
            (
                r
                for r in responses
                if r.get("response_id") == switch.get("decision_response_id")
                and r.get("response_id")
            ),
            None,
        )
        phases.append(
            {
                "switchId": switch.get("switch_id"),
                "taskId": task_key[1],
                "fromProfile": switch.get("from_profile"),
                "toProfile": switch.get("to_profile"),
                "fromEffort": switch.get("from_effort"),
                "toEffort": switch.get("to_effort"),
                "source": switch.get("source"),
                "closed": closed,
                "estimatedSteps": estimated,
                "observedSteps": len(phase),
                "stepEstimateError": len(phase) - estimated
                if closed and estimated is not None
                else None,
                "approvalMs": switch.get("approval_ms"),
                "applyMs": switch.get("apply_ms"),
                "totalSwitchMs": switch.get("total_ms"),
                "decisionResponseCostUsd": usage_cost(
                    decision.get("usage"), prices.get(decision.get("profile"))
                )
                if decision
                else None,
                "firstResponseCacheHitFraction": response_summary(phase[:1], prices)[
                    "cacheHitFraction"
                ],
                "nextProfile": next(
                    (s.get("to_profile") for s in applied[index + 1 :] if task_for(s) == task_key),
                    None,
                ),
                "nextEffort": next(
                    (s.get("to_effort") for s in applied[index + 1 :] if task_for(s) == task_key),
                    None,
                ),
                "sameUsageSavingsVsPreviousProfileUsd": round(
                    sum(previous_costs) - summary["knownCostUsd"], 8
                )
                if known_pairs
                else None,
                **summary,
            }
        )
    by_profile = {
        profile or "unknown": response_summary(
            [r for r in responses if r.get("profile") == profile], prices
        )
        for profile in {r.get("profile") for r in responses}
    }
    routing_delegation = routing_delegation_summary(task_summaries)
    return {
        "coverage": {
            "schemaVersion": data.get("schemaVersion", 1),
            "responsesWithTask": sum(task_for(r)[1] is not None for r in responses),
            "responsesWithoutTask": sum(task_for(r)[1] is None for r in responses),
            "independentlyVerifiedModelResponses": 0,
        },
        "usage": response_summary(responses, prices),
        "byProfile": by_profile,
        "switchOutcomes": dict(Counter(s.get("outcome", "unknown") for s in switches)),
        "routingDelegation": routing_delegation,
        "tasks": task_summaries,
        "phases": phases,
    }


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for report in reports:
        metadata = report.get("metadata", {})
        groups[
            (metadata.get("label") or "unlabelled", metadata.get("routingMode", "legacy"))
        ].append(report)
    rows = []
    for (label, mode), runs in sorted(groups.items()):
        analyses = [analyze(run) for run in sorted(runs, key=lambda run: run.get("startedAt", ""))]
        fragments = [task for analysis in analyses for task in analysis["tasks"]]
        tasks = []
        by_task = defaultdict(list)
        for fragment in fragments:
            by_task[(fragment["threadId"], fragment["taskId"])].append(fragment)
        for parts in by_task.values():
            if len(parts) == 1:
                single = dict(parts[0])
                if single.get("durationScope") == "session-fragment":
                    single["wallMs"] = None
                tasks.append(single)
                continue
            closed = [part for part in parts if part["closed"]]
            combined = {**(closed[-1] if closed else parts[-1])}
            for field in (
                "responses",
                "knownCostUsd",
                "unpricedResponses",
                "failedToolItems",
                "announcedRetries",
                "subagentDelegations",
                "subagentResponsesObserved",
                "pricedSubagentResponses",
            ):
                combined[field] = sum(p[field] for p in parts)
            combined["routerParticipationObserved"] = any(
                p["routerParticipationObserved"] for p in parts
            )
            combined["subagentDelegationObserved"] = any(
                p["subagentDelegationObserved"] for p in parts
            )
            combined["routingDelegationClass"] = routing_delegation_class(
                combined["routerParticipationObserved"],
                combined["subagentDelegationObserved"],
            )
            combined["subagentChildThreads"] = sorted(
                {thread for part in parts for thread in part["subagentChildThreads"]}
            )
            combined["subagentCostCoverage"] = (
                "not-applicable"
                if not combined["subagentDelegations"]
                else ("partial" if combined["pricedSubagentResponses"] else "unavailable")
            )
            coverages = {part["delegationObservationCoverage"] for part in parts}
            combined["delegationObservationCoverage"] = (
                "complete"
                if coverages == {"complete"}
                else ("unavailable" if coverages == {"unavailable"} else "partial")
            )
            # Reconnect gaps and unfinished fragments are not measured task latency.
            combined["wallMs"] = None
            combined["feedback"] = max(
                (p["feedback"] for p in parts if p.get("feedback")),
                key=lambda feedback: feedback.get("timestamp", ""),
                default=None,
            )
            tasks.append(combined)
        durations = [t["wallMs"] for t in tasks if t["wallMs"] is not None]
        phases = [p for a in analyses for p in a["phases"]]
        errors = [abs(p["stepEstimateError"]) for p in phases if p["stepEstimateError"] is not None]
        feedback = [t["feedback"] for t in tasks if t.get("feedback")]
        priced_closed = [
            t for t in tasks if t["closed"] and t["responses"] > 0 and t["unpricedResponses"] == 0
        ]
        success_count = sum(f.get("outcome") == "success" for f in feedback)
        routing_delegation = routing_delegation_summary(tasks)
        rows.append(
            {
                "label": label,
                "routingMode": mode,
                "sessions": len(runs),
                "tasks": len(tasks),
                "closedTasks": sum(t["closed"] for t in tasks),
                "ratedTasks": len(feedback),
                "fullyPricedClosedTasks": len(priced_closed),
                "meanCostPerFullyPricedClosedTaskUsd": mean(
                    t["knownCostUsd"] for t in priced_closed
                )
                if priced_closed
                else None,
                "successFractionAmongRatedTasks": success_count / len(feedback)
                if feedback
                else None,
                "taskOutcomes": dict(Counter(f.get("outcome", "unknown") for f in feedback)),
                "testOutcomes": dict(Counter(f.get("tests", "unknown") for f in feedback)),
                "knownCostUsd": round(sum(a["usage"]["knownCostUsd"] for a in analyses), 8),
                "unpricedResponses": sum(a["usage"]["unpricedResponses"] for a in analyses),
                "responses": sum(a["usage"]["responses"] for a in analyses),
                "medianTaskWallMs": median(durations) if durations else None,
                "meanAbsoluteStepEstimateError": mean(errors) if errors else None,
                "closedPhases": sum(p["closed"] for p in phases),
                "shortPhases": sum(p["closed"] and p["observedSteps"] <= 1 for p in phases),
                "switchBacksWithinTask": sum(
                    p["nextProfile"] == p["fromProfile"] and p["fromProfile"] != p["toProfile"]
                    for p in phases
                    if p["nextProfile"] is not None
                ),
                "switchOutcomes": dict(
                    Counter(
                        s.get("outcome", "unknown") for run in runs for s in run.get("switches", [])
                    )
                ),
                "routingDelegation": routing_delegation,
                "failedToolItems": sum(t["failedToolItems"] for t in tasks),
                "announcedRetries": sum(t["announcedRetries"] for t in tasks),
                "approvalMs": sum(
                    s.get("approval_ms") or 0 for run in runs for s in run.get("switches", [])
                ),
                "applyMs": sum(
                    s.get("apply_ms") or 0 for run in runs for s in run.get("switches", [])
                ),
                "policyVersions": sorted(
                    {r.get("metadata", {}).get("policySha256", "unknown") for r in runs}
                ),
            }
        )
    return {
        "groups": rows,
        "notes": [
            "Groups are observed runs, not matched causal experiments. "
            "Use comparable tasks and initial repository state.",
            "Costs use each session's pricing snapshot; missing usage is excluded and counted.",
            "Open tasks are excluded from duration and step-estimate accuracy statistics.",
            "Task and test outcomes come from explicit feedback; unrated tasks remain unknown.",
            "Delegation intent is not inferred because reports intentionally do not store prompts.",
            "Subagent cost coverage is partial or unavailable unless child response usage and "
            "model attribution are both observed.",
        ],
    }
