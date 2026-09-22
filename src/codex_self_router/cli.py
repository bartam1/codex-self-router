from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from websockets.asyncio.server import Server
from websockets.asyncio.server import serve as websocket_serve

from .config import (
    DIRECTIVE_ROUTES,
    PROFILES,
    ProfileName,
    SwitchApproval,
    default_config_path,
    default_config_yaml,
    get_agent_switching_enabled,
    get_switch_approval,
    load_config,
    set_agent_switching_enabled,
    set_switch_approval,
)
from .evaluation import aggregate, analyze
from .proxy import Bridge
from .report import ReportStore

MINIMUM_CODEX_VERSION = (0, 154, 0)
MACOS_APP_CODEX = Path("/Applications/Codex.app/Contents/Resources/codex")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex-self-router",
        description="Run Codex through a local model and reasoning-effort router.",
    )
    parser.add_argument("--verbose", action="store_true", help="show app-server diagnostics")
    parser.add_argument("--codex-bin", type=Path, help="path to a recent Codex CLI binary")
    parser.add_argument("--report-dir", type=Path, help="directory for JSON session reports")
    parser.add_argument(
        "--config",
        type=Path,
        help=f"router YAML configuration (default: {default_config_path()})",
    )
    parser.add_argument(
        "--disable-agent-switching",
        action="store_true",
        help="disable agent-requested route changes for this process",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_agent_switching_override(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--disable-agent-switching",
            action="store_true",
            default=argparse.SUPPRESS,
            help="disable agent-requested route changes for this process",
        )

    run = subparsers.add_parser("run", help="start the router and a connected Codex TUI")
    add_agent_switching_override(run)
    run.add_argument("--label", help="evaluation cohort or benchmark task label")
    run.add_argument(
        "--fixed-profile", choices=list(ProfileName), help="disable routing for a control run"
    )
    run.add_argument(
        "--switch-approval",
        choices=list(SwitchApproval),
        help="override agent switch approval policy for this run",
    )
    run.add_argument(
        "codex_args",
        nargs=argparse.REMAINDER,
        help="arguments passed to Codex after -- (for example: -- --full-auto)",
    )
    run.add_argument("--skip-version-check", action="store_true", help=argparse.SUPPRESS)

    serve = subparsers.add_parser("serve", help="serve a WebSocket endpoint for codex --remote")
    add_agent_switching_override(serve)
    serve.add_argument("--listen", default="127.0.0.1:4501", metavar="HOST:PORT")
    serve.add_argument("--label", help="evaluation cohort or benchmark task label")
    serve.add_argument(
        "--fixed-profile", choices=list(ProfileName), help="disable routing for a control run"
    )
    serve.add_argument(
        "--switch-approval",
        choices=list(SwitchApproval),
        help="override agent switch approval policy for this server",
    )
    serve.add_argument("--skip-version-check", action="store_true", help=argparse.SUPPRESS)

    report = subparsers.add_parser("report", help="show the latest routing and cost report")
    report.add_argument("--json", action="store_true", help="print the complete JSON report")
    report_selection = report.add_mutually_exclusive_group()
    report_selection.add_argument("--session", help="session id (default: latest)")
    report_selection.add_argument(
        "--all", action="store_true", help="aggregate all sessions by label and routing mode"
    )
    report.add_argument("--label", help="filter aggregate report by evaluation label")

    feedback = subparsers.add_parser(
        "feedback", help="record a human assessment for a measured task"
    )
    feedback.add_argument("--session", required=True)
    feedback.add_argument("--task", required=True)
    feedback.add_argument("--outcome", choices=["success", "rework", "failed"], required=True)
    feedback.add_argument(
        "--tests", choices=["passed", "failed", "not-run", "unknown"], default="unknown"
    )
    feedback.add_argument("--note", help="optional assessment, saved locally")

    doctor = subparsers.add_parser(
        "doctor", help="check the local Codex binary and router profiles"
    )
    add_agent_switching_override(doctor)
    init_config = subparsers.add_parser("init-config", help="write a default router config.yaml")
    init_config.add_argument("--force", action="store_true", help="replace an existing config")
    return parser


def resolve_codex_binary(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Codex binary does not exist: {path}")
        return path
    found = shutil.which("codex")
    if found:
        return Path(found).resolve()
    if MACOS_APP_CODEX.is_file():
        return MACOS_APP_CODEX
    raise RuntimeError("cannot find Codex; pass --codex-bin /path/to/codex")


async def codex_version(codex_bin: Path) -> tuple[str, tuple[int, int, int]]:
    process = await asyncio.create_subprocess_exec(
        str(codex_bin),
        "--version",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    output = (stdout or stderr).decode(errors="replace").strip()
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output)
    if process.returncode != 0 or match is None:
        raise RuntimeError(f"cannot determine Codex version from: {output!r}")
    return output, tuple(int(group) for group in match.groups())


async def require_compatible_codex(codex_bin: Path, skip: bool = False) -> str:
    display, version = await codex_version(codex_bin)
    if not skip and version < MINIMUM_CODEX_VERSION:
        required = ".".join(str(item) for item in MINIMUM_CODEX_VERSION)
        raise RuntimeError(
            f"{display} is too old; codex-self-router needs Codex >= {required} "
            "for turn/settings/update and step_model_switching"
        )
    return display


def parse_listen(value: str) -> tuple[str, int]:
    host, separator, port_text = value.rpartition(":")
    if not separator or not host:
        raise RuntimeError("--listen must be HOST:PORT")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise RuntimeError("listen port must be an integer") from exc
    if not 0 <= port <= 65535:
        raise RuntimeError("listen port must be between 0 and 65535")
    return host, port


async def make_server(
    codex_bin: Path,
    host: str,
    port: int,
    report_store: ReportStore,
    metadata: dict[str, Any] | None = None,
    fixed_profile: ProfileName | None = None,
) -> Server:
    async def handler(websocket: Any) -> None:
        bridge = Bridge(
            websocket,
            codex_bin=codex_bin,
            report_store=report_store,
            metadata=metadata,
            fixed_profile=fixed_profile,
        )
        await bridge.run()

    return await websocket_serve(handler, host, port, max_size=None)


async def command_serve(args: argparse.Namespace, codex_bin: Path, store: ReportStore) -> int:
    display = await require_compatible_codex(codex_bin, args.skip_version_check)
    host, port = parse_listen(args.listen)
    server = await make_server(
        codex_bin,
        host,
        port,
        store,
        {"codexVersion": display, "label": args.label},
        ProfileName(args.fixed_profile) if args.fixed_profile else None,
    )
    addresses = ", ".join(
        f"ws://{sock.getsockname()[0]}:{sock.getsockname()[1]}" for sock in server.sockets
    )
    print(f"codex-self-router listening on {addresses} ({display})")
    try:
        await server.serve_forever()
    except asyncio.CancelledError:
        pass
    finally:
        server.close()
        await server.wait_closed()
    return 0


async def command_run(args: argparse.Namespace, codex_bin: Path, store: ReportStore) -> int:
    display = await require_compatible_codex(codex_bin, args.skip_version_check)
    server = await make_server(
        codex_bin,
        "127.0.0.1",
        0,
        store,
        {"codexVersion": display, "label": args.label},
        ProfileName(args.fixed_profile) if args.fixed_profile else None,
    )
    socket = server.sockets[0]
    host, port = socket.getsockname()[:2]
    remote = f"ws://{host}:{port}"
    codex_args = list(args.codex_args)
    if codex_args and codex_args[0] == "--":
        codex_args.pop(0)
    print(f"routing {display} through {remote}", file=sys.stderr)
    process = await asyncio.create_subprocess_exec(str(codex_bin), "--remote", remote, *codex_args)
    try:
        return await process.wait()
    finally:
        server.close()
        await server.wait_closed()


def print_report(data: dict[str, Any], as_json: bool) -> int:
    data = {**data, "analysis": analyze(data)}
    if as_json:
        print(json.dumps(data, indent=2))
        return 0
    costs = data.get("costs", {})
    responses = data.get("responses", [])
    switches = data.get("switches", [])
    print(f"Session: {data.get('sessionId')}")
    print(f"Responses: {len(responses)}")
    print(f"Switches/directives: {len(switches)}")
    print(f"Routed API-equivalent: ${costs.get('routedApiEquivalentUsd', 0):.6f}")
    print(f"Same usage, all Sol:    ${costs.get('sameObservedUsageAllSolUsd', 0):.6f}")
    print(f"Same usage, all Astra:  ${costs.get('sameObservedUsageAllAstraUsd', 0):.6f}")
    print(f"Estimated vs all Sol:   ${costs.get('estimatedSavingsVsAllSolUsd', 0):.6f}")
    analysis = data["analysis"]
    print(
        f"Usage coverage: {analysis['usage']['pricedResponses']}/{len(responses)} priced; "
        f"reasoning reported for {analysis['usage']['reasoningUsageReportedResponses']}"
    )
    routing_delegation = analysis["routingDelegation"]
    classes = routing_delegation["classes"]
    print(
        "Routing/delegation: "
        + ", ".join(f"{name}={count}" for name, count in classes.items())
        + f"; observation coverage={routing_delegation['delegationObservationCoverage']}"
        + f"; subagent cost coverage={routing_delegation['subagentCostCoverage']}"
    )
    for task in analysis["tasks"]:
        seconds = f"{task['wallMs'] / 1000:.1f}s" if task["wallMs"] is not None else "open/unknown"
        outcome = (task.get("feedback") or {}).get("outcome", "unrated")
        print(
            f"Task {task['taskId']}: {task['responses']} responses, {seconds}, "
            f"status={task['status']}, outcome={outcome}, "
            f"routing/delegation={task['routingDelegationClass']}, "
            f"failed tools={task['failedToolItems']}"
        )
    for phase in analysis["phases"]:
        print(
            f"Phase {phase['fromProfile']}/{phase.get('fromEffort')} → "
            f"{phase['toProfile']}/{phase.get('toEffort')}: "
            f"estimated={phase['estimatedSteps']}, observed={phase['observedSteps']}, "
            f"closed={phase['closed']}, approval={phase['approvalMs']}ms, "
            f"apply={phase['applyMs']}ms"
        )
    print("Note: estimates are not ChatGPT subscription charges or true counterfactual runs.")
    return 0


async def async_main(args: argparse.Namespace) -> int:
    if args.command == "init-config":
        destination = (args.config or default_config_path()).expanduser().resolve()
        if destination.exists() and not args.force:
            raise ValueError(f"config already exists: {destination}; use --force to replace it")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(default_config_yaml(), encoding="utf-8")
        print(f"Wrote {destination}")
        return 0
    loaded_config = load_config(args.config)
    if args.disable_agent_switching:
        set_agent_switching_enabled(False)
    if args.command in {"run", "serve"} and args.switch_approval:
        set_switch_approval(args.switch_approval)
    store = ReportStore(args.report_dir)
    if args.command == "feedback":
        store.feedback(
            args.session,
            args.task,
            {"outcome": args.outcome, "tests": args.tests, "note": args.note},
        )
        print(f"Saved feedback for task {args.task} in session {args.session}.")
        return 0
    if args.command == "report":
        if args.label and not args.all:
            raise ValueError("--label requires --all")
        if args.all:
            reports = store.all()
            if args.label:
                reports = [r for r in reports if r.get("metadata", {}).get("label") == args.label]
            result = aggregate(reports)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                for group in result["groups"]:
                    print(
                        f"{group['label']} / {group['routingMode']}: {group['sessions']} sessions, "
                        f"{group['tasks']} tasks ({group['ratedTasks']} rated), "
                        f"known cost=${group['knownCostUsd']:.6f}, "
                        f"unpriced responses={group['unpricedResponses']}, "
                        f"outcomes={group['taskOutcomes']}"
                    )
                    routing_delegation = group["routingDelegation"]
                    classes = routing_delegation["classes"]
                    print(
                        "  routing/delegation: "
                        + ", ".join(f"{name}={count}" for name, count in classes.items())
                        + "; classified="
                        + f"{routing_delegation['classifiedTasks']}/{routing_delegation['tasks']}"
                        + "; observation coverage="
                        + routing_delegation["delegationObservationCoverage"]
                        + "; delegation rate="
                        + (
                            f"{routing_delegation['subagentDelegationFraction']:.1%}"
                            if routing_delegation["subagentDelegationFraction"] is not None
                            else "n/a"
                        )
                        + "; router participation="
                        + (
                            f"{routing_delegation['routerParticipationFraction']:.1%}"
                            if routing_delegation["routerParticipationFraction"] is not None
                            else "n/a"
                        )
                        + f"; subagent cost coverage="
                        f"{routing_delegation['subagentCostCoverage']}"
                    )
                print("Observed cohorts; compare task mix and quality before claiming savings.")
            return 0
        data = store.read(store.path_for(args.session)) if args.session else store.latest()
        if data is None:
            print(f"No reports found in {store.report_dir}", file=sys.stderr)
            return 1
        return print_report(data, args.json)

    codex_bin = resolve_codex_binary(args.codex_bin)
    if args.command == "doctor":
        display, version = await codex_version(codex_bin)
        print(f"Codex: {display} ({codex_bin})")
        print(f"Config: {loaded_config or 'built-in defaults'}")
        status = "compatible" if version >= MINIMUM_CODEX_VERSION else "too old"
        print(f"App-server switching support: {status}")
        print(
            "Agent-requested switching: "
            + ("enabled" if get_agent_switching_enabled() else "disabled")
        )
        print(f"Agent switch approval: {get_switch_approval().value}")
        for marker, (name, effort) in DIRECTIVE_ROUTES.items():
            if marker.startswith("#"):
                continue
            print(f"{marker} {name}: {PROFILES[name].model} / {effort}")
        return 0 if status == "compatible" else 1
    if args.command == "serve":
        return await command_serve(args, codex_bin, store)
    if args.command == "run":
        return await command_run(args, codex_bin, store)
    raise AssertionError(f"unhandled command {args.command}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        raise SystemExit(asyncio.run(async_main(args)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
