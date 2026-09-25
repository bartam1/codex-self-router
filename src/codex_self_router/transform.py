from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from .config import (
    DIRECTIVE_ROUTES,
    PROFILES,
    ROUTER_NAMESPACE,
    ROUTER_STATE_TOOL,
    ROUTER_TOOL,
    ROUTING_TOOL_SPEC,
    ProfileName,
    get_default_profile,
    merge_routing_policy,
    profile_for_model,
)

# Routing directives are recognized as either the first non-whitespace token or the final
# standalone token. Exact token lookup keeps similar identifiers ordinary text.
_LEADING_DIRECTIVE = re.compile(r"^(?P<leading>\s*)(?P<marker>\S+)(?=\s|$)(?P<rest>[\s\S]*)$")
_TRAILING_DIRECTIVE = re.compile(
    r"^(?P<body>[\s\S]*\S)(?P<separator>\s+)(?P<marker>\S+)(?P<trailing>\s*)$"
)


@dataclass(frozen=True, slots=True)
class DirectiveResult:
    profile: ProfileName | None
    effort: str | None
    marker: str | None
    text: str
    temporary: bool = False


@dataclass(frozen=True, slots=True)
class TurnRoute:
    profile: ProfileName
    effort: str
    source: str
    marker: str | None = None
    temporary: bool = False


def _route_for_marker(marker: str) -> tuple[ProfileName, str, bool] | None:
    temporary = marker.endswith("~")
    base = marker[:-1] if temporary else marker
    route = DIRECTIVE_ROUTES.get(base)
    return (*route, temporary) if route is not None else None


def _apply_directive_to_first_text(params: dict[str, Any]) -> DirectiveResult:
    for item in params.get("input") or []:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        parsed = parse_directive(item.get("text", ""))
        if parsed.profile is not None:
            item["text"] = parsed.text
        return parsed
    return DirectiveResult(None, None, None, "")


def parse_directive(text: str) -> DirectiveResult:
    leading_match = _LEADING_DIRECTIVE.match(text)
    if leading_match is not None and (route := _route_for_marker(leading_match.group("marker"))):
        marker = leading_match.group("marker")
        profile, effort, temporary = route
        # Remove the control marker from model-visible content while preserving intentional
        # leading whitespace and avoiding a surprising leading blank line.
        rest = leading_match.group("rest")
        if rest.startswith("\r\n"):
            rest = rest[2:]
        elif rest.startswith(("\n", " ", "\t")):
            rest = rest[1:]
        return DirectiveResult(
            profile, effort, marker, leading_match.group("leading") + rest, temporary
        )

    trailing_match = _TRAILING_DIRECTIVE.match(text)
    if trailing_match is not None and (route := _route_for_marker(trailing_match.group("marker"))):
        marker = trailing_match.group("marker")
        profile, effort, temporary = route
        return DirectiveResult(
            profile,
            effort,
            marker,
            trailing_match.group("body") + trailing_match.group("trailing"),
            temporary,
        )
    return DirectiveResult(None, None, None, text)


def enable_experimental_api(message: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(message)
    params = result.setdefault("params", {})
    capabilities = params.setdefault("capabilities", {})
    capabilities["experimentalApi"] = True
    return result


def prepare_thread_start(
    message: dict[str, Any],
    fixed_profile: ProfileName | None = None,
) -> tuple[dict[str, Any], ProfileName]:
    result = copy.deepcopy(message)
    params = result.setdefault("params", {})

    explicit = profile_for_model(params.get("model"))
    if params.get("model") and explicit is None and fixed_profile is None:
        raise ValueError(f"unsupported model override: {params['model']}")
    selected = fixed_profile or explicit or get_default_profile()
    if fixed_profile or (explicit is None and not params.get("model")):
        profile = PROFILES[selected]
        params["model"] = profile.model

    tools = list(params.get("dynamicTools") or [])
    tools = [
        tool
        for tool in tools
        if not (tool.get("type") == "namespace" and tool.get("name") == ROUTER_NAMESPACE)
    ]
    if fixed_profile is None:
        tools.append(copy.deepcopy(ROUTING_TOOL_SPEC))
    params["dynamicTools"] = tools
    params["experimentalRawEvents"] = True

    if fixed_profile is not None:
        return result, selected

    params["developerInstructions"] = merge_routing_policy(params.get("developerInstructions"))
    return result, selected


def prepare_turn_start(
    message: dict[str, Any],
    fixed_profile: ProfileName | None = None,
    resumed_profile: ProfileName | None = None,
    resumed_effort: str | None = None,
) -> tuple[dict[str, Any], TurnRoute]:
    result = copy.deepcopy(message)
    params = result.setdefault("params", {})
    directive = _apply_directive_to_first_text(params)

    if fixed_profile is not None:
        fixed_effort = PROFILES[fixed_profile].effort
        if directive.profile is not None and (
            directive.profile != fixed_profile or directive.effort != fixed_effort
        ):
            raise ValueError("model directives cannot change a fixed-profile evaluation run")
        selected, effort, source, marker = fixed_profile, fixed_effort, "fixed-profile", None
    elif directive.profile is not None:
        selected = directive.profile
        effort = str(directive.effort)
        source = "user-directive"
        marker = directive.marker
    else:
        explicit = profile_for_model(params.get("model"))
        if params.get("model") and explicit is None:
            raise ValueError(f"unsupported model override: {params['model']}")
        selected = explicit or resumed_profile or get_default_profile()
        source = (
            "client-override"
            if explicit is not None
            else ("resumed-profile" if resumed_profile else "default")
        )
        marker = None
        profile = PROFILES[selected]
        requested_effort = params.get("effort")
        if requested_effort is not None:
            effort = str(requested_effort)
        elif resumed_effort and selected == resumed_profile and fixed_profile is None:
            effort = resumed_effort
        else:
            effort = profile.effort

    profile = PROFILES[selected]
    params["model"] = profile.model
    params["effort"] = effort
    collaboration_mode = params.get("collaborationMode")
    if isinstance(collaboration_mode, dict):
        settings = collaboration_mode.setdefault("settings", {})
        settings["model"] = profile.model
        settings["reasoning_effort"] = effort
    return result, TurnRoute(selected, effort, source, marker, directive.temporary)


def prepare_turn_steer(
    message: dict[str, Any],
) -> tuple[dict[str, Any], ProfileName | None, str | None, str | None, bool]:
    result = copy.deepcopy(message)
    params = result.setdefault("params", {})
    directive = _apply_directive_to_first_text(params)
    return result, directive.profile, directive.effort, directive.marker, directive.temporary


def is_router_tool_call(message: dict[str, Any], tool: str = ROUTER_TOOL) -> bool:
    return (
        message.get("method") == "item/tool/call"
        and message.get("params", {}).get("namespace") == ROUTER_NAMESPACE
        and message.get("params", {}).get("tool") == tool
    )


def is_route_state_tool_call(message: dict[str, Any]) -> bool:
    return is_router_tool_call(message, ROUTER_STATE_TOOL)


def parse_switch_arguments(
    params: dict[str, Any], current_profile: ProfileName, current_effort: str
) -> tuple[ProfileName, str, str | None]:
    arguments = params.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    try:
        raw_profile = arguments.get("targetProfile")
        raw_effort = arguments.get("targetReasoningEffort")
        if raw_profile is None and raw_effort is None:
            raise ValueError("targetProfile or targetReasoningEffort is required")
        target = (
            ProfileName(str(raw_profile).strip().lower())
            if raw_profile is not None
            else current_profile
        )
        effort = (
            str(raw_effort).strip().lower()
            if raw_effort is not None
            else PROFILES[target].effort
        )
        reason = str(arguments["reason"]).strip() or None if "reason" in arguments else None
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid switch arguments: {exc}") from exc
    if effort not in PROFILES[target].allowed_efforts:
        raise ValueError(f"{target.value} does not allow reasoning effort {effort}")
    return target, effort, reason
