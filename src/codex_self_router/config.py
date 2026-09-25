from __future__ import annotations

import copy
import math
import os
import re
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class ProfileName(StrEnum):
    LUNA = "luna"
    TERRA = "terra"
    SOL = "sol"
    ASTRA = "astra"


class SwitchApproval(StrEnum):
    ALWAYS = "always"
    UPGRADES_ONLY = "upgrades_only"
    NEVER = "never"


EFFORT_RANK = {"low": 0, "medium": 1, "xhigh": 2}


@dataclass(frozen=True, slots=True)
class Price:
    """API-equivalent USD price per million tokens."""

    input: float
    cached_input: float
    cache_write_input: float
    output: float


@dataclass(frozen=True, slots=True)
class Profile:
    name: ProfileName
    model: str
    effort: str
    price: Price
    directive_prefix: str
    allowed_efforts: tuple[str, ...]
    description: str


DEFAULT_ROUTING_POLICY_TEMPLATE = """\
Model and reasoning routing is managed by the self-router.

{{agent_switching}}
- At the start of each substantial phase, proactively choose both the model profile and reasoning
  effort that fit the upcoming work. You are responsible for requesting a change; do not wait for
  the user to suggest it.
- Use self_router.get_current_route only when the active route is uncertain or after a model/effort
  switch; do not repeat it at the start of every turn when the route is already known.
- The active agent-switch approval mode is {{approval_mode}}. `always` prompts for every effective
  change, `upgrades_only` automatically permits changes that do not increase model price or
  reasoning effort, and `never` automatically permits every valid change. A possible approval
  prompt is never a reason to avoid calling the tool.
{{profiles}}
- low: fast execution for straightforward, well-bounded work.
- medium: balanced reasoning for normal implementation and analysis.
- xhigh: deep reasoning for genuinely difficult or consequential work.
- Judge switching overhead by expected inference steps, not user-visible actions. A tool call and
  handling its result normally require multiple inference steps.
- Prefer an effort-only downgrade when the active model remains appropriate. Change models when
  capability, latency, or cost makes another profile a better fit.
- Separate decision-making from mechanical execution. Use a capable route to reach or review a
  consequential decision, then downgrade for deterministic execution when the remaining phase is
  straightforward.
- Luna/low is normally appropriate for read-only file discovery, status and log collection, test
  execution, formatting, and deterministic shell or task-runner commands after the decision is
  already made. Do not downgrade when interpreting the results still requires difficult judgment.
- Terra/medium is normally appropriate for drafting routine pull-request descriptions and status
  comments. Use Sol or Astra when a review comment makes substantive correctness, concurrency,
  security, architecture, financial, or other consequential claims.
- Model-switch authorization never authorizes shell commands, task commands, pull-request creation,
  comments, pushes, or other external side effects. Those actions remain subject to Codex's normal
  permission and approval rules.
- A switched continuation is still the same task. Preserve the full conversation context and do
  not queue optional clarification questions merely because Codex created a turn boundary.
- If the current model or effort is uncertain, request what the phase needs; exact same-target
  requests are no-ops.
- A leading or trailing user directive ({{directives}}) is explicit authorization handled before
  inference. Its selected route is authoritative for that task; do not request another route until
  a later user task. A `~` suffix restores the preceding route afterward.
"""


DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "default_profile": "sol",
    "agent_switching_enabled": True,
    "agent_switch_approval": "always",
    "routing_policy_template": DEFAULT_ROUTING_POLICY_TEMPLATE,
    "effort_levels": {"1": "low", "2": "medium", "3": "xhigh"},
    "legacy_directives": {
        "#1": {"profile": "luna", "effort": "medium"},
        "#2": {"profile": "sol", "effort": "medium"},
        "#3": {"profile": "astra", "effort": "xhigh"},
    },
    "profiles": {
        "luna": {
            "model": "gpt-5.6-luna",
            "default_effort": "medium",
            "directive_prefix": "l",
            "allowed_efforts": ["low", "medium", "xhigh"],
            "description": "Fast high-volume, lookup, collection, and mechanical work.",
            "price": {
                "input": 0.20,
                "cached_input": 0.02,
                "cache_write_input": 0.25,
                "output": 1.20,
            },
        },
        "terra": {
            "model": "gpt-5.6-terra",
            "default_effort": "medium",
            "directive_prefix": "t",
            "allowed_efforts": ["low", "medium", "xhigh"],
            "description": "Everyday production work requiring balanced judgment.",
            "price": {
                "input": 2.00,
                "cached_input": 0.20,
                "cache_write_input": 2.50,
                "output": 12.00,
            },
        },
        "sol": {
            "model": "gpt-5.6-sol",
            "default_effort": "medium",
            "directive_prefix": "s",
            "allowed_efforts": ["low", "medium", "xhigh"],
            "description": "Advanced coding, ambiguous problems, and difficult debugging.",
            "price": {
                "input": 4.00,
                "cached_input": 0.40,
                "cache_write_input": 5.00,
                "output": 20.00,
            },
        },
        "astra": {
            "model": "gpt-6-astra",
            "default_effort": "xhigh",
            "directive_prefix": "a",
            "allowed_efforts": ["low", "medium", "xhigh"],
            "description": "Consequential architecture and the most demanding reasoning.",
            "price": {
                "input": 10.00,
                "cached_input": 1.00,
                "cache_write_input": 12.50,
                "output": 50.00,
            },
        },
    },
}

PROFILES: dict[ProfileName, Profile] = {}
DIRECTIVE_ROUTES: dict[str, tuple[ProfileName, str]] = {}
DEFAULT_PROFILE = ProfileName.SOL
AGENT_SWITCH_APPROVAL = SwitchApproval.ALWAYS
AGENT_SWITCHING_ENABLED = True
EFFORT_LEVELS: dict[str, str] = {}
ROUTER_NAMESPACE = "self_router"
ROUTER_TOOL = "request_model_switch"
ROUTER_STATE_TOOL = "get_current_route"
ROUTING_POLICY = ""
ROUTING_POLICY_TEMPLATE = DEFAULT_ROUTING_POLICY_TEMPLATE
ROUTING_TOOL_SPEC: dict[str, Any] = {}
ACTIVE_CONFIG_PATH: Path | None = None


def default_config_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "codex-self-router" / "config.yaml"


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _price(value: Any, name: str) -> Price:
    data = _mapping(value, f"profiles.{name}.price")
    try:
        numbers = {field.name: float(data[field.name]) for field in fields(Price)}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid price for profile {name}: {exc}") from exc
    if any(not math.isfinite(number) or number < 0 for number in numbers.values()):
        raise ValueError(f"prices for profile {name} must be finite and non-negative")
    return Price(**numbers)


def _render_policy() -> str:
    profile_lines = "\n".join(
        f"- {name.value.title()} ({profile.model}): {profile.description}"
        for name, profile in PROFILES.items()
    )
    directives = ", ".join([*DIRECTIVE_ROUTES, *(f"{marker}~" for marker in DIRECTIVE_ROUTES)])
    replacements = {
        "{{agent_switching}}": (
            "- Agent-requested model and reasoning-effort switches are enabled."
            if AGENT_SWITCHING_ENABLED
            else (
                "- Agent-requested model and reasoning-effort switches are disabled. Do not "
                "attempt to change the route; only explicit user directives or manual client "
                "changes can do so."
            )
        ),
        "{{approval_mode}}": AGENT_SWITCH_APPROVAL.value,
        "{{profiles}}": profile_lines,
        "{{directives}}": directives,
    }
    policy = ROUTING_POLICY_TEMPLATE
    if not AGENT_SWITCHING_ENABLED and "{{agent_switching}}" not in policy:
        policy = "{{agent_switching}}\n" + policy
    for marker, value in replacements.items():
        policy = policy.replace(marker, value)
    return policy


def _rebuild_contract() -> None:
    global ROUTING_POLICY
    ROUTING_POLICY = _render_policy()
    efforts = sorted(
        {effort for profile in PROFILES.values() for effort in profile.allowed_efforts}
    )
    switch_tool = {
        "type": "function",
        "name": ROUTER_TOOL,
        "description": (
            "Proactively request the model and reasoning effort best suited to the next "
            "substantial phase. Either targetProfile or targetReasoningEffort may be "
            "omitted to keep its current value. Profiles are luna, terra, sol, or astra; "
            "reasoning efforts are low, medium, or xhigh. An optional reason may provide context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "targetProfile": {
                    "type": "string",
                    "enum": [name.value for name in PROFILES],
                    "description": (
                        "Destination model profile: luna, terra, sol, or astra; omit to "
                        "keep the current model."
                    ),
                },
                "targetReasoningEffort": {
                    "type": "string",
                    "enum": efforts,
                    "description": "Destination effort; omit to use the profile default.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional context for the routing decision; not required.",
                },
            },
            "anyOf": [
                {"required": ["targetProfile"]},
                {"required": ["targetReasoningEffort"]},
            ],
            "additionalProperties": False,
        },
    }
    state_tool = {
        "type": "function",
        "name": ROUTER_STATE_TOOL,
        "description": (
            "Return the exact active router profile, model, reasoning effort, and "
            "agent-switch configuration. This read-only call never changes state."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }
    spec = {
        "type": "namespace",
        "name": ROUTER_NAMESPACE,
        "description": "Inspect the active route and, when enabled, request route changes.",
        "tools": [switch_tool, state_tool] if AGENT_SWITCHING_ENABLED else [state_tool],
    }
    ROUTING_TOOL_SPEC.clear()
    ROUTING_TOOL_SPEC.update(spec)


def apply_config(data: dict[str, Any], path: Path | None = None) -> None:
    global DEFAULT_PROFILE, AGENT_SWITCH_APPROVAL, EFFORT_LEVELS, ACTIVE_CONFIG_PATH
    global AGENT_SWITCHING_ENABLED, ROUTING_POLICY_TEMPLATE
    if data.get("version") != 1:
        raise ValueError("config version must be 1")
    raw_profiles = _mapping(data.get("profiles"), "profiles")
    profiles: dict[ProfileName, Profile] = {}
    prefixes: set[str] = set()
    for name in ProfileName:
        raw = _mapping(raw_profiles.get(name.value), f"profiles.{name.value}")
        model = str(raw.get("model", "")).strip()
        prefix = str(raw.get("directive_prefix", "")).strip()
        default_effort = str(raw.get("default_effort", "")).strip()
        allowed = tuple(str(item).strip() for item in raw.get("allowed_efforts", []))
        if not model or not prefix or not allowed or default_effort not in allowed:
            raise ValueError(f"invalid model, prefix, or efforts for profile {name.value}")
        if (
            any(not effort for effort in allowed)
            or prefix in prefixes
            or any(c.isspace() for c in prefix)
        ):
            raise ValueError(f"invalid or duplicate directive prefix for profile {name.value}")
        prefixes.add(prefix)
        profiles[name] = Profile(
            name,
            model,
            default_effort,
            _price(raw.get("price"), name.value),
            prefix,
            allowed,
            str(raw.get("description", "")).strip(),
        )
    levels = {
        str(key): str(value)
        for key, value in _mapping(data.get("effort_levels"), "effort_levels").items()
    }
    if not levels or any(not key or not value for key, value in levels.items()):
        raise ValueError("effort_levels must contain non-empty marker/effort pairs")
    routes: dict[str, tuple[ProfileName, str]] = {}
    for name, profile in profiles.items():
        for level, effort in levels.items():
            if effort not in profile.allowed_efforts:
                raise ValueError(f"{name.value} does not allow effort {effort}")
            marker = f"{profile.directive_prefix}{level}"
            if marker in routes:
                raise ValueError(f"duplicate directive marker {marker}")
            routes[marker] = (name, effort)
    for marker, raw in _mapping(data.get("legacy_directives", {}), "legacy_directives").items():
        route = _mapping(raw, f"legacy_directives.{marker}")
        try:
            name = ProfileName(route["profile"])
            effort = str(route["effort"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid legacy directive {marker}: {exc}") from exc
        if effort not in profiles[name].allowed_efforts:
            raise ValueError(f"legacy directive {marker} uses unsupported effort {effort}")
        if not marker or any(c.isspace() for c in marker):
            raise ValueError("directive markers cannot contain whitespace")
        marker = str(marker)
        if marker in routes:
            raise ValueError(f"duplicate directive marker {marker}")
        routes[marker] = (name, effort)
    try:
        default = ProfileName(data.get("default_profile"))
    except ValueError as exc:
        raise ValueError("default_profile must name a configured profile") from exc
    try:
        approval = SwitchApproval(data.get("agent_switch_approval", "always"))
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in SwitchApproval)
        raise ValueError(f"agent_switch_approval must be one of: {choices}") from exc
    agent_switching = data.get("agent_switching_enabled", True)
    if not isinstance(agent_switching, bool):
        raise ValueError("agent_switching_enabled must be true or false")
    template = data.get("routing_policy_template", DEFAULT_ROUTING_POLICY_TEMPLATE)
    if not isinstance(template, str) or not template.strip():
        raise ValueError("routing_policy_template must be a non-empty string")
    unknown_markers = set(re.findall(r"{{[^{}]+}}", template)) - {
        "{{approval_mode}}",
        "{{agent_switching}}",
        "{{profiles}}",
        "{{directives}}",
    }
    if unknown_markers:
        raise ValueError(
            "unknown routing_policy_template placeholders: " + ", ".join(sorted(unknown_markers))
        )
    PROFILES.clear()
    PROFILES.update(profiles)
    DIRECTIVE_ROUTES.clear()
    DIRECTIVE_ROUTES.update(routes)
    DEFAULT_PROFILE = default
    AGENT_SWITCH_APPROVAL = approval
    AGENT_SWITCHING_ENABLED = agent_switching
    ROUTING_POLICY_TEMPLATE = template
    EFFORT_LEVELS = levels
    ACTIVE_CONFIG_PATH = path
    _rebuild_contract()


def load_config(path: Path | None = None) -> Path | None:
    selected = path.expanduser().resolve() if path else default_config_path()
    if not selected.exists():
        if path is not None:
            raise ValueError(f"config file does not exist: {selected}")
        apply_config(copy.deepcopy(DEFAULT_CONFIG))
        return None
    try:
        data = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {selected}: {exc}") from exc
    apply_config(_mapping(data, "config"), selected)
    return selected


def profile_for_model(model: str | None) -> ProfileName | None:
    if model is None:
        return None
    return next((name for name, profile in PROFILES.items() if profile.model == model), None)


def get_default_profile() -> ProfileName:
    return DEFAULT_PROFILE


def get_switch_approval() -> SwitchApproval:
    return AGENT_SWITCH_APPROVAL


def get_agent_switching_enabled() -> bool:
    return AGENT_SWITCHING_ENABLED


def set_agent_switching_enabled(value: bool) -> None:
    global AGENT_SWITCHING_ENABLED
    AGENT_SWITCHING_ENABLED = value
    _rebuild_contract()


def set_switch_approval(value: SwitchApproval | str) -> None:
    global AGENT_SWITCH_APPROVAL
    AGENT_SWITCH_APPROVAL = SwitchApproval(value)
    _rebuild_contract()


def switch_requires_approval(
    previous: ProfileName,
    previous_effort: str,
    target: ProfileName,
    target_effort: str,
    policy: SwitchApproval | None = None,
) -> bool:
    selected = policy or AGENT_SWITCH_APPROVAL
    if selected == SwitchApproval.ALWAYS:
        return True
    if selected == SwitchApproval.NEVER:
        return False
    previous_price = PROFILES[previous].price
    target_price = PROFILES[target].price
    model_upgrade = any(
        getattr(target_price, field) > getattr(previous_price, field)
        for field in asdict(previous_price)
    )
    try:
        effort_upgrade = EFFORT_RANK[target_effort] > EFFORT_RANK[previous_effort]
    except KeyError:
        effort_upgrade = target_effort != previous_effort
    return model_upgrade or effort_upgrade


def config_snapshot() -> dict[str, Any]:
    return {
        "path": str(ACTIVE_CONFIG_PATH) if ACTIVE_CONFIG_PATH else None,
        "defaultProfile": DEFAULT_PROFILE.value,
        "agentSwitchingEnabled": AGENT_SWITCHING_ENABLED,
        "agentSwitchApproval": AGENT_SWITCH_APPROVAL.value,
        "routingPolicyTemplate": ROUTING_POLICY_TEMPLATE,
        "effortLevels": dict(EFFORT_LEVELS),
        "directives": {
            marker: {"profile": profile.value, "effort": effort}
            for marker, (profile, effort) in DIRECTIVE_ROUTES.items()
        },
        "profiles": {
            name.value: {
                "model": profile.model,
                "defaultEffort": profile.effort,
                "allowedEfforts": list(profile.allowed_efforts),
                "directivePrefix": profile.directive_prefix,
            }
            for name, profile in PROFILES.items()
        },
    }


def routing_policy() -> str:
    return ROUTING_POLICY


def merge_routing_policy(instructions: str | None) -> str:
    base = instructions or ""
    markers = (
        "Model and reasoning routing is managed by the self-router.",
        "Model and reasoning routing is available through self_router.request_model_switch.",
        "Model routing is available through self_router.request_model_switch.",
    )
    positions = [base.find(marker) for marker in markers if marker in base]
    if positions:
        base = base[: min(positions)].rstrip()
    return (base + "\n\n" if base else "") + routing_policy()


def default_config_yaml() -> str:
    class ConfigDumper(yaml.SafeDumper):
        pass

    def represent_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
        style = "|" if "\n" in value else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    ConfigDumper.add_representer(str, represent_string)
    return yaml.dump(DEFAULT_CONFIG, Dumper=ConfigDumper, sort_keys=False)


apply_config(copy.deepcopy(DEFAULT_CONFIG))
