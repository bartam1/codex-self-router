from __future__ import annotations

import copy
import math

import pytest
import yaml

from codex_self_router.config import (
    DEFAULT_CONFIG,
    DIRECTIVE_ROUTES,
    PROFILES,
    ROUTER_STATE_TOOL,
    ROUTING_TOOL_SPEC,
    ProfileName,
    SwitchApproval,
    apply_config,
    default_config_yaml,
    get_agent_switching_enabled,
    get_default_profile,
    get_switch_approval,
    load_config,
    routing_policy,
    switch_requires_approval,
)
from codex_self_router.transform import parse_directive


@pytest.fixture(autouse=True)
def restore_defaults():
    apply_config(copy.deepcopy(DEFAULT_CONFIG))
    yield
    apply_config(copy.deepcopy(DEFAULT_CONFIG))


def test_default_yaml_round_trips_and_exposes_full_matrix(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(default_config_yaml(), encoding="utf-8")
    assert load_config(path) == path
    assert get_default_profile() == ProfileName.SOL
    assert get_agent_switching_enabled()
    assert get_switch_approval() == SwitchApproval.ALWAYS
    assert "Separate decision-making from mechanical execution" in routing_policy()
    assert "{{profiles}}" not in routing_policy()
    assert len([marker for marker in DIRECTIVE_ROUTES if not marker.startswith("#")]) == 12
    assert DIRECTIVE_ROUTES["l1"] == (ProfileName.LUNA, "low")
    assert DIRECTIVE_ROUTES["t2"] == (ProfileName.TERRA, "medium")
    assert DIRECTIVE_ROUTES["s3"] == (ProfileName.SOL, "xhigh")
    assert DIRECTIVE_ROUTES["a3"] == (ProfileName.ASTRA, "xhigh")
    assert PROFILES[ProfileName.TERRA].model == "gpt-5.6-terra"
    parsed = parse_directive("a1 investigate")
    assert (parsed.profile, parsed.effort, parsed.text) == (
        ProfileName.ASTRA,
        "low",
        "investigate",
    )


def test_yaml_can_change_defaults_prefixes_models_and_prices(tmp_path):
    data = copy.deepcopy(DEFAULT_CONFIG)
    data["default_profile"] = "terra"
    data["profiles"]["terra"]["model"] = "custom-terra"
    data["profiles"]["terra"]["directive_prefix"] = "e"
    data["profiles"]["terra"]["price"]["output"] = 7.5
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    load_config(path)
    assert get_default_profile() == ProfileName.TERRA
    assert PROFILES[ProfileName.TERRA].model == "custom-terra"
    assert PROFILES[ProfileName.TERRA].price.output == 7.5
    assert DIRECTIVE_ROUTES["e3"] == (ProfileName.TERRA, "xhigh")
    assert "t3" not in DIRECTIVE_ROUTES


def test_invalid_yaml_routes_fail_closed():
    data = copy.deepcopy(DEFAULT_CONFIG)
    data["profiles"]["terra"]["directive_prefix"] = "l"
    with pytest.raises(ValueError, match="duplicate directive prefix"):
        apply_config(data)

    data = copy.deepcopy(DEFAULT_CONFIG)
    data["effort_levels"]["3"] = "ultra"
    with pytest.raises(ValueError, match="does not allow effort ultra"):
        apply_config(data)

    data = copy.deepcopy(DEFAULT_CONFIG)
    data["agent_switch_approval"] = "sometimes"
    with pytest.raises(ValueError, match="agent_switch_approval must be one of"):
        apply_config(data)

    data = copy.deepcopy(DEFAULT_CONFIG)
    data["routing_policy_template"] = "Use {{unknown}}."
    with pytest.raises(ValueError, match="unknown routing_policy_template placeholders"):
        apply_config(data)

    data = copy.deepcopy(DEFAULT_CONFIG)
    data["agent_switching_enabled"] = "no"
    with pytest.raises(ValueError, match="must be true or false"):
        apply_config(data)

    data = copy.deepcopy(DEFAULT_CONFIG)
    data["legacy_directives"]["l1"] = {"profile": "astra", "effort": "xhigh"}
    with pytest.raises(ValueError, match="duplicate directive marker l1"):
        apply_config(data)

    data = copy.deepcopy(DEFAULT_CONFIG)
    data["profiles"]["luna"]["price"]["input"] = math.nan
    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_config(data)


def test_switch_approval_policy_controls_prompting():
    data = copy.deepcopy(DEFAULT_CONFIG)
    data["agent_switch_approval"] = "never"
    apply_config(data)
    assert get_switch_approval() == SwitchApproval.NEVER
    assert "active agent-switch approval mode is never" in routing_policy()

    assert not switch_requires_approval(ProfileName.SOL, "medium", ProfileName.ASTRA, "xhigh")
    assert not switch_requires_approval(
        ProfileName.ASTRA,
        "xhigh",
        ProfileName.LUNA,
        "low",
        SwitchApproval.UPGRADES_ONLY,
    )
    assert switch_requires_approval(
        ProfileName.LUNA,
        "low",
        ProfileName.ASTRA,
        "low",
        SwitchApproval.UPGRADES_ONLY,
    )
    assert switch_requires_approval(
        ProfileName.SOL,
        "low",
        ProfileName.SOL,
        "xhigh",
        SwitchApproval.UPGRADES_ONLY,
    )


def test_upgrades_only_uses_semantic_effort_order_not_yaml_order():
    data = copy.deepcopy(DEFAULT_CONFIG)
    data["effort_levels"] = {"3": "xhigh", "1": "low", "2": "medium"}
    apply_config(data)

    assert switch_requires_approval(
        ProfileName.SOL,
        "low",
        ProfileName.SOL,
        "xhigh",
        SwitchApproval.UPGRADES_ONLY,
    )
    assert not switch_requires_approval(
        ProfileName.SOL,
        "xhigh",
        ProfileName.SOL,
        "low",
        SwitchApproval.UPGRADES_ONLY,
    )


def test_agent_switching_can_be_disabled_without_disabling_user_directives():
    data = copy.deepcopy(DEFAULT_CONFIG)
    data["agent_switching_enabled"] = False
    apply_config(data)

    assert not get_agent_switching_enabled()
    assert [tool["name"] for tool in ROUTING_TOOL_SPEC["tools"]] == [ROUTER_STATE_TOOL]
    assert "Agent-requested model and reasoning-effort switches are disabled" in routing_policy()
    assert parse_directive("a2~ design it").profile == ProfileName.ASTRA


def test_old_policy_template_gets_authoritative_agent_switch_status():
    data = copy.deepcopy(DEFAULT_CONFIG)
    data["agent_switching_enabled"] = False
    data["routing_policy_template"] = "Custom policy. {{profiles}}"
    apply_config(data)

    assert routing_policy().startswith(
        "- Agent-requested model and reasoning-effort switches are disabled"
    )


def test_custom_policy_template_is_injected_and_old_config_falls_back():
    data = copy.deepcopy(DEFAULT_CONFIG)
    data["routing_policy_template"] = "Custom policy.\n{{approval_mode}}\n{{profiles}}\n"
    apply_config(data)
    assert routing_policy().startswith("Custom policy.")
    assert "\nalways\n" in routing_policy()
    assert "Sol (gpt-5.6-sol)" in routing_policy()

    del data["routing_policy_template"]
    apply_config(data)
    assert routing_policy().startswith("Model and reasoning routing is managed")
