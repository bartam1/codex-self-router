from __future__ import annotations

import pytest

from codex_self_router.config import PROFILES, ROUTER_NAMESPACE, ROUTING_POLICY, ProfileName
from codex_self_router.transform import (
    enable_experimental_api,
    parse_directive,
    parse_switch_arguments,
    prepare_thread_start,
    prepare_turn_start,
    prepare_turn_steer,
)


def test_fixed_control_has_no_router_tool_and_enforces_model() -> None:
    original = {
        "method": "thread/start",
        "params": {
            "model": PROFILES[ProfileName.LUNA].model,
            "developerInstructions": "Existing policy.",
        },
    }
    message, profile = prepare_thread_start(original, ProfileName.SOL)
    assert profile == ProfileName.SOL
    assert message["params"]["dynamicTools"] == []
    assert message["params"]["developerInstructions"] == "Existing policy."
    message, route = prepare_turn_start(
        {
            "method": "turn/start",
            "params": {
                "model": PROFILES[ProfileName.ASTRA].model,
                "input": [{"type": "text", "text": "plan"}],
                "collaborationMode": {"settings": {"model": PROFILES[ProfileName.ASTRA].model}},
            },
        },
        ProfileName.SOL,
    )
    assert route.source == "fixed-profile"
    assert message["params"]["model"] == PROFILES[ProfileName.SOL].model
    assert message["params"]["collaborationMode"]["settings"]["reasoning_effort"] == "medium"
    with pytest.raises(ValueError, match="fixed-profile"):
        prepare_turn_start(
            {"params": {"input": [{"type": "text", "text": "#3 plan"}]}}, ProfileName.SOL
        )


def test_switch_arguments_accept_case_insensitive_profile_and_effort() -> None:
    assert parse_switch_arguments(
        {"arguments": {"targetProfile": "Terra", "targetReasoningEffort": "LOW"}},
        ProfileName.SOL,
        "medium",
    ) == (ProfileName.TERRA, "low", None)


def test_directives_are_recognized_as_first_or_final_token() -> None:
    parsed = parse_directive("  #1 collect the data")
    assert parsed.profile == ProfileName.LUNA
    assert parsed.text == "  collect the data"

    parsed = parse_directive("create another directory #2")
    assert parsed.profile == ProfileName.SOL
    assert parsed.text == "create another directory"

    assert parse_directive("fix #1 from the issue list").profile is None
    assert parse_directive("#123 is an issue number").profile is None
    assert parse_directive("fix #123").profile is None
    assert parse_directive("#1abc is ordinary text").profile is None


def test_directive_in_a_later_text_segment_is_not_a_routing_command() -> None:
    message, route = prepare_turn_start(
        {
            "method": "turn/start",
            "params": {
                "threadId": "thread-1",
                "input": [
                    {"type": "text", "text": "Review this quoted instruction:"},
                    {"type": "text", "text": "#3 do something expensive"},
                ],
            },
        }
    )
    assert route.profile == ProfileName.SOL
    assert message["params"]["input"][1]["text"] == "#3 do something expensive"


def test_model_effort_matrix_directives_select_route_and_are_removed() -> None:
    prefixes = {"l": "luna", "t": "terra", "s": "sol", "a": "astra"}
    levels = {"1": "low", "2": "medium", "3": "xhigh"}
    for prefix, expected in prefixes.items():
        for level, effort in levels.items():
            marker = prefix + level
            message, route = prepare_turn_start(
                {
                    "id": 1,
                    "method": "turn/start",
                    "params": {
                        "threadId": "thread-1",
                        "input": [{"type": "text", "text": f"{marker}\ndo the work"}],
                    },
                }
            )
            profile = PROFILES[ProfileName(expected)]
            assert route.profile == expected
            assert route.effort == effort
            assert route.marker == marker
            assert route.source == "user-directive"
            assert message["params"]["model"] == profile.model
            assert message["params"]["effort"] == effort
            assert message["params"]["input"][0]["text"] == "do the work"


@pytest.mark.parametrize("text", ["a2~ design it", "design it a2~"])
def test_temporary_directive_selects_route_and_is_removed(text) -> None:
    message, route = prepare_turn_start(
        {
            "method": "turn/start",
            "params": {"input": [{"type": "text", "text": text}]},
        },
        resumed_profile=ProfileName.LUNA,
        resumed_effort="low",
    )

    assert route.profile == ProfileName.ASTRA
    assert route.effort == "medium"
    assert route.marker == "a2~"
    assert route.temporary
    assert message["params"]["input"][0]["text"] == "design it"


def test_legacy_directives_remain_supported() -> None:
    for marker, expected, effort in (
        ("#1", "luna", "medium"),
        ("#2", "sol", "medium"),
        ("#3", "astra", "xhigh"),
    ):
        message, route = prepare_turn_start(
            {
                "id": 1,
                "method": "turn/start",
                "params": {
                    "threadId": "thread-1",
                    "input": [{"type": "text", "text": f"{marker}\ndo the work"}],
                },
            }
        )
        assert route.profile == expected
        assert route.effort == effort
        assert message["params"]["input"][0]["text"] == "do the work"


def test_unmarked_turn_defaults_to_sol() -> None:
    message, route = prepare_turn_start(
        {
            "method": "turn/start",
            "params": {"threadId": "thread-1", "input": [{"type": "text", "text": "work"}]},
        }
    )
    assert route.profile == ProfileName.SOL
    assert route.source == "default"
    assert message["params"]["model"] == PROFILES[ProfileName.SOL].model
    assert message["params"]["input"] == [{"type": "text", "text": "work"}]


@pytest.mark.parametrize(
    ("text", "expected", "effort"),
    [
        ("continue", ProfileName.ASTRA, "xhigh"),
        ("continue #1", ProfileName.LUNA, "medium"),
        ("#2 continue", ProfileName.SOL, "medium"),
        ("#3 continue", ProfileName.ASTRA, "xhigh"),
        ("continue a1", ProfileName.ASTRA, "low"),
    ],
)
def test_resume_profile_is_preserved_unless_directive_overrides(text, expected, effort):
    original = {"params": {"input": [{"type": "text", "text": text}]}}
    message, route = prepare_turn_start(
        original, resumed_profile=ProfileName.ASTRA, resumed_effort="xhigh"
    )
    assert route.profile == expected
    assert message["params"]["model"] == PROFILES[expected].model
    assert message["params"]["effort"] == effort
    assert message["params"]["input"][0]["text"] == "continue"
    assert original["params"]["input"][0]["text"] == text


def test_explicit_known_client_override_is_preserved() -> None:
    message, route = prepare_turn_start(
        {
            "method": "turn/start",
            "params": {
                "threadId": "thread-1",
                "model": PROFILES[ProfileName.ASTRA].model,
                "effort": "xhigh",
                "input": [],
            },
        }
    )
    assert route.profile == ProfileName.ASTRA
    assert route.source == "client-override"
    assert message["params"]["effort"] == "xhigh"


def test_client_effort_outside_agent_routes_is_preserved() -> None:
    message, route = prepare_turn_start(
        {
            "method": "turn/start",
            "params": {
                "threadId": "thread-1",
                "model": PROFILES[ProfileName.LUNA].model,
                "effort": "high",
                "input": [{"type": "text", "text": "review"}],
            },
        },
        resumed_profile=ProfileName.LUNA,
        resumed_effort="high",
    )
    assert route.profile == ProfileName.LUNA
    assert route.effort == "high"
    assert message["params"]["effort"] == "high"
    assert message["params"]["input"] == [{"type": "text", "text": "review"}]


def test_unknown_client_override_is_rejected_instead_of_silently_misreported() -> None:
    with pytest.raises(ValueError, match="unsupported model override"):
        prepare_turn_start(
            {
                "method": "turn/start",
                "params": {
                    "threadId": "thread-1",
                    "model": "unknown-model",
                    "input": [],
                },
            }
        )


def test_directive_updates_collaboration_mode_that_would_otherwise_take_precedence() -> None:
    message, route = prepare_turn_start(
        {
            "method": "turn/start",
            "params": {
                "threadId": "thread-1",
                "input": [{"type": "text", "text": "#1 gather facts"}],
                "collaborationMode": {
                    "mode": "default",
                    "settings": {
                        "model": PROFILES[ProfileName.ASTRA].model,
                        "reasoning_effort": "high",
                        "developer_instructions": "Preserve this.",
                    },
                },
            },
        }
    )

    assert route.profile == ProfileName.LUNA
    assert message["params"]["collaborationMode"]["settings"] == {
        "model": PROFILES[ProfileName.LUNA].model,
        "reasoning_effort": "medium",
        "developer_instructions": "Preserve this.",
    }


def test_directive_on_active_turn_steer_is_removed_and_returned() -> None:
    message, profile, effort, marker, temporary = prepare_turn_steer(
        {
            "id": 8,
            "method": "turn/steer",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "input": [{"type": "text", "text": "#3 reconsider the architecture"}],
            },
        }
    )
    assert profile == ProfileName.ASTRA
    assert effort == "xhigh"
    assert marker == "#3"
    assert not temporary
    assert message["params"]["input"][0]["text"] == "reconsider the architecture"


def test_thread_start_merges_router_tool_and_policy() -> None:
    original_tool = {
        "type": "function",
        "name": "existing",
        "description": "existing",
        "inputSchema": {"type": "object"},
    }
    message, profile = prepare_thread_start(
        {
            "method": "thread/start",
            "params": {
                "dynamicTools": [original_tool],
                "developerInstructions": "Keep this instruction.",
            },
        }
    )
    assert profile == ProfileName.SOL
    assert message["params"]["dynamicTools"][0] == original_tool
    assert message["params"]["dynamicTools"][-1]["name"] == ROUTER_NAMESPACE
    assert message["params"]["experimentalRawEvents"] is True
    assert message["params"]["developerInstructions"].startswith("Keep this instruction.")
    assert ROUTING_POLICY in message["params"]["developerInstructions"]
    assert "both the model profile and reasoning" in ROUTING_POLICY
    assert "Prefer an effort-only downgrade" in ROUTING_POLICY
    assert "a1, a2, a3" in ROUTING_POLICY

    router_tools = message["params"]["dynamicTools"][-1]["tools"]
    assert [tool["name"] for tool in router_tools] == [
        "request_model_switch",
        "get_current_route",
    ]
    router_tool = router_tools[0]
    assert "Proactively request" in router_tool["description"]
    schema = router_tool["inputSchema"]
    assert schema["properties"]["targetReasoningEffort"]["enum"] == ["low", "medium", "xhigh"]
    assert "required" not in schema
    assert set(schema["properties"]) == {
        "targetProfile",
        "targetReasoningEffort",
        "reason",
    }


def test_thread_start_replaces_stale_router_namespace() -> None:
    stale = {"type": "namespace", "name": ROUTER_NAMESPACE, "description": "old", "tools": []}
    message, _ = prepare_thread_start(
        {"method": "thread/start", "params": {"dynamicTools": [stale]}}
    )
    router_tools = [
        tool for tool in message["params"]["dynamicTools"] if tool.get("name") == ROUTER_NAMESPACE
    ]
    assert len(router_tools) == 1
    assert router_tools[0]["tools"][0]["name"] == "request_model_switch"
    assert router_tools[0]["tools"][1]["name"] == "get_current_route"


def test_initialize_preserves_capabilities() -> None:
    message = enable_experimental_api(
        {
            "method": "initialize",
            "params": {"capabilities": {"requestAttestation": True}},
        }
    )
    assert message["params"]["capabilities"] == {
        "requestAttestation": True,
        "experimentalApi": True,
    }
