from __future__ import annotations

import copy

import pytest

from codex_self_router.config import DEFAULT_CONFIG, apply_config


@pytest.fixture(autouse=True)
def isolate_router_config():
    apply_config(copy.deepcopy(DEFAULT_CONFIG))
    yield
    apply_config(copy.deepcopy(DEFAULT_CONFIG))
