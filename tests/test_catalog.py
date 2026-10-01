from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from codex_self_router.catalog import PROBE_THREAD_ID, read_model_catalog
from codex_self_router.cli import async_main, build_parser
from codex_self_router.config import PROFILES, default_config_yaml


def model_entries() -> list[dict]:
    return [
        {
            "model": profile.model,
            "supportedReasoningEfforts": [
                {"reasoningEffort": effort} for effort in profile.allowed_efforts
            ],
        }
        for profile in PROFILES.values()
    ]


@pytest.mark.asyncio
async def test_catalog_reads_every_page():
    entries = model_entries()
    calls = []

    async def request(method, params):
        calls.append((method, params))
        if "cursor" not in params:
            return {"result": {"data": entries[:2], "nextCursor": "next-page"}}
        return {"result": {"data": entries[2:], "nextCursor": None}}

    assert await read_model_catalog(request) == {"result": {"data": entries, "nextCursor": None}}
    assert calls == [
        ("model/list", {"limit": 100, "includeHidden": True}),
        ("model/list", {"limit": 100, "includeHidden": True, "cursor": "next-page"}),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"error": {"message": "catalog unavailable"}}, "catalog unavailable"),
        ({"result": {}}, "invalid catalog"),
        ({"result": {"data": [None]}}, "invalid model entry"),
        ({"result": {"data": [], "nextCursor": 1}}, "invalid or repeated cursor"),
        ({"result": {"data": [], "nextCursor": ""}}, "invalid or repeated cursor"),
    ],
)
async def test_catalog_rejects_invalid_pages(response, message):
    async def request(method, params):
        return response

    with pytest.raises(RuntimeError, match=message):
        await read_model_catalog(request)


@pytest.mark.asyncio
async def test_catalog_rejects_repeated_cursor():
    calls = 0

    async def request(method, params):
        nonlocal calls
        calls += 1
        return {"result": {"data": [], "nextCursor": "same-page"}}

    with pytest.raises(RuntimeError, match="repeated cursor"):
        await read_model_catalog(request)
    assert calls == 2


@pytest.fixture
def fake_codex(tmp_path):
    def create(entries=None, settings_error=f"thread not found: {PROBE_THREAD_ID}"):
        catalog = model_entries() if entries is None else entries
        binary = tmp_path / "fake-codex"
        log = tmp_path / "requests.jsonl"
        # Large pages exercise the same stream limit as the live app-server probe.
        script = f"""#!{sys.executable}
import json
import sys
from pathlib import Path

entries = {catalog!r}
settings_error = {settings_error!r}
log = Path({str(log)!r})
if "--version" in sys.argv:
    print("codex-cli 0.159.3")
    sys.exit(0)
assert sys.argv[1:] == ["--enable", "step_model_switching", "app-server", "--listen", "stdio://"]
for line in sys.stdin:
    message = json.loads(line)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(message) + "\\n")
    method = message["method"]
    if method == "initialized":
        continue
    if method == "initialize":
        response = {{"result": {{"userAgent": "fake"}}}}
        print(json.dumps({{"method": "test/notification", "params": {{}}}}), flush=True)
    elif method == "turn/settings/update":
        response = {{"error": {{"code": -32600, "message": settings_error}}}}
    elif method == "model/list":
        first = "cursor" not in message["params"]
        page = entries[:2] if first else entries[2:]
        for entry in page:
            entry["description"] = "x" * 100_000
        response = {{"result": {{"data": page, "nextCursor": "next-page" if first else None}}}}
    else:
        raise AssertionError("doctor must not create a thread or turn: " + method)
    print(json.dumps({{"id": message["id"], **response}}), flush=True)
"""
        binary.write_text(script, encoding="utf-8")
        binary.chmod(0o755)
        return binary, log

    return create


async def run_doctor(binary: Path, tmp_path: Path) -> int:
    config = tmp_path / "config.yaml"
    config.write_text(default_config_yaml(), encoding="utf-8")
    args = build_parser().parse_args(
        ["--config", str(config), "--codex-bin", str(binary), "doctor"]
    )
    return await async_main(args)


@pytest.mark.asyncio
async def test_doctor_checks_real_protocol_without_inference(fake_codex, tmp_path, capsys):
    binary, log = fake_codex()
    assert await run_doctor(binary, tmp_path) == 0
    output = capsys.readouterr().out
    assert "settings-update API recognized" in output
    assert "all configured models and efforts advertised" in output
    assert "not checked (doctor runs no inference)" in output
    requests = [json.loads(line) for line in log.read_text().splitlines()]
    assert [request["method"] for request in requests] == [
        "initialize",
        "initialized",
        "turn/settings/update",
        "model/list",
        "model/list",
    ]
    assert requests[0]["params"]["capabilities"]["experimentalApi"] is True
    assert requests[2]["params"]["threadId"] == PROBE_THREAD_ID
    assert requests[-1]["params"]["cursor"] == "next-page"


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["model", "effort"])
async def test_doctor_fails_on_missing_configured_capability(fake_codex, tmp_path, capsys, missing):
    entries = model_entries()
    if missing == "model":
        model = entries.pop()["model"]
        expected = f"model unavailable: {model}"
    else:
        effort = entries[-1]["supportedReasoningEfforts"].pop()["reasoningEffort"]
        expected = f"{entries[-1]['model']} does not advertise effort {effort}"
    binary, _ = fake_codex(entries)
    assert await run_doctor(binary, tmp_path) == 1
    assert expected in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", ["Method not found", "turn settings updates require the step_model_switching feature"]
)
async def test_doctor_fails_when_settings_api_is_unavailable(fake_codex, tmp_path, capsys, error):
    binary, log = fake_codex(settings_error=error)
    assert await run_doctor(binary, tmp_path) == 1
    expected = (
        f"App-server capability check failed: app-server settings-update probe failed: {error}"
    )
    assert expected in capsys.readouterr().out
    requests = [json.loads(line) for line in log.read_text().splitlines()]
    assert requests[-1]["method"] == "turn/settings/update"
