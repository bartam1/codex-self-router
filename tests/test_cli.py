from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from codex_self_router.cli import build_parser, command_run
from codex_self_router.report import ReportStore


@pytest.mark.parametrize(
    "arguments",
    [
        ["resume"],
        ["resume", "--last"],
        ["resume", "--all"],
        ["fork"],
        ["--cd", "/other repo", "resume"],
        ["resume", "--cd=/other repo"],
        ["-C/other-repo", "resume"],
        ["resume", "-C", "/other-repo"],
    ],
)
async def test_router_launch_supplies_local_cwd_and_preserves_explicit_scope(
    tmp_path, monkeypatch, arguments
):
    monkeypatch.chdir(tmp_path)
    server = SimpleNamespace(
        sockets=[SimpleNamespace(getsockname=lambda: ("127.0.0.1", 4501))],
        close=Mock(),
        wait_closed=AsyncMock(),
    )
    monkeypatch.setattr("codex_self_router.cli.make_server", AsyncMock(return_value=server))
    monkeypatch.setattr(
        "codex_self_router.cli.require_compatible_codex", AsyncMock(return_value="test Codex")
    )
    launch = AsyncMock(return_value=SimpleNamespace(wait=AsyncMock(return_value=0)))
    monkeypatch.setattr("codex_self_router.cli.asyncio.create_subprocess_exec", launch)
    args = build_parser().parse_args(["run", "--", *arguments])
    assert await command_run(args, Path("/test/codex"), ReportStore(tmp_path)) == 0
    passed = list(launch.await_args.args[3:])
    if (
        arguments[0] in {"resume", "fork"}
        and len(arguments) <= 2
        and not any(argument.startswith(("-C", "--cd")) for argument in arguments)
    ):
        assert passed == ["--cd", str(tmp_path), *arguments]
    else:
        assert passed == arguments
    server.close.assert_called_once()
    server.wait_closed.assert_awaited_once()
