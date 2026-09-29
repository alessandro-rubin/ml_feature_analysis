"""The tessa-dashboard entry point builds the right streamlit command."""

import sys
from pathlib import Path

import pytest

from tessa.dashboard import cli


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    seen: list[list[str]] = []
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd: seen.append(cmd) or 0)
    return seen


def test_launches_installed_app_with_root(calls: list[list[str]]):
    assert cli.main(["my/runs", "--", "--server.port", "8502"]) == 0
    (cmd,) = calls
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert Path(cmd[4]).name == "app.py" and Path(cmd[4]).exists()
    assert cmd[5:] == ["--server.port", "8502", "--", "--root", "my/runs"]


def test_default_root(calls: list[list[str]]):
    cli.main([])
    assert calls[0][-2:] == ["--root", "outputs/runs"]


def test_missing_streamlit(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    assert cli.main([]) == 1
    assert "dashboard" in capsys.readouterr().err
