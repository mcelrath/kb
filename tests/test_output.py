"""Unit tests for kb/cli/output.py — agent/user mode, color, ANSI-aware width."""
import pytest

from kb.cli import output


def test_is_agent_env_override(monkeypatch):
    monkeypatch.setenv("KB_AGENT", "1")
    assert output.is_agent() is True
    monkeypatch.setenv("KB_AGENT", "0")
    assert output.is_agent() is False


def test_is_agent_claudecode(monkeypatch):
    monkeypatch.delenv("KB_AGENT", raising=False)
    monkeypatch.setenv("CLAUDECODE", "1")
    assert output.is_agent() is True


def test_color_noop_in_agent_mode(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    assert output.c("hi", "red") == "hi"          # no escapes for agents


def test_color_wraps_in_user_mode(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    out = output.c("hi", "red")
    assert out.startswith("\033[31m") and out.endswith(output.RESET) and "hi" in out


def test_color_raw_code_and_falsy(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    assert output.c("x", "\033[35m").startswith("\033[35m")
    assert output.c("x", None) == "x"             # no color → unchanged


def test_visible_len_ignores_ansi():
    assert output.visible_len("\033[31mabc\033[0m") == 3
    assert output.visible_len("plain") == 5


def test_term_width_none_for_agents(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    assert output.term_width() is None            # agents never truncate


def test_term_width_int_for_users(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    w = output.term_width(default=80)
    assert isinstance(w, int) and w > 0


def test_truncate_none_or_fits_unchanged():
    assert output.truncate("hello", None) == "hello"
    assert output.truncate("hello", 0) == "hello"
    assert output.truncate("hello", 10) == "hello"


def test_truncate_plain_counts_visible():
    out = output.truncate("abcdefghij", 5)        # 10 chars -> 5
    assert output.visible_len(out) == 5
    assert out.endswith("…") or out.endswith("…\033[0m")


def test_truncate_ansi_aware(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    colored = "\033[31mABCDEFGHIJ\033[0m"          # 10 visible, wrapped in red
    out = output.truncate(colored, 4)
    assert output.visible_len(out) == 4            # ellipsis counts as 1 visible
    assert "\033[31m" in out                       # opening escape preserved
    assert out.endswith(output.RESET)              # closed so color can't bleed


def test_fit_line_passthrough_for_agents(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", True)
    long = "x" * 500
    assert output.fit_line(long) == long           # agents: never truncated


def test_fit_line_truncates_for_users(monkeypatch):
    monkeypatch.setattr(output, "AGENT_MODE", False)
    monkeypatch.setattr(output, "term_width", lambda default=100: 20)
    out = output.fit_line("y" * 500)
    assert output.visible_len(out) == 20
