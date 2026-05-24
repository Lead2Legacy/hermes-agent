import json
import subprocess
from types import SimpleNamespace

import pytest

from agent.claude_cli_adapter import (
    ClaudeCliError,
    _build_claude_cli_command,
    _build_mcp_env,
    build_claude_cli_prompt,
    normalize_claude_cli_response,
    run_claude_cli_completion,
    sanitized_claude_cli_env,
)


def test_build_claude_cli_prompt_preserves_system_and_turn_order():
    prompt = build_claude_cli_prompt(
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Reply OK"},
        ]
    )

    assert "System:\nYou are concise." in prompt
    assert "User:\nHello" in prompt
    assert "Assistant:\nHi" in prompt
    assert prompt.rstrip().endswith("Assistant:")


def test_normalize_claude_cli_response_accepts_plain_text_output():
    response = normalize_claude_cli_response("CLAUDE_TEXT_OK\n", model="claude-sonnet-4-6")

    assert response.choices[0].message.content == "CLAUDE_TEXT_OK"
    assert response.choices[0].message.tool_calls is None
    assert response.choices[0].finish_reason == "stop"
    assert response.model == "claude-sonnet-4-6"


def test_normalize_claude_cli_response_accepts_stream_json_events():
    output = "\n".join(
        [
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": " world"}]}}),
        ]
    )

    response = normalize_claude_cli_response(output, model="claude-opus-4-7")

    assert response.choices[0].message.content == "Hello world"


def test_sanitized_claude_cli_env_preserves_path_home_and_removes_api_keys(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/Users/atlas")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-anthropic")
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes")

    env = sanitized_claude_cli_env()

    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/Users/atlas"
    assert env["HERMES_HOME"] == "/tmp/hermes"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_run_claude_cli_completion_uses_print_mode_model_and_text_output(monkeypatch):
    calls = []

    def fake_run(cmd, *, input, text, capture_output, timeout, env, check):
        calls.append(
            {
                "cmd": cmd,
                "input": input,
                "text": text,
                "capture_output": capture_output,
                "timeout": timeout,
                "env": env,
                "check": check,
            }
        )
        return SimpleNamespace(returncode=0, stdout="CLAUDE_TEXT_OK\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    response = run_claude_cli_completion(
        {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Reply exactly CLAUDE_TEXT_OK"}],
            "timeout": 12,
        },
        binary="/custom/claude",
    )

    assert calls[0]["cmd"] == [
        "/custom/claude",
        "-p",
        "--model",
        "claude-sonnet-4-6",
        "--output-format",
        "text",
        "--disallowedTools",
        "Bash",
    ]
    assert "Reply exactly CLAUDE_TEXT_OK" in calls[0]["input"]
    assert calls[0]["timeout"] == 12
    assert response.choices[0].message.content == "CLAUDE_TEXT_OK"


def test_build_claude_cli_command_adds_hermes_mcp_config_when_enabled(monkeypatch):
    monkeypatch.setattr(
        "agent.claude_cli_adapter._write_hermes_tools_mcp_config",
        lambda: "/tmp/hermes-tools.json",
    )
    monkeypatch.setattr(
        "agent.claude_cli_adapter._claude_mcp_tool_names",
        lambda: ["mcp__hermes-tools__skills_list", "mcp__hermes-tools__web_search"],
    )

    cmd, config_path = _build_claude_cli_command(
        "/custom/claude",
        "claude-sonnet-4-6",
        {"mcp_tools": True},
    )

    assert config_path == "/tmp/hermes-tools.json"
    assert cmd == [
        "/custom/claude",
        "-p",
        "--model",
        "claude-sonnet-4-6",
        "--output-format",
        "text",
        "--disallowedTools",
        "Bash",
        "--mcp-config",
        "/tmp/hermes-tools.json",
        "--strict-mcp-config",
        "--allowedTools",
        "mcp__hermes-tools__skills_list",
        "mcp__hermes-tools__web_search",
    ]


def test_build_mcp_env_is_minimal_and_redacts_provider_keys(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-home")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    env = _build_mcp_env()

    assert env["HERMES_HOME"] == "/tmp/hermes-home"
    assert env["HERMES_QUIET"] == "1"
    assert env["HERMES_REDACT_SECRETS"] == "true"
    assert "PYTHONPATH" in env
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_run_claude_cli_completion_raises_on_nonzero_exit(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="bad auth")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ClaudeCliError, match="bad auth"):
        run_claude_cli_completion(
            {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
            binary="/custom/claude",
        )
