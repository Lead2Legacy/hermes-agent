import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent.claude_cli_adapter import (
    ClaudeCliError,
    ClaudeCliQuotaError,
    ClaudeSessionManager,
    _build_claude_cli_command,
    _build_mcp_env,
    build_claude_cli_prompt,
    build_claude_cli_resume_prompt,
    cancel_claude_cli,
    normalize_claude_cli_response,
    run_claude_cli_completion,
    run_claude_cli_streaming,
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
            "mcp_tools": False,
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
    monkeypatch.setenv("HERMES_SESSION_ID", "session-123")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "456")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    env = _build_mcp_env()

    assert env["HERMES_HOME"] == "/tmp/hermes-home"
    assert env["HERMES_SESSION_ID"] == "session-123"
    assert env["HERMES_KANBAN_TASK"] == "task-123"
    assert env["HERMES_KANBAN_RUN_ID"] == "456"
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
            {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
                "mcp_tools": False,
            },
            binary="/custom/claude",
        )


def test_build_claude_cli_command_enables_mcp_tools_by_default(monkeypatch):
    """No env var, no kwarg → MCP bridge is wired in automatically."""
    monkeypatch.delenv("HERMES_CLAUDE_CLI_MCP_TOOLS", raising=False)
    monkeypatch.setattr(
        "agent.claude_cli_adapter._write_hermes_tools_mcp_config",
        lambda: "/tmp/hermes-tools.json",
    )
    monkeypatch.setattr(
        "agent.claude_cli_adapter._claude_mcp_tool_names",
        lambda: ["mcp__hermes-tools__terminal", "mcp__hermes-tools__read_file"],
    )

    cmd, config_path = _build_claude_cli_command(
        "/custom/claude",
        "claude-sonnet-4-6",
        {},
    )

    assert config_path == "/tmp/hermes-tools.json"
    assert "--mcp-config" in cmd
    assert "--strict-mcp-config" in cmd
    assert "mcp__hermes-tools__terminal" in cmd


def test_build_claude_cli_command_respects_explicit_opt_out():
    """``mcp_tools=False`` kwarg suppresses MCP wiring even when env says on."""
    cmd, config_path = _build_claude_cli_command(
        "/custom/claude",
        "claude-sonnet-4-6",
        {"mcp_tools": False},
    )

    assert config_path is None
    assert "--mcp-config" not in cmd


def _write_fake_claude(tmp_path, body: str):
    script = tmp_path / "fake_claude.py"
    script.write_text(body, encoding="utf-8")
    return sys.executable, script


def _python_binary_with_script(binary, script):
    # run_claude_cli_streaming expects a single executable path. Wrap the script
    # in a tiny shell shim so the normal Claude args are still passed through.
    wrapper = script.with_suffix(".sh")
    wrapper.write_text(f"#! /bin/sh\nexec {binary} {script} \"$@\"\n", encoding="utf-8")
    wrapper.chmod(0o755)
    return str(wrapper)


def test_build_claude_cli_command_stream_json_adds_resume_flags():
    cmd, config_path = _build_claude_cli_command(
        "/custom/claude",
        "claude-opus-4-7",
        {"mcp_tools": False, "output_format": "stream-json", "resume_session_id": "sess_123"},
    )

    assert config_path is None
    assert "--output-format" in cmd
    assert "stream-json" in cmd
    assert "--include-partial-messages" in cmd
    assert "--verbose" in cmd
    assert "--setting-sources" in cmd
    assert cmd[-2:] == ["--resume", "sess_123"]


def test_build_claude_cli_resume_prompt_uses_only_latest_user_turn():
    prompt = build_claude_cli_resume_prompt(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ]
    )

    assert prompt == "second"


def test_run_claude_cli_streaming_captures_session_usage_and_deltas(tmp_path):
    binary, script = _write_fake_claude(
        tmp_path,
        """
import json, sys
sys.stdin.read()
print(json.dumps({'type':'system','session_id':'sess_abc'}), flush=True)
print(json.dumps({'type':'assistant','message':{'content':[{'type':'text','text':'Hel'}]}}), flush=True)
print(json.dumps({'type':'assistant','delta':{'text':'lo'}}), flush=True)
print(json.dumps({'type':'result','usage':{'input_tokens':7,'output_tokens':2,'cache_read_input_tokens':3}}), flush=True)
""",
    )
    deltas = []
    first = []
    agent = SimpleNamespace(_interrupt_requested=False)

    response = run_claude_cli_streaming(
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "mcp_tools": False},
        agent=agent,
        binary=_python_binary_with_script(binary, script),
        on_text_delta=deltas.append,
        on_first_delta=lambda: first.append(True),
    )

    assert response.choices[0].message.content == "Hello"
    assert deltas == ["Hel", "lo"]
    assert first == [True]
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 2
    assert response.usage.cache_read_input_tokens == 3
    assert agent._claude_session_id == "sess_abc"


def test_run_claude_cli_streaming_resumes_with_newest_user_turn_only(tmp_path):
    record = tmp_path / "record.jsonl"
    binary, script = _write_fake_claude(
        tmp_path,
        f"""
import json, sys
stdin = sys.stdin.read()
with open({str(record)!r}, 'a', encoding='utf-8') as fh:
    fh.write(json.dumps({{'argv': sys.argv[1:], 'stdin': stdin}}) + '\\n')
print(json.dumps({{'type':'system','session_id':'sess_next'}}), flush=True)
print(json.dumps({{'type':'assistant','delta':{{'text':'ok'}}}}), flush=True)
""",
    )
    manager = ClaudeSessionManager()
    manager.update(session_id="sess_prev", model="claude-sonnet-4-6")
    agent = SimpleNamespace(_interrupt_requested=False, _claude_session_manager=manager)

    run_claude_cli_streaming(
        {
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "new"},
            ],
            "mcp_tools": False,
        },
        agent=agent,
        binary=_python_binary_with_script(binary, script),
    )

    rec = json.loads(record.read_text(encoding="utf-8").splitlines()[0])
    assert "--resume" in rec["argv"]
    assert "sess_prev" in rec["argv"]
    assert rec["stdin"].strip() == "new"
    assert agent._claude_session_id == "sess_next"


def test_run_claude_cli_streaming_recovers_from_stale_resume(tmp_path):
    record = tmp_path / "runs.txt"
    binary, script = _write_fake_claude(
        tmp_path,
        f"""
import json, sys
argv = sys.argv[1:]
with open({str(record)!r}, 'a', encoding='utf-8') as fh:
    fh.write('resume' if '--resume' in argv else 'fresh')
    fh.write('\\n')
if '--resume' in argv:
    print('session not found', file=sys.stderr)
    sys.exit(1)
print(json.dumps({{'type':'system','session_id':'sess_fresh'}}), flush=True)
print(json.dumps({{'type':'assistant','delta':{{'text':'fresh ok'}}}}), flush=True)
""",
    )
    manager = ClaudeSessionManager()
    manager.update(session_id="stale", model="claude-sonnet-4-6")
    agent = SimpleNamespace(_interrupt_requested=False, _claude_session_manager=manager)

    response = run_claude_cli_streaming(
        {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "mcp_tools": False},
        agent=agent,
        binary=_python_binary_with_script(binary, script),
    )

    assert record.read_text(encoding="utf-8").splitlines() == ["resume", "fresh"]
    assert response.choices[0].message.content == "fresh ok"
    assert agent._claude_session_id == "sess_fresh"


def test_claude_session_manager_invalidates_on_model_switch():
    manager = ClaudeSessionManager()
    manager.update(session_id="sess_1", model="claude-sonnet-4-6")

    assert manager.get_resume_session_id("claude-opus-4-7") is None
    assert manager.session_id is None


def test_run_claude_cli_streaming_raises_quota_error_from_json(tmp_path):
    binary, script = _write_fake_claude(
        tmp_path,
        """
import json, sys
sys.stdin.read()
print(json.dumps({'type':'error','error':{'message':"You're out of extra usage"}}), flush=True)
""",
    )

    with pytest.raises(ClaudeCliQuotaError):
        run_claude_cli_streaming(
            {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}], "mcp_tools": False},
            binary=_python_binary_with_script(binary, script),
        )


def test_cancel_claude_cli_terminates_live_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    agent = SimpleNamespace(_claude_cli_process=proc)

    cancel_claude_cli(agent)

    assert proc.poll() is not None
