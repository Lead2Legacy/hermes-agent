"""Claude CLI subprocess adapter.

Text-only first slice for provider ``claude-cli`` / api_mode ``claude_cli``.
The adapter intentionally does not expose Hermes tools yet; tool access will be
added via an MCP bridge in a later phase.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class ClaudeCliError(RuntimeError):
    """Raised when the Claude CLI subprocess fails."""


_SECRET_ENV_FRAGMENTS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
)

_SAFE_ENV_NAMES = {
    "HOME",
    "PATH",
    "SHELL",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "HERMES_HOME",
    "CLAUDE_CONFIG_DIR",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
}


def sanitized_claude_cli_env(source: dict[str, str] | None = None) -> dict[str, str]:
    """Return a minimal env for Claude CLI without leaking provider secrets."""
    src = dict(source or os.environ)
    env: dict[str, str] = {}
    for key, value in src.items():
        upper = key.upper()
        if upper in _SAFE_ENV_NAMES:
            env[key] = value
            continue
        if upper.startswith("CLAUDE_") and not any(fragment in upper for fragment in _SECRET_ENV_FRAGMENTS):
            env[key] = value
    return env


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "image_url":
                    parts.append("[image omitted]")
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def build_claude_cli_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten Hermes/OpenAI-style messages into a Claude CLI print prompt."""
    lines: list[str] = []
    role_labels = {
        "system": "System",
        "developer": "Developer",
        "user": "User",
        "assistant": "Assistant",
        "tool": "Tool",
    }
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").strip().lower()
        label = role_labels.get(role, role.title() or "User")
        text = _content_to_text(msg.get("content")).strip()
        if not text:
            continue
        lines.append(f"{label}:\n{text}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def _extract_stream_json_text(output: str) -> str:
    pieces: list[str] = []
    for raw_line in (output or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        # Claude CLI stream-json commonly emits assistant/message payloads with
        # content blocks. Keep the parser permissive because CLI event shapes
        # have changed across releases.
        for container in (event, event.get("message") if isinstance(event.get("message"), dict) else None):
            if not isinstance(container, dict):
                continue
            content = container.get("content")
            if isinstance(content, str):
                pieces.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        pieces.append(block["text"])
        if isinstance(event.get("delta"), dict) and isinstance(event["delta"].get("text"), str):
            pieces.append(event["delta"]["text"])
    return "".join(pieces).strip()


def normalize_claude_cli_response(output: str, *, model: str) -> SimpleNamespace:
    """Convert Claude CLI stdout into the OpenAI-compatible response shape."""
    text = _extract_stream_json_text(output)
    if not text:
        text = (output or "").strip()
    message = SimpleNamespace(
        role="assistant",
        content=text,
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    return SimpleNamespace(
        id=f"claude-cli-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=0,
        model=model,
        choices=[choice],
        usage=usage,
    )


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _should_enable_mcp_tools(api_kwargs: dict[str, Any]) -> bool:
    return bool(api_kwargs.get("mcp_tools")) or _env_flag("HERMES_CLAUDE_CLI_MCP_TOOLS")


def _build_mcp_env() -> dict[str, str]:
    env: dict[str, str] = {
        "HERMES_QUIET": "1",
        "HERMES_REDACT_SECRETS": "true",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    for name in ("HERMES_HOME", "HERMES_PROFILE", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK"):
        value = os.getenv(name)
        if value:
            env[name] = value
    return env


def _write_hermes_tools_mcp_config() -> str:
    config = {
        "mcpServers": {
            "hermes-tools": {
                "command": sys.executable,
                "args": ["-m", "agent.transports.claude_cli_tools_mcp_server"],
                "env": _build_mcp_env(),
            }
        }
    }
    fh = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="hermes-claude-mcp-",
        delete=False,
    )
    try:
        json.dump(config, fh)
        fh.flush()
        return fh.name
    finally:
        fh.close()


def _claude_mcp_tool_names() -> list[str]:
    from agent.transports.claude_cli_tools_mcp_server import EXPOSED_TOOLS

    return [f"mcp__hermes-tools__{name}" for name in EXPOSED_TOOLS]


def _build_claude_cli_command(claude_bin: str, model: str, api_kwargs: dict[str, Any]) -> tuple[list[str], str | None]:
    # Disable Claude Code's built-in shell. Hermes should own the execution
    # boundary; when MCP is enabled below, only explicitly allow-listed Hermes
    # MCP tools may add external capabilities.
    cmd = [
        claude_bin,
        "-p",
        "--model",
        model,
        "--output-format",
        "text",
        "--disallowedTools",
        "Bash",
    ]
    mcp_config_path: str | None = None
    if _should_enable_mcp_tools(api_kwargs):
        mcp_config_path = _write_hermes_tools_mcp_config()
        cmd.extend([
            "--mcp-config",
            mcp_config_path,
            "--strict-mcp-config",
            "--allowedTools",
            *_claude_mcp_tool_names(),
        ])
    return cmd, mcp_config_path


def run_claude_cli_completion(
    api_kwargs: dict[str, Any],
    *,
    binary: str | None = None,
    env: dict[str, str] | None = None,
) -> SimpleNamespace:
    """Run ``claude -p`` for a single text-only completion."""
    model = str(api_kwargs.get("model") or "").strip() or "claude-sonnet-4-6"
    messages = api_kwargs.get("messages") or []
    prompt = build_claude_cli_prompt(messages)
    timeout = api_kwargs.get("timeout") or int(os.getenv("HERMES_CLAUDE_CLI_TIMEOUT", "120"))
    claude_bin = binary or os.getenv("HERMES_CLAUDE_CLI_PATH") or shutil.which("claude") or "/Users/atlas/.local/bin/claude"
    cmd, mcp_config_path = _build_claude_cli_command(claude_bin, model, api_kwargs)
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env or sanitized_claude_cli_env(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClaudeCliError(f"Claude CLI timed out after {timeout}s") from exc
    except OSError as exc:
        raise ClaudeCliError(f"Claude CLI failed to start: {exc}") from exc
    finally:
        if mcp_config_path:
            try:
                os.unlink(mcp_config_path)
            except OSError:
                pass
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "Claude CLI exited non-zero").strip()
        raise ClaudeCliError(detail)
    return normalize_claude_cli_response(proc.stdout, model=model)
