"""Claude CLI subprocess adapter.

Adapter for provider ``claude-cli`` / api_mode ``claude_cli``. Hermes tools
are exposed to Claude CLI via the ``claude_cli_tools_mcp_server`` MCP bridge
by default. Claude's built-in Bash is disabled, so the MCP bridge is the
only execution surface — Hermes owns the tool boundary end to end.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


class ClaudeCliError(RuntimeError):
    """Raised when the Claude CLI subprocess fails."""


class ClaudeCliQuotaError(ClaudeCliError):
    """Raised when Claude CLI reports subscription/usage quota exhaustion."""


class ClaudeCliStaleSessionError(ClaudeCliError):
    """Raised internally when ``--resume`` points at an unusable session."""


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

_CLAUDE_SESSION_ID_FIELDS = (
    "session_id",
    "sessionId",
    "conversation_id",
    "conversationId",
)

_QUOTA_ERROR_SNIPPETS = (
    "out of extra usage",
    "usage limit",
    "rate limit exceeded",
    "quota exceeded",
    "subscription",
)

_STALE_SESSION_SNIPPETS = (
    "session not found",
    "session expired",
    "invalid session",
    "no conversation found",
    "conversation not found",
    "could not resume",
    "model mismatch",
)

_STREAM_SENTINEL = object()


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


def build_claude_cli_resume_prompt(messages: list[dict[str, Any]]) -> str:
    """Return only the newest user turn for ``--resume`` calls."""
    for msg in reversed(messages or []):
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role") or "").strip().lower() == "user":
            text = _content_to_text(msg.get("content")).strip()
            if text:
                return text
    return build_claude_cli_prompt(messages)


def _extract_stream_json_text(output: str) -> str:
    state = _StreamState()
    for raw_line in (output or "").splitlines():
        event = _parse_json_line(raw_line)
        if event:
            state.apply_event(event)
    return state.text.strip()


def _usage_namespace_from_dict(usage: dict[str, Any] | None) -> SimpleNamespace:
    usage = usage or {}

    def _to_int(value: Any) -> int:
        try:
            return int(value or 0)
        except Exception:
            return 0

    input_tokens = _to_int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
    output_tokens = _to_int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
    cache_read = _to_int(usage.get("cache_read_input_tokens", usage.get("cache_read_tokens", 0)))
    cache_write = _to_int(usage.get("cache_creation_input_tokens", usage.get("cache_write_tokens", 0)))
    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    if isinstance(prompt_details, dict):
        cache_read = cache_read or _to_int(prompt_details.get("cached_tokens"))
        cache_write = cache_write or _to_int(prompt_details.get("cache_creation_tokens") or prompt_details.get("cache_write_tokens"))
    total = _to_int(usage.get("total_tokens")) or input_tokens + output_tokens + cache_read + cache_write
    return SimpleNamespace(
        prompt_tokens=input_tokens + cache_read + cache_write,
        completion_tokens=output_tokens,
        total_tokens=total,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cache_read, cache_creation_tokens=cache_write),
    )


def normalize_claude_cli_response(
    output: str,
    *,
    model: str,
    usage: dict[str, Any] | None = None,
    session_id: str | None = None,
    stop_reason: str | None = None,
) -> SimpleNamespace:
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
    choice = SimpleNamespace(index=0, message=message, finish_reason=stop_reason or "stop")
    return SimpleNamespace(
        id=f"claude-cli-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=0,
        model=model,
        choices=[choice],
        usage=_usage_namespace_from_dict(usage),
        provider_data={"session_id": session_id} if session_id else {},
    )


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _should_enable_mcp_tools(api_kwargs: dict[str, Any]) -> bool:
    """Default-on. Opt out with ``mcp_tools=False`` kwarg or
    ``HERMES_CLAUDE_CLI_MCP_TOOLS`` set to 0/false/no/off."""
    if "mcp_tools" in api_kwargs:
        return bool(api_kwargs["mcp_tools"])
    raw = os.getenv("HERMES_CLAUDE_CLI_MCP_TOOLS")
    if raw is None or raw == "":
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_mcp_env() -> dict[str, str]:
    env: dict[str, str] = {
        "HERMES_QUIET": "1",
        "HERMES_REDACT_SECRETS": "true",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    for name in (
        "HERMES_HOME",
        "HERMES_PROFILE",
        "HERMES_SESSION_ID",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
    ):
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
    from agent.transports.claude_cli_tools_mcp_server import available_mcp_tool_names

    return available_mcp_tool_names()


def _build_claude_cli_command(claude_bin: str, model: str, api_kwargs: dict[str, Any]) -> tuple[list[str], str | None]:
    # Disable Claude Code's built-in shell. Hermes should own the execution
    # boundary; when MCP is enabled below, only explicitly allow-listed Hermes
    # MCP tools may add external capabilities.
    output_format = str(api_kwargs.get("output_format") or "text")
    cmd = [
        claude_bin,
        "-p",
        "--model",
        model,
        "--output-format",
        output_format,
        "--disallowedTools",
        "Bash",
    ]
    if output_format == "stream-json":
        cmd.extend(["--include-partial-messages", "--verbose", "--setting-sources", "user"])
    session_id = api_kwargs.get("resume_session_id")
    if session_id:
        cmd.extend(["--resume", str(session_id)])
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


def _parse_json_line(raw_line: str) -> dict[str, Any] | None:
    line = (raw_line or "").strip()
    if not line:
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _extract_session_id(event: dict[str, Any]) -> str | None:
    for field in _CLAUDE_SESSION_ID_FIELDS:
        value = event.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    message = event.get("message")
    if isinstance(message, dict):
        for field in _CLAUDE_SESSION_ID_FIELDS:
            value = message.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _is_quota_text(text: str) -> bool:
    lower = (text or "").lower()
    return any(snippet in lower for snippet in _QUOTA_ERROR_SNIPPETS)


def _is_stale_session_text(text: str) -> bool:
    lower = (text or "").lower()
    return any(snippet in lower for snippet in _STALE_SESSION_SNIPPETS)


def _event_error_text(event: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("error", "message", "result"):
        value = event.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for subkey in ("message", "error", "text", "type"):
                sub = value.get(subkey)
                if isinstance(sub, str):
                    parts.append(sub)
    return " ".join(parts)


def _strip_mcp_prefix(name: str) -> str:
    """Strip the ``mcp__hermes-tools__`` prefix so tool names match the OpenAI surface."""
    prefix = "mcp__hermes-tools__"
    if isinstance(name, str) and name.startswith(prefix):
        return name[len(prefix):]
    return name


def _extract_tool_result_text(content: Any) -> str:
    """Pull the user-visible text out of a Claude ``tool_result`` content payload."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


class _StreamState:
    def __init__(self) -> None:
        self.pieces: list[str] = []
        self.usage: dict[str, Any] = {}
        self.session_id: str | None = None
        self.stop_reason: str | None = None
        # Tool-event mirroring: track tool_use and tool_result blocks so we
        # only fire each callback once even if the same id appears in both a
        # streaming partial event and the final assistant message.
        self.tool_calls: dict[str, dict[str, Any]] = {}
        self.tool_results_seen: set[str] = set()

    @property
    def text(self) -> str:
        return "".join(self.pieces)

    def apply_event(self, event: dict[str, Any]) -> SimpleNamespace:
        deltas: list[str] = []
        tool_starts: list[tuple[str, str, dict[str, Any]]] = []
        tool_results: list[tuple[str, str, dict[str, Any], str]] = []

        session_id = _extract_session_id(event)
        if session_id:
            self.session_id = session_id
        usage = event.get("usage")
        if isinstance(usage, dict):
            self.usage.update(usage)
        result = event.get("result")
        if isinstance(result, dict):
            usage = result.get("usage")
            if isinstance(usage, dict):
                self.usage.update(usage)
        if isinstance(event.get("stop_reason"), str):
            self.stop_reason = event["stop_reason"]
        if isinstance(event.get("subtype"), str) and event.get("type") == "result":
            subtype = event["subtype"]
            self.stop_reason = "stop" if subtype == "success" else (subtype or self.stop_reason)

        containers = [event]
        if isinstance(event.get("message"), dict):
            containers.append(event["message"])
        if isinstance(result, dict):
            containers.append(result)
        cb_start = event.get("content_block")
        if isinstance(cb_start, dict):
            containers.append({"content": [cb_start]})

        for container in containers:
            content = container.get("content")
            if isinstance(content, str):
                deltas.append(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "tool_use":
                        tid = block.get("id")
                        name = block.get("name")
                        if isinstance(tid, str) and isinstance(name, str) and tid not in self.tool_calls:
                            display_name = _strip_mcp_prefix(name)
                            args = block.get("input")
                            args_dict: dict[str, Any] = args if isinstance(args, dict) else {}
                            self.tool_calls[tid] = {"name": display_name, "input": args_dict}
                            tool_starts.append((tid, display_name, args_dict))
                    elif btype == "tool_result":
                        tid = block.get("tool_use_id")
                        if isinstance(tid, str) and tid not in self.tool_results_seen:
                            self.tool_results_seen.add(tid)
                            info = self.tool_calls.get(tid, {})
                            tool_results.append(
                                (
                                    tid,
                                    info.get("name", ""),
                                    info.get("input", {}),
                                    _extract_tool_result_text(block.get("content")),
                                )
                            )
                    elif isinstance(block.get("text"), str):
                        deltas.append(block["text"])
        delta = event.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            deltas.append(delta["text"])
        text = event.get("text")
        if isinstance(text, str) and event.get("type") in {"content_block_delta", "assistant_delta", "delta"}:
            deltas.append(text)

        if deltas:
            self.pieces.extend(deltas)
        return SimpleNamespace(
            text_deltas=deltas,
            tool_starts=tool_starts,
            tool_results=tool_results,
        )


class ClaudeSessionManager:
    """Per-agent Claude CLI session state and process lifecycle."""

    def __init__(self) -> None:
        self.session_id: str | None = None
        self.model: str | None = None
        self.current_process: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def get_resume_session_id(self, model: str) -> str | None:
        if self.model and self.model != model:
            self.invalidate()
            return None
        return self.session_id

    def update(self, *, session_id: str | None, model: str) -> None:
        if session_id:
            self.session_id = session_id
            self.model = model

    def invalidate(self) -> None:
        self.session_id = None
        self.model = None

    def set_process(self, proc: subprocess.Popen | None) -> None:
        self.current_process = proc

    def cancel_current(self) -> None:
        proc = self.current_process
        if not proc or proc.poll() is not None:
            return
        _terminate_process(proc)


def _get_session_manager(agent: Any | None) -> ClaudeSessionManager | None:
    if agent is None:
        return None
    manager = getattr(agent, "_claude_session_manager", None)
    if manager is None:
        manager = ClaudeSessionManager()
        setattr(agent, "_claude_session_manager", manager)
    # Keep compatibility with simple status/introspection fields.
    if getattr(agent, "_claude_session_id", None) and not manager.session_id:
        manager.session_id = getattr(agent, "_claude_session_id")
        manager.model = getattr(agent, "_claude_session_model", None)
    return manager


def _terminate_process(proc: subprocess.Popen, *, grace_seconds: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=1)
        except Exception:
            pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _spawn_stream_process(cmd: list[str], prompt: str, env: dict[str, str]) -> subprocess.Popen:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    if proc.stdin is not None:
        try:
            proc.stdin.write(prompt)
            if not prompt.endswith("\n"):
                proc.stdin.write("\n")
            proc.stdin.close()
        except BrokenPipeError:
            pass
    return proc


def _consume_stream_jsonl(
    proc: subprocess.Popen,
    *,
    model: str,
    on_text_delta: Callable[[str], None] | None = None,
    on_first_delta: Callable[[], None] | None = None,
    on_tool_start: Callable[[str, str, dict[str, Any]], None] | None = None,
    on_tool_result: Callable[[str, str, dict[str, Any], str], None] | None = None,
    interrupt_check: Callable[[], bool] | None = None,
    watchdog_seconds: int = 280,
) -> SimpleNamespace:
    state = _StreamState()
    stderr_parts: list[str] = []
    first_delta_fired = False
    line_queue: queue.Queue[Any] = queue.Queue()

    def _read_stdout() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line_queue.put(line)
        finally:
            line_queue.put(_STREAM_SENTINEL)

    def _read_stderr() -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            stderr_parts.append(line)

    stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    last_activity = time.time()
    stdout_done = False

    while not stdout_done:
        if interrupt_check and interrupt_check():
            _terminate_process(proc)
            raise InterruptedError("Agent interrupted during Claude CLI call")
        if time.time() - last_activity > watchdog_seconds:
            _terminate_process(proc)
            raise ClaudeCliError(f"Claude CLI stream idle for {watchdog_seconds}s")
        try:
            item = line_queue.get(timeout=0.1)
        except queue.Empty:
            if proc.poll() is not None and not stdout_thread.is_alive():
                break
            continue
        if item is _STREAM_SENTINEL:
            stdout_done = True
            continue
        last_activity = time.time()
        event = _parse_json_line(str(item))
        if event is None:
            continue
        err_text = _event_error_text(event)
        if err_text:
            if _is_quota_text(err_text):
                _terminate_process(proc)
                raise ClaudeCliQuotaError(err_text)
            if _is_stale_session_text(err_text):
                _terminate_process(proc)
                raise ClaudeCliStaleSessionError(err_text)
        outcome = state.apply_event(event)
        text_deltas = outcome.text_deltas
        if (text_deltas or outcome.tool_starts or outcome.tool_results) and not first_delta_fired:
            first_delta_fired = True
            if on_first_delta:
                try:
                    on_first_delta()
                except Exception:
                    pass
        if text_deltas and on_text_delta:
            for delta in text_deltas:
                on_text_delta(delta)
        if on_tool_start:
            for tc_id, name, args in outcome.tool_starts:
                try:
                    on_tool_start(tc_id, name, args)
                except Exception:
                    pass
        if on_tool_result:
            for tc_id, name, args, result_text in outcome.tool_results:
                try:
                    on_tool_result(tc_id, name, args, result_text)
                except Exception:
                    pass

    try:
        returncode = proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        _terminate_process(proc)
        raise ClaudeCliError("Claude CLI did not exit after stream ended")
    stderr = "".join(stderr_parts).strip()
    if returncode != 0:
        detail = stderr or "Claude CLI exited non-zero"
        if _is_quota_text(detail):
            raise ClaudeCliQuotaError(detail)
        if _is_stale_session_text(detail):
            raise ClaudeCliStaleSessionError(detail)
        raise ClaudeCliError(detail)
    return normalize_claude_cli_response(
        state.text,
        model=model,
        usage=state.usage,
        session_id=state.session_id,
        stop_reason=state.stop_reason,
    )


def _claude_binary(binary: str | None = None) -> str:
    return binary or os.getenv("HERMES_CLAUDE_CLI_PATH") or shutil.which("claude") or "/Users/atlas/.local/bin/claude"


def run_claude_cli_streaming(
    api_kwargs: dict[str, Any],
    *,
    agent: Any | None = None,
    binary: str | None = None,
    env: dict[str, str] | None = None,
    on_text_delta: Callable[[str], None] | None = None,
    on_first_delta: Callable[[], None] | None = None,
    on_tool_start: Callable[[str, str, dict[str, Any]], None] | None = None,
    on_tool_result: Callable[[str, str, dict[str, Any], str], None] | None = None,
) -> SimpleNamespace:
    """Run Claude CLI with stream-json output, session resume, usage, and cancellation."""
    model = str(api_kwargs.get("model") or "").strip() or "claude-sonnet-4-6"
    messages = api_kwargs.get("messages") or []
    manager = _get_session_manager(agent)
    lock = manager.lock if manager else threading.Lock()
    with lock:
        resume_session_id = manager.get_resume_session_id(model) if manager else None
        attempts: list[str | None] = [resume_session_id] if resume_session_id else [None]
        if resume_session_id:
            attempts.append(None)
        last_error: Exception | None = None
        for attempt_session_id in attempts:
            attempt_kwargs = dict(api_kwargs)
            attempt_kwargs["output_format"] = "stream-json"
            if attempt_session_id:
                attempt_kwargs["resume_session_id"] = attempt_session_id
            else:
                attempt_kwargs.pop("resume_session_id", None)
            prompt = build_claude_cli_resume_prompt(messages) if attempt_session_id else build_claude_cli_prompt(messages)
            cmd, mcp_config_path = _build_claude_cli_command(_claude_binary(binary), model, attempt_kwargs)
            proc: subprocess.Popen | None = None
            try:
                proc = _spawn_stream_process(cmd, prompt, env or sanitized_claude_cli_env())
                if manager:
                    manager.set_process(proc)
                if agent is not None:
                    setattr(agent, "_claude_cli_process", proc)
                response = _consume_stream_jsonl(
                    proc,
                    model=model,
                    on_text_delta=on_text_delta,
                    on_first_delta=on_first_delta,
                    on_tool_start=on_tool_start,
                    on_tool_result=on_tool_result,
                    interrupt_check=(lambda: bool(getattr(agent, "_interrupt_requested", False))) if agent is not None else None,
                    watchdog_seconds=int(os.getenv("HERMES_CLAUDE_CLI_WATCHDOG_SECONDS", "280")),
                )
                session_id = (getattr(response, "provider_data", {}) or {}).get("session_id")
                if manager:
                    manager.update(session_id=session_id, model=model)
                if agent is not None:
                    setattr(agent, "_claude_session_id", session_id or (manager.session_id if manager else None))
                    setattr(agent, "_claude_session_model", model)
                return response
            except ClaudeCliStaleSessionError as exc:
                last_error = exc
                if manager:
                    manager.invalidate()
                if not attempt_session_id:
                    raise
                continue
            finally:
                if manager:
                    manager.set_process(None)
                if agent is not None:
                    setattr(agent, "_claude_cli_process", None)
                if proc is not None and proc.poll() is None:
                    _terminate_process(proc)
                if mcp_config_path:
                    try:
                        os.unlink(mcp_config_path)
                    except OSError:
                        pass
        if last_error:
            raise last_error
        raise ClaudeCliError("Claude CLI streaming failed without response")


def cancel_claude_cli(agent: Any) -> None:
    """Terminate any live Claude CLI subprocess owned by *agent*."""
    manager = _get_session_manager(agent)
    if manager:
        manager.cancel_current()
    proc = getattr(agent, "_claude_cli_process", None)
    if proc is not None:
        _terminate_process(proc)


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
    cmd, mcp_config_path = _build_claude_cli_command(_claude_binary(binary), model, api_kwargs)
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
        if _is_quota_text(detail):
            raise ClaudeCliQuotaError(detail)
        raise ClaudeCliError(detail)
    return normalize_claude_cli_response(proc.stdout, model=model)
