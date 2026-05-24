"""Hermes-tools-as-MCP server for the claude-cli runtime.

Sibling of hermes_tools_mcp_server.py (Codex variant). The difference
is policy: Codex has its own shell/file tools we trust, so the Codex
MCP server intentionally hides those Hermes tools. Claude CLI runs
with Bash disabled (see claude_cli_adapter._build_claude_cli_command)
so Hermes owns the execution boundary — therefore we DO expose
terminal/read_file/write_file/patch/search_files/process here.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _load_profile_env() -> bool:
    """Load the active Hermes profile's .env into the MCP server process.

    The Claude CLI adapter intentionally starts this MCP child with a minimal
    non-secret environment so provider credentials are not handed to Claude
    itself through process inheritance. Tools such as web_search still need
    their backend keys, though, and normal Hermes runtime entry points load
    those from HERMES_HOME/.env before tool discovery. Mirror that behavior
    inside the MCP server process so GPT/OpenAI and Claude CLI see the same
    available Hermes tools when the same profile is active.
    """
    try:
        from dotenv import load_dotenv
        from hermes_constants import get_hermes_home
    except Exception as exc:  # pragma: no cover - optional dependency guard
        logger.debug("profile env load skipped: %s", exc)
        return False

    env_path = Path(get_hermes_home()) / ".env"
    if not env_path.exists():
        return False
    try:
        return bool(load_dotenv(str(env_path), override=True, encoding="utf-8"))
    except UnicodeDecodeError:
        return bool(load_dotenv(str(env_path), override=True, encoding="latin-1"))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("profile env load failed for %s: %s", env_path, exc)
        return False


EXPOSED_TOOLS: tuple[str, ...] = (
    "terminal", "process", "read_file", "write_file", "patch", "search_files",
    "web_search", "web_extract",
    "browser_navigate", "browser_click", "browser_type", "browser_press",
    "browser_snapshot", "browser_scroll", "browser_back",
    "browser_get_images", "browser_console", "browser_vision",
    "vision_analyze", "image_generate",
    "skill_view", "skills_list", "text_to_speech",
    "kanban_complete", "kanban_block", "kanban_comment", "kanban_heartbeat",
    "kanban_show", "kanban_list", "kanban_create", "kanban_unblock", "kanban_link",
)


def _available_exposed_tool_specs() -> dict[str, dict[str, Any]]:
    """Return currently available Hermes tool schemas for EXPOSED_TOOLS.

    This applies the same toolset/check_fn filtering as the normal Hermes
    runtime. Names may be listed in EXPOSED_TOOLS as a policy allow-list but
    absent here when the active profile lacks credentials (web/vision/TTS) or
    runtime gates (kanban worker/orchestrator env).
    """
    from model_tools import get_tool_definitions

    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }
    return {name: all_defs[name] for name in EXPOSED_TOOLS if name in all_defs}


def available_mcp_tool_names() -> list[str]:
    """Return Claude CLI MCP tool names that will actually be registered."""
    _load_profile_env()
    return [f"mcp__hermes-tools__{name}" for name in _available_exposed_tool_specs()]


def _build_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(f"requires mcp package: {exc}") from exc

    from model_tools import handle_function_call

    mcp = FastMCP(
        "hermes-tools",
        instructions="Hermes tool surface for Claude CLI runtime.",
    )

    available_specs = _available_exposed_tool_specs()

    exposed_count = 0
    for name in EXPOSED_TOOLS:
        spec = available_specs.get(name)
        if spec is None:
            logger.debug("skipping %s — not registered", name)
            continue
        description = spec.get("description") or f"Hermes {name} tool"

        def _make_handler(tool_name: str):
            def _dispatch(**kwargs: Any) -> str:
                # FastMCP wraps **kwargs signatures as a single "kwargs" JSON
                # parameter; clients send {"kwargs": {...}} and we receive
                # kwargs={"kwargs": {...}}. Unwrap when that pattern is detected.
                if (
                    len(kwargs) == 1
                    and "kwargs" in kwargs
                    and isinstance(kwargs["kwargs"], dict)
                ):
                    kwargs = kwargs["kwargs"]
                try:
                    return handle_function_call(tool_name, kwargs or {})
                except Exception as exc:
                    logger.exception("tool %s raised", tool_name)
                    return json.dumps({"error": str(exc), "tool": tool_name})
            _dispatch.__name__ = tool_name
            _dispatch.__doc__ = description
            return _dispatch

        try:
            mcp.add_tool(_make_handler(name), name=name, description=description)
        except TypeError:
            handler = _make_handler(name)
            handler = mcp.tool(name=name, description=description)(handler)
        exposed_count += 1

    logger.info("claude-cli MCP server registered %d/%d tools", exposed_count, len(EXPOSED_TOOLS))
    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")
    _load_profile_env()
    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"cannot start: {exc}\n")
        return 2
    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("crashed")
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
