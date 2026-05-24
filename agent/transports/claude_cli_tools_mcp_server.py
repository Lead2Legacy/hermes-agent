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
from typing import Any, Optional

logger = logging.getLogger(__name__)

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


def _build_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(f"requires mcp package: {exc}") from exc

    from model_tools import get_tool_definitions, handle_function_call

    mcp = FastMCP(
        "hermes-tools",
        instructions="Hermes tool surface for Claude CLI runtime.",
    )

    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

    exposed_count = 0
    for name in EXPOSED_TOOLS:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug("skipping %s — not registered", name)
            continue
        description = spec.get("description") or f"Hermes {name} tool"

        def _make_handler(tool_name: str):
            def _dispatch(**kwargs: Any) -> str:
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
