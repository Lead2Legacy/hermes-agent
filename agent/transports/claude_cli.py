"""Claude CLI transport shim.

The Claude CLI adapter returns an OpenAI ChatCompletion-like SimpleNamespace.
This transport reuses the chat-completions normalization/validation surface so
the shared conversation loop can treat ``api_mode='claude_cli'`` like the other
registered runtimes.
"""

from __future__ import annotations

from agent.transports.chat_completions import ChatCompletionsTransport


class ClaudeCliTransport(ChatCompletionsTransport):
    """Transport for api_mode='claude_cli'."""

    @property
    def api_mode(self) -> str:
        return "claude_cli"


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("claude_cli", ClaudeCliTransport)
