"""Tests for the Claude CLI Hermes-tools MCP bridge.

Claude CLI runs with Bash disabled in the Hermes adapter, so this MCP
server is the execution boundary. These tests pin the policy differences
from the Codex MCP bridge and the profile/.env loading needed for parity
with the normal Hermes runtime.
"""

from __future__ import annotations


class TestClaudeCliModuleSurface:
    def test_module_imports_clean(self):
        from agent.transports import claude_cli_tools_mcp_server as m

        assert callable(m.main)
        assert callable(m._build_server)
        assert isinstance(m.EXPOSED_TOOLS, tuple)
        assert len(m.EXPOSED_TOOLS) > 0

    def test_execution_tools_are_exposed_for_claude_cli(self):
        """Unlike Codex, Claude CLI has Bash disabled by the adapter.

        Hermes must expose terminal/file/process tools through MCP so switching
        from GPT/OpenAI to Claude CLI keeps the same operator capabilities.
        """
        from agent.transports.claude_cli_tools_mcp_server import EXPOSED_TOOLS

        for required in (
            "terminal",
            "process",
            "read_file",
            "write_file",
            "patch",
            "search_files",
        ):
            assert required in EXPOSED_TOOLS

    def test_agent_loop_tools_not_exposed(self):
        """These need live AIAgent state and cannot work from stateless MCP."""
        from agent.transports.claude_cli_tools_mcp_server import EXPOSED_TOOLS

        for agent_loop_tool in ("delegate_task", "memory", "session_search", "todo"):
            assert agent_loop_tool not in EXPOSED_TOOLS


class TestEnvLoading:
    def test_load_profile_env_loads_hermes_home_dotenv(self, tmp_path, monkeypatch):
        from agent.transports import claude_cli_tools_mcp_server as m

        env_file = tmp_path / ".env"
        env_file.write_text("BRAVE_SEARCH_API_KEY=unit-test-key\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

        assert m._load_profile_env() is True
        assert m.os.environ["BRAVE_SEARCH_API_KEY"] == "unit-test-key"

    def test_load_profile_env_falls_back_to_latin1(self, tmp_path, monkeypatch):
        from agent.transports import claude_cli_tools_mcp_server as m

        env_file = tmp_path / ".env"
        env_file.write_bytes("BRAVE_SEARCH_API_KEY=caf\xe9\n".encode("latin-1"))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

        assert m._load_profile_env() is True
        assert m.os.environ["BRAVE_SEARCH_API_KEY"] == "café"

    def test_available_mcp_tool_names_loads_env_and_filters_to_available(self, monkeypatch):
        from agent.transports import claude_cli_tools_mcp_server as m

        calls: list[str] = []
        monkeypatch.setattr(m, "_load_profile_env", lambda: calls.append("env") or True)
        monkeypatch.setattr(
            m,
            "_available_exposed_tool_specs",
            lambda: {"terminal": {}, "read_file": {}},
        )

        assert m.available_mcp_tool_names() == [
            "mcp__hermes-tools__terminal",
            "mcp__hermes-tools__read_file",
        ]
        assert calls == ["env"]


class TestMain:
    def test_main_loads_profile_env_before_building_server(self, monkeypatch):
        import agent.transports.claude_cli_tools_mcp_server as m

        calls: list[str] = []

        class FakeServer:
            def run(self):
                calls.append("run")

        monkeypatch.setattr(m, "_load_profile_env", lambda: calls.append("env") or True)
        monkeypatch.setattr(m, "_build_server", lambda: calls.append("build") or FakeServer())

        rc = m.main([])

        assert rc == 0
        assert calls == ["env", "build", "run"]

    def test_main_returns_2_when_mcp_unavailable(self, monkeypatch):
        import agent.transports.claude_cli_tools_mcp_server as m

        monkeypatch.setattr(m, "_load_profile_env", lambda: True)

        def boom_build(*_a, **_kw):
            raise ImportError("mcp not installed")

        monkeypatch.setattr(m, "_build_server", boom_build)
        assert m.main(["--verbose"]) == 2

    def test_main_handles_keyboard_interrupt(self, monkeypatch):
        import agent.transports.claude_cli_tools_mcp_server as m

        monkeypatch.setattr(m, "_load_profile_env", lambda: True)

        class FakeServer:
            def run(self):
                raise KeyboardInterrupt()

        monkeypatch.setattr(m, "_build_server", lambda: FakeServer())
        assert m.main([]) == 0

    def test_main_returns_1_on_runtime_error(self, monkeypatch):
        import agent.transports.claude_cli_tools_mcp_server as m

        monkeypatch.setattr(m, "_load_profile_env", lambda: True)

        class CrashingServer:
            def run(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(m, "_build_server", lambda: CrashingServer())
        assert m.main([]) == 1
