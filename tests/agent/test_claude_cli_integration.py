from types import SimpleNamespace

from agent.chat_completion_helpers import (
    build_api_kwargs,
    interruptible_api_call,
    interruptible_streaming_api_call,
)
from agent.transports import get_transport


def test_build_api_kwargs_for_claude_cli_is_text_only_and_omits_tools():
    agent = SimpleNamespace(
        api_mode="claude_cli",
        model="claude-sonnet-4-6",
        tools=[{"type": "function", "function": {"name": "terminal"}}],
        max_tokens=None,
        request_overrides={},
    )

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    assert kwargs == {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_interruptible_api_call_routes_claude_cli_to_streaming_adapter(monkeypatch):
    calls = []
    expected_response = SimpleNamespace(choices=[])

    def fake_run(api_kwargs, *, agent=None, **kwargs):
        calls.append((api_kwargs, agent, kwargs))
        return expected_response

    monkeypatch.setattr("agent.claude_cli_adapter.run_claude_cli_streaming", fake_run)
    agent = SimpleNamespace(
        api_mode="claude_cli",
        _touch_activity=lambda message: None,
        _interrupt_requested=False,
    )
    api_kwargs = {"model": "claude-sonnet-4-6", "messages": []}

    response = interruptible_api_call(agent, api_kwargs)

    assert response is expected_response
    assert calls == [(api_kwargs, agent, {})]


def test_interruptible_streaming_api_call_routes_claude_cli_to_streaming_adapter(monkeypatch):
    calls = []
    stream_chunks = []
    first_delta = []
    expected_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="streamed ok"))]
    )

    def fake_run(api_kwargs, *, agent=None, on_text_delta=None, on_first_delta=None, **kwargs):
        calls.append((api_kwargs, agent, on_text_delta, on_first_delta))
        if on_first_delta:
            on_first_delta()
        if on_text_delta:
            on_text_delta("streamed ok")
        return expected_response

    monkeypatch.setattr("agent.claude_cli_adapter.run_claude_cli_streaming", fake_run)
    agent = SimpleNamespace(
        api_mode="claude_cli",
        _touch_activity=lambda message: None,
        _interrupt_requested=False,
        _fire_stream_delta=stream_chunks.append,
        _has_stream_consumers=lambda: True,
    )
    api_kwargs = {"model": "claude-sonnet-4-6", "messages": []}

    response = interruptible_streaming_api_call(
        agent,
        api_kwargs,
        on_first_delta=lambda: first_delta.append(True),
    )

    assert response is expected_response
    assert calls == [(api_kwargs, agent, stream_chunks.append, calls[0][3])]
    assert first_delta == [True]
    assert stream_chunks == ["streamed ok"]


def test_claude_cli_transport_registered_and_normalizes_adapter_response():
    transport = get_transport("claude_cli")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ok",
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    assert transport is not None
    assert transport.validate_response(response) is True
    normalized = transport.normalize_response(response)
    assert normalized.content == "ok"
    assert normalized.finish_reason == "stop"
    assert normalized.usage.prompt_tokens == 1
    assert normalized.usage.completion_tokens == 2
