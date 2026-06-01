# =============================================================
# aria/agent/test_client.py
# =============================================================
# Pytest test suite for agent/client.py.
#
# Usage:
#   cd aria
#   pytest agent/test_client.py -v
#
# Ollama does not need to be running. All network calls are
# intercepted by mocking the openai SDK at the boundary —
# OllamaClient._client.chat.completions.create — so tests are
# fast, deterministic, and CI-safe.
# =============================================================

from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, APIStatusError

from agent.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    TEMPERATURE_BALANCED,
    TEMPERATURE_CREATIVE,
    TEMPERATURE_PRECISE,
    ChatResponse,
    Message,
    OllamaClient,
    OllamaClientError,
)

# =============================================================
# HELPERS — build realistic mock API responses
# =============================================================


def make_completion(
    content: str, model: str = "phi3.5", prompt_tokens: int = 10, completion_tokens: int = 20
) -> MagicMock:
    """
    Build a mock that looks like an openai ChatCompletion object.
    Mirrors the structure the real SDK returns so our assertions
    reflect what the code actually has to handle.
    """
    mock = MagicMock()
    mock.choices[0].message.content = content
    mock.model = model
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    mock.usage.total_tokens = prompt_tokens + completion_tokens
    return mock


def make_stream_chunks(tokens: list[str]) -> list[MagicMock]:
    """
    Build a list of mock stream chunks, one per token.
    Each chunk has the same structure the openai SDK yields
    during a streaming completion.
    """
    chunks = []
    for token in tokens:
        chunk = MagicMock()
        chunk.choices[0].delta.content = token
        chunks.append(chunk)

    # Final chunk has no content (metadata-only), as the real API does
    empty = MagicMock()
    empty.choices[0].delta.content = None
    chunks.append(empty)

    return chunks


@pytest.fixture
def client() -> OllamaClient:
    """A default OllamaClient instance for every test."""
    return OllamaClient()


@pytest.fixture
def mock_create(client):
    """
    Patch OllamaClient._client.chat.completions.create for the
    duration of a test. Yields the mock so tests can set its
    return_value or side_effect.
    """
    with patch.object(client._client.chat.completions, "create") as mock:
        yield mock


# =============================================================
# MESSAGE DATACLASS
# =============================================================


class TestMessage:
    def test_role_and_content_stored(self):
        m = Message(role="user", content="Hello")
        assert m.role == "user"
        assert m.content == "Hello"

    def test_to_dict_returns_correct_keys(self):
        m = Message(role="system", content="You are an assistant.")
        d = m.to_dict()
        assert d == {"role": "system", "content": "You are an assistant."}

    def test_all_roles_accepted(self):
        for role in ("system", "user", "assistant"):
            m = Message(role=role, content="text")
            assert m.to_dict()["role"] == role


# =============================================================
# CHATRESPONSE DATACLASS
# =============================================================


class TestChatResponse:
    def test_fields_stored_correctly(self):
        r = ChatResponse(
            content="Hello!",
            model="phi3.5",
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
        )
        assert r.content == "Hello!"
        assert r.model == "phi3.5"
        assert r.prompt_tokens == 5
        assert r.completion_tokens == 10
        assert r.total_tokens == 15

    def test_usage_summary_format(self):
        r = ChatResponse(
            content="x", model="phi3.5", prompt_tokens=5, completion_tokens=10, total_tokens=15
        )
        assert "5 prompt" in r.usage_summary
        assert "10 completion" in r.usage_summary
        assert "15 total" in r.usage_summary

    def test_default_token_counts_are_zero(self):
        r = ChatResponse(content="x", model="phi3.5")
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.total_tokens == 0


# =============================================================
# CLIENT CONSTRUCTION
# =============================================================


class TestOllamaClientConstruction:
    def test_default_model(self):
        c = OllamaClient()
        assert c.model == DEFAULT_MODEL

    def test_default_base_url(self):
        c = OllamaClient()
        assert c.base_url == DEFAULT_BASE_URL

    def test_custom_model(self):
        c = OllamaClient(model="llama3.2:3b")
        assert c.model == "llama3.2:3b"

    def test_custom_base_url(self):
        c = OllamaClient(base_url="http://localhost:9999/v1")
        assert c.base_url == "http://localhost:9999/v1"

    def test_repr_contains_model_and_url(self):
        c = OllamaClient(model="phi3.5", base_url="http://localhost:11434/v1")
        r = repr(c)
        assert "phi3.5" in r
        assert "http://localhost:11434/v1" in r

    def test_from_groq_sets_correct_base_url(self):
        c = OllamaClient.from_groq(api_key="test-key")
        assert "groq.com" in c.base_url

    def test_from_groq_default_model(self):
        c = OllamaClient.from_groq(api_key="test-key")
        assert c.model == "llama3-8b-8192"

    def test_from_groq_custom_model(self):
        c = OllamaClient.from_groq(api_key="test-key", model="llama3-70b-8192")
        assert c.model == "llama3-70b-8192"


# =============================================================
# CONVENIENCE MESSAGE CONSTRUCTORS
# =============================================================


class TestMessageConstructors:
    def test_system_message(self):
        m = OllamaClient.system("You are helpful.")
        assert m.role == "system"
        assert m.content == "You are helpful."

    def test_user_message(self):
        m = OllamaClient.user("What time is it?")
        assert m.role == "user"
        assert m.content == "What time is it?"

    def test_assistant_message(self):
        m = OllamaClient.assistant("It is noon.")
        assert m.role == "assistant"
        assert m.content == "It is noon."


# =============================================================
# TEMPERATURE CONSTANTS
# =============================================================


class TestTemperatureConstants:
    def test_precise_is_zero(self):
        assert TEMPERATURE_PRECISE == 0.0

    def test_balanced_between_precise_and_creative(self):
        assert TEMPERATURE_PRECISE < TEMPERATURE_BALANCED < TEMPERATURE_CREATIVE

    def test_creative_at_most_one(self):
        assert TEMPERATURE_CREATIVE <= 1.0


# =============================================================
# CHAT — happy path
# =============================================================


class TestChat:
    def test_returns_chat_response(self, client, mock_create):
        mock_create.return_value = make_completion("Hello!")
        response = client.chat([OllamaClient.user("Hi")])
        assert isinstance(response, ChatResponse)

    def test_response_content_is_correct(self, client, mock_create):
        mock_create.return_value = make_completion("Task created.")
        response = client.chat([OllamaClient.user("Create a task")])
        assert response.content == "Task created."

    def test_response_model_is_correct(self, client, mock_create):
        mock_create.return_value = make_completion("Hi", model="phi3.5")
        response = client.chat([OllamaClient.user("Hi")])
        assert response.model == "phi3.5"

    def test_token_counts_populated(self, client, mock_create):
        mock_create.return_value = make_completion("Hi", prompt_tokens=8, completion_tokens=4)
        response = client.chat([OllamaClient.user("Hi")])
        assert response.prompt_tokens == 8
        assert response.completion_tokens == 4
        assert response.total_tokens == 12

    def test_messages_serialized_as_dicts(self, client, mock_create):
        """The SDK must receive plain dicts, not Message objects."""
        mock_create.return_value = make_completion("ok")
        messages = [
            OllamaClient.system("You are helpful."),
            OllamaClient.user("Hello"),
        ]
        client.chat(messages)
        call_kwargs = mock_create.call_args.kwargs
        for m in call_kwargs["messages"]:
            assert isinstance(m, dict)
            assert "role" in m
            assert "content" in m

    def test_default_temperature_is_balanced(self, client, mock_create):
        mock_create.return_value = make_completion("ok")
        client.chat([OllamaClient.user("Hi")])
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["temperature"] == TEMPERATURE_BALANCED

    def test_custom_temperature_is_passed_through(self, client, mock_create):
        mock_create.return_value = make_completion("ok")
        client.chat([OllamaClient.user("Hi")], temperature=TEMPERATURE_PRECISE)
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["temperature"] == TEMPERATURE_PRECISE

    def test_max_tokens_not_sent_when_none(self, client, mock_create):
        mock_create.return_value = make_completion("ok")
        client.chat([OllamaClient.user("Hi")], max_tokens=None)
        call_kwargs = mock_create.call_args.kwargs
        assert "max_tokens" not in call_kwargs

    def test_max_tokens_sent_when_provided(self, client, mock_create):
        mock_create.return_value = make_completion("ok")
        client.chat([OllamaClient.user("Hi")], max_tokens=256)
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 256

    def test_correct_model_name_sent(self, client, mock_create):
        mock_create.return_value = make_completion("ok")
        client.chat([OllamaClient.user("Hi")])
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == client.model

    def test_multi_turn_conversation(self, client, mock_create):
        """All messages in history must be forwarded to the API."""
        mock_create.return_value = make_completion("Got it.")
        messages = [
            OllamaClient.system("You manage tasks."),
            OllamaClient.user("Add a task."),
            OllamaClient.assistant("What should I call it?"),
            OllamaClient.user("Call it 'Write tests'."),
        ]
        client.chat(messages)
        call_kwargs = mock_create.call_args.kwargs
        assert len(call_kwargs["messages"]) == 4


# =============================================================
# CHAT — error handling
# =============================================================


class TestChatErrors:
    def test_connection_error_raises_ollama_client_error(self, client, mock_create):
        mock_create.side_effect = APIConnectionError(request=MagicMock())
        with pytest.raises(OllamaClientError) as exc_info:
            client.chat([OllamaClient.user("Hi")])
        assert "ollama serve" in str(exc_info.value).lower()

    def test_api_status_error_raises_ollama_client_error(self, client, mock_create):
        mock_create.side_effect = APIStatusError(
            message="model not found",
            response=MagicMock(status_code=404),
            body={},
        )
        with pytest.raises(OllamaClientError) as exc_info:
            client.chat([OllamaClient.user("Hi")])
        assert "404" in str(exc_info.value)

    def test_ollama_client_error_is_exception(self):
        assert issubclass(OllamaClientError, Exception)


# =============================================================
# STREAM — happy path
# =============================================================


class TestStream:
    def test_yields_tokens_as_strings(self, client, mock_create):
        mock_create.return_value = iter(make_stream_chunks(["Hello", " world"]))
        tokens = list(client.stream([OllamaClient.user("Hi")]))
        assert tokens == ["Hello", " world"]

    def test_empty_delta_chunks_are_skipped(self, client, mock_create):
        """
        The final chunk has content=None. It must not appear in
        the yielded tokens.
        """
        mock_create.return_value = iter(make_stream_chunks(["Only this"]))
        tokens = list(client.stream([OllamaClient.user("Hi")]))
        assert None not in tokens

    def test_stream_flag_sent_to_api(self, client, mock_create):
        mock_create.return_value = iter(make_stream_chunks(["ok"]))
        list(client.stream([OllamaClient.user("Hi")]))
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["stream"] is True

    def test_concatenated_stream_matches_full_response(self, client, mock_create):
        tokens = ["Task", " created", " successfully", "."]
        mock_create.return_value = iter(make_stream_chunks(tokens))
        result = "".join(client.stream([OllamaClient.user("Create task")]))
        assert result == "Task created successfully."

    def test_stream_uses_correct_model(self, client, mock_create):
        mock_create.return_value = iter(make_stream_chunks(["ok"]))
        list(client.stream([OllamaClient.user("Hi")]))
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == client.model

    def test_stream_default_temperature_is_balanced(self, client, mock_create):
        mock_create.return_value = iter(make_stream_chunks(["ok"]))
        list(client.stream([OllamaClient.user("Hi")]))
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["temperature"] == TEMPERATURE_BALANCED


# =============================================================
# STREAM — error handling
# =============================================================


class TestStreamErrors:
    def test_connection_error_raises_ollama_client_error(self, client, mock_create):
        mock_create.side_effect = APIConnectionError(request=MagicMock())
        with pytest.raises(OllamaClientError):
            list(client.stream([OllamaClient.user("Hi")]))

    def test_api_status_error_raises_ollama_client_error(self, client, mock_create):
        mock_create.side_effect = APIStatusError(
            message="not found",
            response=MagicMock(status_code=404),
            body={},
        )
        with pytest.raises(OllamaClientError):
            list(client.stream([OllamaClient.user("Hi")]))
