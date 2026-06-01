# =============================================================
# aria/agent/client.py
# =============================================================
# Thin wrapper around the Ollama API using the OpenAI-compatible
# interface. This is the single point of contact between ARIA
# and any LLM — local or cloud.
#
# Design goals:
#   - One class, one responsibility: send messages, get a reply
#   - Streaming and non-streaming both supported
#   - Configurable base_url so swapping Ollama for Groq or any
#     other OpenAI-compatible provider is a one-line change
#   - All responses come back as a ChatResponse dataclass so
#     the agent layer never has to touch raw API objects
#   - Errors are caught and re-raised as OllamaClientError so
#     callers don't need to know about openai internals
# =============================================================

from __future__ import annotations

import os
from collections.abc import Generator
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, OpenAI

# =============================================================
# CONFIGURATION DEFAULTS
# Override via environment variables or pass directly to the
# OllamaClient constructor.
# =============================================================

DEFAULT_BASE_URL = os.getenv("ARIA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.getenv("ARIA_MODEL", "phi3.5")
DEFAULT_TIMEOUT = float(os.getenv("ARIA_TIMEOUT", "120.0"))  # seconds

# Temperature presets — named so the agent layer expresses
# intent rather than magic numbers.
TEMPERATURE_PRECISE = 0.0  # deterministic: structured JSON outputs
TEMPERATURE_BALANCED = 0.3  # slight variation: suggestions, warnings
TEMPERATURE_CREATIVE = 0.7  # more variation: summaries, narratives


# =============================================================
# RESPONSE DATACLASS
# Wraps the raw API response into a clean, typed object.
# The agent layer works only with ChatResponse — never with
# raw openai completion objects.
# =============================================================


@dataclass
class Message:
    """A single message in a conversation turn."""

    role: str  # 'system' | 'user' | 'assistant'
    content: str

    def to_dict(self) -> dict:
        """Serialize to the dict format the OpenAI API expects."""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """
    The result of a single non-streaming chat completion.

    content    — the model's reply text
    model      — which model produced the reply
    prompt_tokens     — tokens consumed by the input
    completion_tokens — tokens consumed by the output
    total_tokens      — sum of the above
    """

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def usage_summary(self) -> str:
        return (
            f"{self.prompt_tokens} prompt + "
            f"{self.completion_tokens} completion = "
            f"{self.total_tokens} total tokens"
        )


# =============================================================
# EXCEPTIONS
# =============================================================


class OllamaClientError(Exception):
    """
    Raised when the Ollama API call fails for any reason.
    Wraps both connection errors (server not running) and
    API-level errors (bad model name, malformed request).
    """


# =============================================================
# CLIENT
# =============================================================


class OllamaClient:
    """
    OpenAI-compatible client pointed at a local Ollama server.

    Usage — non-streaming:
        client   = OllamaClient()
        response = client.chat([Message("user", "Hello")])
        print(response.content)

    Usage — streaming:
        for chunk in client.stream([Message("user", "Hello")]):
            print(chunk, end="", flush=True)

    Swap to Groq (or any OpenAI-compatible provider) by passing
    a different base_url and api_key:
        client = OllamaClient(
            base_url = "https://api.groq.com/openai/v1",
            api_key  = os.getenv("GROQ_API_KEY"),
            model    = "llama3-8b-8192",
        )
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_key: str = "ollama",  # Ollama ignores this; required by openai SDK
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url

        # The openai SDK handles connection pooling, retries, and
        # serialization. We just redirect it at the local Ollama server.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def chat(
        self,
        messages: list[Message],
        temperature: float = TEMPERATURE_BALANCED,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """
        Send a list of messages and return the complete response.

        Blocks until the model finishes generating. Use stream()
        if you need to display tokens as they arrive.

        Args:
            messages:    Conversation history as Message objects.
            temperature: Controls randomness. Use TEMPERATURE_*
                         constants from this module.
            max_tokens:  Optional hard cap on response length.

        Returns:
            ChatResponse with the model's reply and token usage.

        Raises:
            OllamaClientError: If the server is unreachable or
                               returns an error status.
        """
        try:
            kwargs = dict(
                model=self.model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
            )
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            completion = self._client.chat.completions.create(**kwargs)

            usage = completion.usage
            return ChatResponse(
                content=completion.choices[0].message.content,
                model=completion.model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            )

        except APIConnectionError as e:
            raise OllamaClientError(
                f"Could not reach Ollama at '{self.base_url}'. "
                f"Is 'ollama serve' running?\nDetail: {e}"
            ) from e

        except APIStatusError as e:
            raise OllamaClientError(
                f"Ollama returned an error (HTTP {e.status_code}). "
                f"Is model '{self.model}' pulled?\nDetail: {e.message}"
            ) from e

    def stream(
        self,
        messages: list[Message],
        temperature: float = TEMPERATURE_BALANCED,
    ) -> Generator[str, None, None]:
        """
        Send messages and yield response tokens as they arrive.

        Designed for use in a for-loop:
            for token in client.stream(messages):
                print(token, end="", flush=True)

        Args:
            messages:    Conversation history as Message objects.
            temperature: Controls randomness.

        Yields:
            Individual token strings as the model generates them.

        Raises:
            OllamaClientError: If the server is unreachable or
                               returns an error status.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

        except APIConnectionError as e:
            raise OllamaClientError(
                f"Could not reach Ollama at '{self.base_url}'. "
                f"Is 'ollama serve' running?\nDetail: {e}"
            ) from e

        except APIStatusError as e:
            raise OllamaClientError(
                f"Ollama returned an error (HTTP {e.status_code}). "
                f"Is model '{self.model}' pulled?\nDetail: {e.message}"
            ) from e

    # ----------------------------------------------------------
    # Convenience constructors
    # ----------------------------------------------------------

    @classmethod
    def from_groq(cls, api_key: str, model: str = "llama3-8b-8192") -> OllamaClient:
        """
        Return a client pointed at Groq's free inference API.
        Groq uses the same OpenAI-compatible interface, so all
        agent code works unchanged — just swap the client.

        Usage:
            client = OllamaClient.from_groq(os.getenv("GROQ_API_KEY"))
        """
        return cls(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            model=model,
        )

    # ----------------------------------------------------------
    # Helpers used by the agent layer
    # ----------------------------------------------------------

    @staticmethod
    def system(content: str) -> Message:
        """Convenience constructor for a system message."""
        return Message(role="system", content=content)

    @staticmethod
    def user(content: str) -> Message:
        """Convenience constructor for a user message."""
        return Message(role="user", content=content)

    @staticmethod
    def assistant(content: str) -> Message:
        """Convenience constructor for an assistant message."""
        return Message(role="assistant", content=content)

    def __repr__(self) -> str:
        return f"OllamaClient(model='{self.model}', base_url='{self.base_url}')"
