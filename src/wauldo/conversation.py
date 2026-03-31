"""Conversation helper — manages chat history automatically."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, List, Optional

from .http_types import ChatMessage, ChatRequest

if TYPE_CHECKING:
    from .http_client import HttpClient

logger = logging.getLogger("wauldo")


class Conversation:
    """Stateful conversation that tracks message history.

    Thread-safe: all mutations are guarded by an internal lock.

    Usage::

        client = HttpClient()
        conv = client.conversation(system="You are a helpful assistant.")
        reply = conv.say("Hello!")
        follow_up = conv.say("Tell me more.")
    """

    def __init__(
        self,
        client: HttpClient,
        system: Optional[str] = None,
        model: str = "default",
    ) -> None:
        self._client = client
        self._history: List[ChatMessage] = []
        self._model = model
        self._lock = threading.Lock()
        if system:
            self._history.append(ChatMessage.system(system))

    def say(self, message: str) -> str:
        """Send a message and get a reply. History is managed automatically.

        Args:
            message: The user message to send.

        Returns:
            The assistant's reply as a plain string.

        Example::

            conv = client.conversation(system="You are helpful.")
            answer = conv.say("What is Python?")
        """
        with self._lock:
            rollback_len = len(self._history)
            self._history.append(ChatMessage.user(message))
            request = ChatRequest(model=self._model, messages=list(self._history))
            try:
                response = self._client.chat(request)
            except Exception:
                self._history = self._history[:rollback_len]
                raise
            reply = response.choices[0].message.content or ""
            self._history.append(ChatMessage.assistant(reply))
            logger.debug("Conversation turn: %d messages", len(self._history))
            return reply

    @property
    def history(self) -> List[ChatMessage]:
        """Return a copy of the conversation history.

        Returns:
            A new list containing all messages (system, user, assistant).

        Example::

            for msg in conv.history:
                print(f"{msg.role}: {msg.content}")
        """
        with self._lock:
            return list(self._history)

    def clear(self) -> None:
        """Reset the conversation history, keeping the system prompt if set.

        User and assistant messages are removed. The system prompt (if any)
        is preserved so subsequent ``say()`` calls retain the same persona.
        """
        with self._lock:
            system = self._history[0] if self._history and self._history[0].role == "system" else None
            self._history.clear()
            if system is not None:
                self._history.append(system)
            logger.debug("Conversation history cleared (system prompt %s)", "kept" if system else "none")

    def __len__(self) -> int:
        """Return the number of messages in history.

        Returns:
            Count including system, user, and assistant messages.

        Example::

            conv = client.conversation(system="Hi")
            assert len(conv) == 1  # system message
        """
        with self._lock:
            return len(self._history)

    def __repr__(self) -> str:
        with self._lock:
            return f"Conversation(model={self._model!r}, messages={len(self._history)})"
