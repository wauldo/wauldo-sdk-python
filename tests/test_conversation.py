"""Tests for Conversation helper."""

from wauldo.mock_client import MockHttpClient


class TestConversation:
    def test_conversation_from_mock_client(self):
        mock = MockHttpClient(chat_response="Hello!")
        conv = mock.conversation(system="You are helpful")
        assert len(conv) == 1  # system message

    def test_conversation_clear_preserves_system(self):
        mock = MockHttpClient(chat_response="Hello!")
        conv = mock.conversation(system="You are helpful")
        assert len(conv) == 1
        conv.clear()
        assert len(conv) == 1  # system prompt preserved
        assert conv.history[0].role == "system"

    def test_conversation_clear_without_system(self):
        mock = MockHttpClient(chat_response="Hello!")
        conv = mock.conversation()
        assert len(conv) == 0
        conv.clear()
        assert len(conv) == 0

    def test_conversation_say(self):
        mock = MockHttpClient(chat_response="42")
        conv = mock.conversation(system="Be concise")
        reply = conv.say("What is the answer?")
        assert reply == "42"
        assert len(conv) == 3  # system + user + assistant

    def test_conversation_multi_turn(self):
        mock = MockHttpClient(chat_response="reply")
        conv = mock.conversation()
        conv.say("first")
        conv.say("second")
        assert len(conv) == 4  # 2 user + 2 assistant

    def test_conversation_repr(self):
        mock = MockHttpClient()
        conv = mock.conversation(system="sys", model="test-model")
        r = repr(conv)
        assert "test-model" in r
        assert "1" in r


class TestMockHttpClientCompleteness:
    def test_mock_has_conversation(self):
        mock = MockHttpClient()
        conv = mock.conversation()
        assert conv is not None

    def test_mock_has_rag_ask(self):
        mock = MockHttpClient(chat_response="answer")
        result = mock.rag_ask("question", "some text")
        assert isinstance(result, str)

    def test_mock_rag_ask_returns_answer(self):
        mock = MockHttpClient(chat_response="the answer")
        result = mock.rag_ask("what?", "context text")
        assert result == "the answer"
