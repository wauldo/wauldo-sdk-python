"""Tests for SDK exceptions."""

from wauldo.exceptions import (
    WauldoError,
    AgentConnectionError,
    AgentTimeoutError,
    ServerError,
    ValidationError,
    ToolNotFoundError,
)


class TestWauldoError:
    def test_basic_error(self):
        error = WauldoError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.code is None

    def test_error_with_code(self):
        error = WauldoError("Test error", code=123)
        assert str(error) == "[123] Test error"
        assert error.code == 123

    def test_error_with_data(self):
        error = WauldoError("Test", data={"key": "value"})
        assert error.data == {"key": "value"}


class TestConnectionError:
    def test_default_message(self):
        error = AgentConnectionError()
        assert "connect" in str(error).lower()
        assert error.code == -32000

    def test_custom_message(self):
        error = AgentConnectionError("Custom message")
        assert str(error) == "[-32000] Custom message"

    def test_is_subclass_of_base(self):
        error = AgentConnectionError("test")
        assert isinstance(error, WauldoError)

    def test_does_not_shadow_builtin(self):
        """Ensure AgentConnectionError is NOT the Python builtin ConnectionError."""
        assert AgentConnectionError is not builtins_ConnectionError()


class TestServerError:
    def test_creation(self):
        error = ServerError("Server failed", code=-32603)
        assert error.message == "Server failed"
        assert error.code == -32603


class TestValidationError:
    def test_with_field(self):
        error = ValidationError("Invalid value", field="name")
        assert error.field == "name"
        assert error.code == -32602

    def test_without_field(self):
        error = ValidationError("Invalid")
        assert error.field is None


class TestTimeoutError:
    def test_default(self):
        error = AgentTimeoutError()
        assert "timed out" in str(error).lower()
        assert error.code == -32001

    def test_with_timeout_value(self):
        error = AgentTimeoutError("Timed out", timeout=30.0)
        assert error.timeout == 30.0

    def test_is_subclass_of_base(self):
        error = AgentTimeoutError("test")
        assert isinstance(error, WauldoError)

    def test_does_not_shadow_builtin(self):
        """Ensure AgentTimeoutError is NOT the Python builtin TimeoutError."""
        assert AgentTimeoutError is not builtins_TimeoutError()


class TestToolNotFoundError:
    def test_creation(self):
        error = ToolNotFoundError("unknown_tool")
        assert error.tool_name == "unknown_tool"
        assert "unknown_tool" in str(error)
        assert error.code == -32601


def builtins_ConnectionError():
    """Return the real Python builtin ConnectionError."""
    import builtins
    return builtins.ConnectionError


def builtins_TimeoutError():
    """Return the real Python builtin TimeoutError."""
    import builtins
    return builtins.TimeoutError
