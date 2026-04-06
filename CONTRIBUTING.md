# Contributing to Wauldo Python SDK

Thanks for contributing! Here's how to get started quickly.

## Setup (< 5 minutes)

```bash
git clone https://github.com/YOUR_USERNAME/wauldo-sdk-python
cd wauldo-sdk-python
pip install -e ".[dev]"
pytest
```

All tests pass without an API key — they use `MockHttpClient` internally.

## Running tests

```bash
pytest                    # all tests
pytest tests/test_http_mock.py  # specific file
pytest -x                 # stop on first failure
```

For async tests, `pytest-asyncio` is included in dev deps. Just write `async def test_...` and it works.

## Testing without a server

You don't need a Wauldo API key to develop. Use `MockHttpClient` for everything:

```python
from wauldo import MockHttpClient

client = MockHttpClient(chat_response="test answer")
result = client.rag_query("question?")
assert result.answer == "test answer"
```

If you're adding a new feature, write tests against the mock client first.

## Code style

- Python 3.9+ (no walrus operator patterns that break 3.9)
- Type hints on all public methods
- Formatting: `ruff` with 100 char line length (config is in `pyproject.toml`)
- 4 spaces, LF line endings

Run the linter before pushing:

```bash
ruff check src/ tests/
mypy src/
```

## Making a PR

1. One PR per feature/fix, keep it small
2. Add tests for new functionality
3. Update README if you're adding something user-facing
4. Make sure `pytest` and `ruff check` pass

### Commit messages

Write plain English, describe what you did:

```
add retry logic to async client
fix timeout handling in streaming
update mock client to support file uploads
```

### What gets merged quickly

- Has tests
- Doesn't break existing tests
- Small and focused (< 300 lines is ideal)
- Touches one concern at a time

## Good first issues

Check [issues labeled `good first issue`](https://github.com/wauldo/wauldo-sdk-python/labels/good%20first%20issue). Adding methods to `MockHttpClient` is a great place to start since you can write and test everything offline.

## Questions?

Open an issue or comment on an existing one. We're happy to help.
