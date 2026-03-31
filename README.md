# Wauldo Python SDK

[![PyPI](https://img.shields.io/pypi/v/wauldo.svg)](https://pypi.org/project/wauldo/)
[![Downloads](https://img.shields.io/pypi/dm/wauldo.svg)](https://pypi.org/project/wauldo/)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

> **Verified AI answers from your documents.** Every response includes source citations, confidence scores, and an audit trail — or we don't answer at all.

Official Python SDK for the [Wauldo API](https://wauldo.com) — the AI inference layer with smart model routing and zero hallucinations.

## Why Wauldo?

- **Zero hallucinations** — every answer is verified against source documents
- **Smart model routing** — auto-selects the cheapest model that meets quality (save 40-80% on AI costs)
- **One API, 7+ providers** — OpenAI, Anthropic, Google, Qwen, Meta, Mistral, DeepSeek with automatic fallback
- **OpenAI-compatible** — swap your `base_url`, keep your existing code
- **Full audit trail** — confidence score, grounded status, model used, latency on every response

## Quick Start

```python
from wauldo import HttpClient

client = HttpClient(base_url="https://api.wauldo.com", api_key="YOUR_API_KEY")

reply = client.chat_simple("auto", "What is Python?")
print(reply)
```

## Installation

```bash
pip install wauldo
```

**Requirements:** Python 3.9+

## Features

### Chat Completions

```python
from wauldo import HttpClient, ChatRequest, HttpChatMessage

client = HttpClient(base_url="https://api.wauldo.com", api_key="YOUR_API_KEY")

request = ChatRequest(
    model="auto",
    messages=[
        HttpChatMessage.system("You are a helpful assistant."),
        HttpChatMessage.user("Explain Python decorators"),
    ],
)
response = client.chat(request)
print(response.choices[0].message.content)
```

### RAG — Upload & Query

```python
# Upload a document
upload = client.rag_upload(content="Contract text here...", filename="contract.txt")
print(f"Indexed {upload.chunks_count} chunks")

# Query with verified answer
result = client.rag_query("What are the payment terms?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.get_confidence():.0%}")
print(f"Grounded: {result.audit.grounded}")
for source in result.sources:
    print(f"  Source ({source.score:.0%}): {source.content[:80]}")
```

### Streaming (SSE)

```python
from wauldo import ChatRequest, HttpChatMessage

request = ChatRequest(model="auto", messages=[HttpChatMessage.user("Hello!")])
for chunk in client.chat_stream(request):
    print(chunk, end="", flush=True)
```

### Conversation Helper

```python
conv = client.conversation(system="You are an expert on Python.", model="auto")
reply = conv.say("What are list comprehensions?")
follow_up = conv.say("Give me a nested example")
```

## Error Handling

```python
from wauldo import WauldoError, ServerError, AgentTimeoutError

try:
    response = client.chat(ChatRequest.quick("auto", "Hello"))
except ServerError as e:
    print(f"Server error: {e}")
except AgentTimeoutError:
    print("Request timed out")
except WauldoError as e:
    print(f"SDK error: {e}")
```

## RapidAPI

```python
client = HttpClient(
    base_url="https://api.wauldo.com",
    headers={
        "X-RapidAPI-Key": "YOUR_RAPIDAPI_KEY",
        "X-RapidAPI-Host": "smart-rag-api.p.rapidapi.com",
    },
)
```

Get your free API key (300 req/month): [RapidAPI](https://rapidapi.com/binnewzzin/api/smart-rag-api)

## Links

- [Website](https://wauldo.com)
- [Documentation](https://wauldo.com/docs)
- [Live Demo](https://api.wauldo.com/demo)
- [Cost Calculator](https://wauldo.com/calculator)
- [Status](https://wauldo.com/status)

## Contributing

Found a bug? Have a feature request? [Open an issue](https://github.com/wauldo/wauldo-sdk-python/issues).

## License

MIT — see [LICENSE](./LICENSE)
