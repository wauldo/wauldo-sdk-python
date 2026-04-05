# Wauldo Python SDK

[![PyPI](https://img.shields.io/pypi/v/wauldo.svg)](https://pypi.org/project/wauldo/)
[![Downloads](https://img.shields.io/pypi/dm/wauldo.svg)](https://pypi.org/project/wauldo/)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

> **Verified AI answers from your documents — or no answer at all.**

`0% hallucination` | `83% accuracy` | `61 eval tasks` | `14 LLMs tested`

```bash
pip install wauldo
```

```python
from wauldo import HttpClient

client = HttpClient(base_url="https://api.wauldo.com", api_key="YOUR_API_KEY")

# Upload a document
client.rag_upload(content="Our refund policy allows returns within 14 days...", filename="policy.txt")

# Ask a question — answer is verified against the source
result = client.rag_query("What is the refund policy?")
print(result.answer)
print(result.sources)
```

```
Output:
Answer: Returns are accepted within 14 days of purchase.
Sources: policy.txt — "Our refund policy allows returns within 14 days"
```

No verification = no answer. If it can't be grounded, Wauldo returns "insufficient evidence" instead of guessing.

---

## Why Wauldo?

Most RAG APIs: **retrieve → generate → hope it's correct.**

Wauldo: **retrieve → extract facts → generate → verify → return or refuse.**

| What | How |
|------|-----|
| **Zero hallucinations** | Every answer verified against source documents |
| **Fact-Check API** | Verify any claim against any source (3 modes) |
| **Citation Verify** | Detect phantom citations and uncited claims |
| **PDF & DOCX Upload** | Server-side extraction with quality scoring |
| **Smart routing** | Auto-selects cheapest model that meets quality |
| **OpenAI-compatible** | Swap your `base_url`, keep your existing code |
| **Async support** | Full async/await with `pip install wauldo[async]` |

---

## Examples

### Upload a document and ask questions

```python
from wauldo import HttpClient

client = HttpClient(base_url="https://api.wauldo.com", api_key="YOUR_API_KEY")

# Upload
upload = client.rag_upload(content="Contract text here...", filename="contract.txt")
print(f"Indexed {upload.chunks_count} chunks")

# Query — answer is verified against the source
result = client.rag_query("What are the payment terms?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.get_confidence():.0%}")
print(f"Grounded: {result.audit.grounded}")
```

### Upload a PDF directly

```python
result = client.upload_file("contract.pdf", title="Q3 Contract")
print(f"Extracted {result.chunks_count} chunks, quality: {result.quality_label}")
```

### Fact-check an LLM answer

```python
result = client.fact_check(
    text="Returns are accepted within 60 days.",
    source_context="Our policy allows returns within 14 days.",
    mode="lexical",
)
print(result.verdict)  # "rejected"
print(result.action)   # "block"
print(result.claims[0].reason)  # "numerical_mismatch"
```

### Verify citations

```python
result = client.verify_citation(
    text="The policy covers damage [Source: Manual]. Warranty is unlimited.",
    sources=[{"name": "Manual", "content": "Coverage for accidental damage only."}],
)
print(result.citation_ratio)     # 0.5
print(result.uncited_sentences)  # ["Warranty is unlimited."]
print(result.phantom_count)      # 0
```

### Chat (OpenAI-compatible)

```python
reply = client.chat_simple("auto", "Explain Python decorators")
print(reply)
```

### Streaming

```python
from wauldo import ChatRequest, HttpChatMessage

request = ChatRequest(model="auto", messages=[HttpChatMessage.user("Hello!")])
for chunk in client.chat_stream(request):
    print(chunk, end="", flush=True)
```

### Async Client

```python
from wauldo import AsyncHttpClient

async with AsyncHttpClient(base_url="https://api.wauldo.com", api_key="YOUR_API_KEY") as client:
    result = await client.rag_query("What are the payment terms?")
    print(result.answer)

    async for chunk in client.chat_stream(request):
        print(chunk, end="", flush=True)
```

Install with `pip install wauldo[async]`. All sync methods have async equivalents.

*Async support contributed by [@qorexdev](https://github.com/qorexdev).*

---

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

---

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

Free tier (300 req/month): [RapidAPI](https://rapidapi.com/binnewzzin/api/smart-rag-api)

---

## Links

- [Website](https://wauldo.com) | [Docs](https://wauldo.com/docs) | [Demo](https://wauldo.com/demo) | [Benchmarks](https://dev.to/wauldo/how-we-achieved-0-hallucination-rate-in-our-rag-api-with-benchmarks-4g54)

## Contributing

Found a bug? Have a feature request? [Open an issue](https://github.com/wauldo/wauldo-sdk-python/issues). PRs welcome — check the [good first issues](https://github.com/wauldo/wauldo-sdk-python/labels/good%20first%20issue).

## Contributors

- [@qorexdev](https://github.com/qorexdev) — async client

## License

MIT — see [LICENSE](./LICENSE)
