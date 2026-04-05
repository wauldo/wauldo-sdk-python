<h1 align="center">Wauldo Python SDK</h1>

<p align="center">
  <strong>Verified AI answers from your documents — or no answer at all.</strong>
</p>

<p align="center">
  Most RAG APIs guess. Wauldo verifies.
</p>

<p align="center">
  <b>0% hallucination</b> &nbsp;|&nbsp; 83% accuracy &nbsp;|&nbsp; 61 eval tasks &nbsp;|&nbsp; 14 LLMs tested
</p>

<p align="center">
  <a href="https://pypi.org/project/wauldo/"><img src="https://img.shields.io/pypi/v/wauldo.svg" alt="PyPI" /></a>&nbsp;
  <a href="https://pypi.org/project/wauldo/"><img src="https://img.shields.io/pypi/dm/wauldo.svg" alt="Downloads" /></a>&nbsp;
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python" />&nbsp;
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT" />
</p>

<p align="center">
  <a href="https://wauldo.com/demo">Demo</a> &bull;
  <a href="https://wauldo.com/docs">Docs</a> &bull;
  <a href="https://rapidapi.com/binnewzzin/api/smart-rag-api">Free API Key</a> &bull;
  <a href="https://dev.to/wauldo/how-we-achieved-0-hallucination-rate-in-our-rag-api-with-benchmarks-4g54">Benchmarks</a>
</p>

---

## Quickstart (30 seconds)

```bash
pip install wauldo
```

```python
from wauldo import HttpClient

client = HttpClient(base_url="https://api.wauldo.com", api_key="YOUR_API_KEY")

# Upload a document
client.rag_upload(content="Our refund policy allows returns within 60 days...", filename="policy.txt")

# Ask a question — answer is verified against the source
result = client.rag_query("What is the refund policy?")
print(result.answer)
print(result.sources)
```

```
Output:
Answer: Returns are accepted within 60 days of purchase.
Sources: policy.txt — "Our refund policy allows returns within 60 days"
Grounded: true | Confidence: 0.92
```

[Try the demo](https://wauldo.com/demo) | [Get a free API key](https://rapidapi.com/binnewzzin/api/smart-rag-api)

---

## Why Wauldo (and not standard RAG)

**Typical RAG pipeline**

```
retrieve → generate → hope it's correct
```

**Wauldo pipeline**

```
retrieve → extract facts → generate → verify → return or refuse
```

If the answer can't be verified, it returns **"insufficient evidence"** instead of guessing.

### See the difference

```
Document: "Refunds are processed within 60 days"

Typical RAG:  "Refunds are processed within 30 days"     ← wrong
Wauldo:       "Refunds are processed within 60 days"     ← verified
              or "insufficient evidence" if unclear       ← safe
```

---

## Examples

### Upload a PDF and ask questions

```python
# Upload — text extraction + quality scoring happens server-side
result = client.upload_file("contract.pdf", title="Q3 Contract")
print(f"Extracted {result.chunks_count} chunks, quality: {result.quality_label}")

# Query
result = client.rag_query("What are the payment terms?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.get_confidence():.0%}")
print(f"Grounded: {result.audit.grounded}")
```

### Fact-check any LLM output

```python
result = client.fact_check(
    text="Returns are accepted within 60 days.",
    source_context="Our policy allows returns within 14 days.",
    mode="lexical",
)
print(result.verdict)           # "rejected"
print(result.action)            # "block"
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
```

### Chat (OpenAI-compatible)

```python
reply = client.chat_simple("auto", "Explain Python decorators")
print(reply)
```

### Streaming

**Sync — print tokens as they arrive:**

```python
import sys
from wauldo import ChatRequest, HttpChatMessage

request = ChatRequest(
    model="auto",
    messages=[
        HttpChatMessage.system("You are a helpful assistant."),
        HttpChatMessage.user("Explain Python decorators"),
    ],
)

for chunk in client.chat_stream(request):
    sys.stdout.write(chunk)
    sys.stdout.flush()
print()
```

**Async streaming** (requires `pip install wauldo[async]`):

```python
import asyncio
from wauldo import AsyncHttpClient, ChatRequest, HttpChatMessage

async def main():
    async with AsyncHttpClient(base_url="https://api.wauldo.com", api_key="YOUR_API_KEY") as client:
        req = ChatRequest.quick("auto", "How does HTTP/2 multiplexing work?")
        async for token in client.chat_stream(req):
            print(token, end="", flush=True)
        print()

asyncio.run(main())
```

**RAG query with streaming answer:**

```python
client.rag_upload(content="Our SLA guarantees 99.9% uptime...", filename="sla.txt")

req = ChatRequest(
    model="auto",
    messages=[HttpChatMessage.user("What uptime does the SLA guarantee?")],
)
for chunk in client.chat_stream(req):
    print(chunk, end="", flush=True)
print()
```

**Error handling during streaming:**

```python
from wauldo import WauldoError, ServerError, AgentConnectionError

try:
    for chunk in client.chat_stream(request):
        print(chunk, end="", flush=True)
except AgentConnectionError:
    print("\n[connection lost]")
except ServerError as e:
    print(f"\n[server error: {e}]")
except WauldoError as e:
    print(f"\n[error: {e}]")
```

See [`examples/streaming_chat.py`](examples/streaming_chat.py) and [`examples/async_streaming.py`](examples/async_streaming.py) for runnable scripts.

---

## Async Support

```bash
pip install wauldo[async]
```

```python
import asyncio
from wauldo import AsyncHttpClient

async def main():
    async with AsyncHttpClient(base_url="https://api.wauldo.com", api_key="YOUR_API_KEY") as client:
        result = await client.rag_query("What are the payment terms?")
        print(result.answer)

asyncio.run(main())
```

All sync methods have async equivalents. *Contributed by [@qorexdev](https://github.com/qorexdev).*

---

## Features

- **Pre-generation fact extraction** — numbers, dates, limits injected as constraints before the LLM call
- **Post-generation grounding check** — every answer verified against sources
- **Citation validation** — detects phantom references
- **Fact-check API** — verify any claim against any source (3 modes: lexical, hybrid, semantic)
- **Native PDF/DOCX upload** — server-side extraction with quality scoring
- **Smart model routing** — auto-selects cheapest model that meets quality
- **OpenAI-compatible** — swap your `base_url`, keep your existing code
- **Sync + Async** — full async/await support

---

## Built For

- Production RAG systems that need **reliable answers**
- Teams where **"confidently wrong" is unacceptable**
- Legal, finance, healthcare, support automation
- Anyone replacing "hope-based" RAG

---

## Benchmarks

| Metric | Result |
|--------|--------|
| Hallucination rate | **0%** |
| Accuracy | **83%** (17% = correct refusals) |
| Eval tasks | 61 |
| LLMs tested | 14 models, 3 runs each |
| Avg latency | ~1.2s |

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

## Contributing

PRs welcome. Check the [good first issues](https://github.com/wauldo/wauldo-sdk-python/labels/good%20first%20issue).

## Contributors

- [@qorexdev](https://github.com/qorexdev) — async client

## License

MIT — see [LICENSE](./LICENSE)
