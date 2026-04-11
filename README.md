<div align="center">

<br />

# 🐍 Wauldo Python SDK

### Verified RAG for Python — trust score on every answer

<br />

**Your LLM passes demos.**
**It fails in production.**

One import, two lines — plug Wauldo Guard on top of LangChain / LlamaIndex / Haystack and get a numeric trust_score + verdict (`SAFE` / `CONFLICT` / `UNVERIFIED` / `BLOCK`) on every response.

<br />

[![PyPI](https://img.shields.io/pypi/v/wauldo.svg?style=for-the-badge&label=pypi&color=3776ab)](https://pypi.org/project/wauldo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Leaderboard](https://img.shields.io/badge/📊_96%25_adversarial-wauldo.com%2Fleaderboard-3b82f6?style=for-the-badge)](https://wauldo.com/leaderboard)

<br />

<sub>Python 3.9+ · MIT · wraps any RAG pipeline · reproducible bench: [wauldo-leaderboard](https://github.com/wauldo/wauldo-leaderboard)</sub>

</div>

---

## Quickstart (30 seconds)

```bash
pip install wauldo
```

### Try it locally (no API key needed)

```python
from wauldo import MockHttpClient

client = MockHttpClient(chat_response="Returns are accepted within 60 days.")

client.rag_upload(content="Our refund policy allows returns within 60 days.", filename="policy.txt")
result = client.rag_query("What is the refund policy?")
print(result.answer)    # Returns are accepted within 60 days.
print(result.sources)   # [RagSource(document_id='mock-doc-001', ...)]
```

Run [`examples/quickstart.py`](examples/quickstart.py) for the full offline walkthrough.

### With a real server

```python
from wauldo import HttpClient

client = HttpClient(base_url="https://api.wauldo.com", api_key="YOUR_API_KEY")

client.rag_upload(content="Our refund policy allows returns within 60 days...", filename="policy.txt")
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

### Guard — catch hallucinations (2 lines)

```python
result = client.guard(
    text="Returns are accepted within 60 days of purchase",
    source_context="Our return policy allows returns within 14 days.",
)
print(result.verdict)            # "rejected"
print(result.action)             # "block"
print(result.claims[0].reason)   # "numerical_mismatch"
print(result.is_blocked)         # True
print(result.hallucination_rate) # 1.0
```

Guard verifies any LLM output against source documents. Wrong answers get blocked before they reach your users. 3 modes: `lexical` (<1ms), `hybrid` (~50ms), `semantic` (~500ms).

### Upload a PDF and ask questions

```python
result = client.upload_file("contract.pdf", title="Q3 Contract")
print(f"Extracted {result.chunks_count} chunks, quality: {result.quality_label}")
# -> Extracted 12 chunks, quality: high

result = client.rag_query("What are the payment terms?")
print(f"Answer: {result.answer}")
# -> Answer: Net 30 from invoice date.
print(f"Confidence: {result.get_confidence():.0%}")
# -> Confidence: 94%
print(f"Grounded: {result.audit.grounded}")
# -> Grounded: True
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

### Real-world use cases

| Example | Description |
|---------|-------------|
| [`pdf_qa.py`](examples/pdf_qa.py) | Upload a product manual PDF and ask technical questions |
| [`support_chatbot.py`](examples/support_chatbot.py) | Build a verified support chatbot from FAQ docs |
| [`contract_analysis.py`](examples/contract_analysis.py) | Extract clauses from a contract + fact-check claims |
| [`multi_document.py`](examples/multi_document.py) | Cross-reference answers across multiple documents |

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

## CLI

*Contributed by [@qorexdev](https://github.com/qorexdev).*

```bash
# Set your API key
export WAULDO_API_KEY=your_key

# Upload a document
wauldo upload --content "Our return policy allows returns within 14 days."

# Query
wauldo query "What is the return policy?"

# Guard — fact-check a claim
wauldo fact-check --text "Returns accepted within 60 days" --source "Returns within 14 days."

# Verify citations
wauldo verify --text "The policy covers damage [Source: Manual]."

# Offline mode (no server)
wauldo query "test" --mock
```

---

## Features

- **Pre-generation fact extraction** — numbers, dates, limits injected as constraints before the LLM call
- **Post-generation grounding check** — every answer verified against sources
- **Citation validation** — detects phantom references
- **Analytics & Insights** — track token savings, cache performance, cost per hour, and per-tenant traffic
- **Guard method** — hallucination firewall: `client.guard(text, source_context)` → verdict, action, claims with confidence scoring
- **Fact-check API** — verify any claim against any source (3 modes: lexical, hybrid, semantic)
- **Native PDF/DOCX upload** — server-side extraction with quality scoring
- **Smart model routing** — auto-selects cheapest model that meets quality
- **OpenAI-compatible** — swap your `base_url`, keep your existing code
- **Sync + Async** — full async/await support
- **CLI** — `wauldo upload`, `wauldo query`, `wauldo fact-check` from terminal

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

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ConnectionError` | Server unreachable | Check `base_url`, make sure the server is running |
| `ServerError (401)` | Bad API key | Verify your key at [RapidAPI dashboard](https://rapidapi.com/binnewzzin/api/smart-rag-api) |
| `AgentTimeoutError` | Request took too long | Pass `timeout_ms=30000` or try a smaller document |
| `ModuleNotFoundError: aiohttp` | Async extras not installed | Run `pip install wauldo[async]` |
| `ImportError: MockHttpClient` | Old SDK version | Run `pip install --upgrade wauldo` |

Still stuck? [Open an issue](https://github.com/wauldo/wauldo-sdk-python/issues/new).

---

## Contributing

PRs welcome. Check the [good first issues](https://github.com/wauldo/wauldo-sdk-python/labels/good%20first%20issue).

### Contributors

- [@qorexdev](https://github.com/qorexdev) — async client, streaming, MockHttpClient, quickstart, CONTRIBUTING guide
- [@dagangtj](https://github.com/dagangtj) — analytics demo + MockHttpClient analytics methods

---

## 🔗 Related

- **[wauldo.com](https://wauldo.com)** — platform
- **[wauldo.com/leaderboard](https://wauldo.com/leaderboard)** — live RAG framework bench (6 frameworks, daily refresh)
- **[wauldo.com/guard](https://wauldo.com/guard)** — verification layer docs
- **[github.com/wauldo/wauldo-leaderboard](https://github.com/wauldo/wauldo-leaderboard)** — reproducible bench runner, MIT
- **[github.com/wauldo/wauldo-sdk-js](https://github.com/wauldo/wauldo-sdk-js)** — TypeScript peer SDK
- **[github.com/wauldo/wauldo-sdk-rust](https://github.com/wauldo/wauldo-sdk-rust)** — Rust peer SDK

---

## 📄 License

MIT — see [LICENSE](./LICENSE).

<div align="center">

<br />

<sub>Built by the Wauldo team. If this changed your mind about your RAG stack, give it a ⭐.</sub>

</div>
