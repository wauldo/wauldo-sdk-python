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

### Guard — catch hallucinations in 3 lines

```python
from wauldo import HttpClient

client = HttpClient(base_url="https://api.wauldo.com", api_key="YOUR_API_KEY")

result = client.guard(
    text="Returns are accepted within 60 days.",
    source_context="Our policy allows returns within 14 days.",
)
print(result.verdict)       # "rejected"
print(result.claims[0].reason)  # "numerical_mismatch"
```

### Verified RAG — upload, ask, verify

```python
client.rag_upload(content="Our refund policy allows returns within 60 days...", filename="policy.txt")

result = client.rag_query("What is the refund policy?")
print(result.answer)        # Verified answer with sources
print(result.sources)       # [RagSource(document_id='...', score=0.95)]
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
result = client.upload_file("contract.pdf", title="Q3 Contract")
print(f"Extracted {result.chunks_count} chunks")

result = client.rag_query("What are the payment terms?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.get_confidence():.0%}")
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

---

## Features

- **Pre-generation fact extraction** — numbers, dates, limits injected as constraints before the LLM call
- **Post-generation grounding check** — every answer verified against sources
- **Guard API** — verify any claim against any source (3 modes: lexical, hybrid, semantic)
- **Native PDF/DOCX upload** — server-side extraction with quality scoring
- **Smart model routing** — auto-selects cheapest model that meets quality
- **OpenAI-compatible** — swap your `base_url`, keep your existing code
- **Sync** — simple, synchronous API

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
