"""Quickstart: see the SDK in action without an API key.

Uses MockHttpClient for a local dry run, then shows how to
switch to a real server with one line change.

Run:
    pip install wauldo
    python examples/quickstart.py
"""

from wauldo import MockHttpClient

client = MockHttpClient(chat_response="Returns are accepted within 60 days of purchase.")

# upload a doc and query it
upload = client.rag_upload(content="Our refund policy allows returns within 60 days.", filename="policy.txt")
print(f"Uploaded: {upload.document_id} ({upload.chunks_count} chunks)")

result = client.rag_query("What is the refund policy?")
print(f"Answer: {result.answer}")
print(f"Sources: {len(result.sources)}")
for src in result.sources:
    print(f"  [{src.score:.2f}] {src.content}")

# chat completion
reply = client.chat_simple("default", "Explain RAG in one sentence")
print(f"\nChat: {reply}")

# streaming
print("\nStreaming: ", end="")
from wauldo import ChatRequest, HttpChatMessage

req = ChatRequest(model="default", messages=[HttpChatMessage.user("Hello")])
for chunk in client.chat_stream(req):
    print(chunk, end="", flush=True)
print()

print("\n--- To use a real server, replace MockHttpClient with HttpClient ---")
print("  from wauldo import HttpClient")
print('  client = HttpClient(base_url="https://api.wauldo.com", api_key="YOUR_KEY")')
