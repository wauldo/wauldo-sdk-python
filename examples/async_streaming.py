"""Async SSE streaming with the Wauldo AsyncHttpClient."""

import asyncio
import sys
from wauldo import AsyncHttpClient, ChatRequest, HttpChatMessage


async def basic_stream():
    """Stream a chat response, printing tokens as they arrive."""
    async with AsyncHttpClient(base_url="http://localhost:3000") as client:
        req = ChatRequest(
            model="auto",
            messages=[
                HttpChatMessage.system("You are a helpful assistant."),
                HttpChatMessage.user("Explain how HTTP/2 multiplexing works."),
            ],
        )

        chunks = 0
        async for token in client.chat_stream(req):
            sys.stdout.write(token)
            sys.stdout.flush()
            chunks += 1

        print(f"\n\n({chunks} chunks)")


async def stream_with_timeout():
    """Stream with a manual timeout using asyncio.wait_for."""
    async with AsyncHttpClient(base_url="http://localhost:3000") as client:
        req = ChatRequest.quick("auto", "Write a short poem about databases.")

        try:
            async for token in asyncio.wait_for(
                collect_stream(client, req), timeout=30.0
            ):
                print(token, end="", flush=True)
        except asyncio.TimeoutError:
            print("\n[timed out]")


async def collect_stream(client, req):
    async for token in client.chat_stream(req):
        yield token


if __name__ == "__main__":
    asyncio.run(basic_stream())
