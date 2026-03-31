"""SSE streaming chat completion using the Wauldo HttpClient."""

import sys
from wauldo import HttpClient, ChatRequest, HttpChatMessage

def main() -> None:
    client = HttpClient(base_url="http://localhost:3000")

    request = ChatRequest(
        model="qwen2.5:7b",
        messages=[
            HttpChatMessage.system("You are a helpful assistant."),
            HttpChatMessage.user("Write a haiku about Rust programming."),
        ],
        temperature=0.7,
    )

    print("Streaming response: ", end="", flush=True)
    token_count = 0
    for chunk in client.chat_stream(request):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        token_count += 1

    print(f"\n\nReceived {token_count} chunks.")

if __name__ == "__main__":
    main()
