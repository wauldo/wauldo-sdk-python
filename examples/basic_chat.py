"""Basic chat completion using the Wauldo HttpClient."""

from wauldo import HttpClient, ChatRequest, HttpChatMessage

def main() -> None:
    client = HttpClient(base_url="http://localhost:3000")

    # Quick one-liner chat
    reply = client.chat_simple("qwen2.5:7b", "What is Rust?")
    print(f"Simple reply: {reply[:120]}...")

    # Full request with parameters
    request = ChatRequest(
        model="qwen2.5:7b",
        messages=[
            HttpChatMessage.system("You are a concise assistant."),
            HttpChatMessage.user("Explain async/await in 2 sentences."),
        ],
        temperature=0.3,
        max_tokens=200,
    )
    response = client.chat(request)
    print(f"Model: {response.model}")
    print(f"Reply: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.total_tokens}")

if __name__ == "__main__":
    main()
