"""Build a support chatbot from FAQ/knowledge base docs.

Uploads support articles, then runs an interactive Q&A loop
where answers are always verified against the source material.
"""

import os
from wauldo import HttpClient

API_KEY = os.getenv("WAULDO_API_KEY", "")
BASE_URL = os.getenv("WAULDO_BASE_URL", "https://api.wauldo.com")

SUPPORT_DOCS = {
    "returns.txt": (
        "Returns & Refunds Policy\n\n"
        "Items can be returned within 30 days of purchase. "
        "Refunds are processed within 5-7 business days. "
        "Items must be in original packaging. "
        "Digital products are non-refundable after download."
    ),
    "shipping.txt": (
        "Shipping Information\n\n"
        "Standard shipping: 5-7 business days. "
        "Express shipping: 1-2 business days ($12.99 extra). "
        "Free shipping on orders over $50. "
        "International shipping available to 40+ countries."
    ),
    "account.txt": (
        "Account & Billing\n\n"
        "Password reset via email takes up to 10 minutes. "
        "Two-factor authentication is available in Settings > Security. "
        "Billing cycle is the 1st of each month. "
        "Cancel anytime — no cancellation fees."
    ),
}


def main():
    client = HttpClient(base_url=BASE_URL, api_key=API_KEY)

    # index all support docs
    for name, content in SUPPORT_DOCS.items():
        upload = client.rag_upload(content=content, filename=name)
        print(f"Indexed {name}: {upload.chunks_count} chunks")

    print("\nSupport bot ready. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        resp = client.rag_query(question, top_k=3)
        confidence = resp.get_confidence()

        if confidence and confidence < 0.4:
            print("Bot: I don't have enough information to answer that reliably.")
            print("     Please contact support@example.com for help.\n")
        else:
            print(f"Bot: {resp.answer}")
            if resp.sources:
                src = resp.sources[0]
                print(f"     (source: {src.document_id}, score: {src.score:.2f})\n")


if __name__ == "__main__":
    main()
