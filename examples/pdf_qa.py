"""Upload a product manual (PDF) and ask technical questions about it."""

import os
from wauldo import HttpClient

API_KEY = os.getenv("WAULDO_API_KEY", "")
BASE_URL = os.getenv("WAULDO_BASE_URL", "https://api.wauldo.com")


def main():
    client = HttpClient(base_url=BASE_URL, api_key=API_KEY)

    # upload the PDF — server handles text extraction
    result = client.upload_file("product_manual.pdf", title="Product Manual")
    print(f"Uploaded: {result.document_id} ({result.chunks_count} chunks)")
    if result.quality:
        print(f"Quality: {result.quality.label} (score {result.quality.score:.2f})")

    # ask questions against the indexed manual
    questions = [
        "What is the maximum operating temperature?",
        "How do I reset the device to factory settings?",
        "What warranty does this product come with?",
    ]
    for q in questions:
        resp = client.rag_query(q, top_k=3)
        print(f"\nQ: {q}")
        print(f"A: {resp.answer}")
        if resp.audit:
            print(f"   confidence={resp.audit.confidence:.0%}, grounded={resp.audit.grounded}")


if __name__ == "__main__":
    main()
