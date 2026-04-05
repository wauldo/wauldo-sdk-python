"""Upload multiple documents and cross-reference answers between them."""

import os
from wauldo import HttpClient

API_KEY = os.getenv("WAULDO_API_KEY", "")
BASE_URL = os.getenv("WAULDO_BASE_URL", "https://api.wauldo.com")

DOCUMENTS = {
    "q3_report.txt": (
        "Q3 2024 Financial Report\n\n"
        "Revenue: $2.4M (up 18% YoY). Operating expenses: $1.8M. "
        "Net profit: $420K. Customer count grew from 850 to 1,120. "
        "Churn rate decreased to 3.2%. ARR reached $9.6M."
    ),
    "q2_report.txt": (
        "Q2 2024 Financial Report\n\n"
        "Revenue: $2.1M (up 12% YoY). Operating expenses: $1.7M. "
        "Net profit: $280K. Customer count: 850. "
        "Churn rate: 4.1%. ARR: $8.4M."
    ),
    "product_roadmap.txt": (
        "Product Roadmap H2 2024\n\n"
        "Q3: Launch enterprise SSO and audit logs. "
        "Q4: Release API v2 with webhook support and batch processing. "
        "Target: reduce onboarding time from 3 days to 4 hours. "
        "Hiring: 2 senior engineers, 1 product designer by end of Q3."
    ),
}


def main():
    client = HttpClient(base_url=BASE_URL, api_key=API_KEY)

    for name, content in DOCUMENTS.items():
        upload = client.rag_upload(content=content, filename=name)
        print(f"Indexed {name}: {upload.chunks_count} chunks")

    print()

    # cross-reference queries that span multiple docs
    queries = [
        "How did revenue change between Q2 and Q3?",
        "What was the customer growth trajectory?",
        "Did the Q3 hiring targets align with the product roadmap?",
        "What is the current ARR and how did it grow?",
    ]

    for q in queries:
        resp = client.rag_query(q, top_k=5)
        print(f"Q: {q}")
        print(f"A: {resp.answer}")
        if resp.sources:
            docs = {s.document_id for s in resp.sources}
            print(f"   sources: {', '.join(docs)}")
        print()


if __name__ == "__main__":
    main()
