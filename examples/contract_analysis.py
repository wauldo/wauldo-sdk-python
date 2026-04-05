"""Upload a contract and extract specific clauses using verified RAG queries."""

import os
from wauldo import HttpClient

API_KEY = os.getenv("WAULDO_API_KEY", "")
BASE_URL = os.getenv("WAULDO_BASE_URL", "https://api.wauldo.com")

SAMPLE_CONTRACT = """
SERVICE AGREEMENT

1. TERM: This agreement is effective from January 1, 2025 through December 31, 2025,
   with automatic renewal for successive one-year periods unless terminated with 90 days
   written notice.

2. PAYMENT TERMS: Client shall pay $5,000/month, due within 30 days of invoice date.
   Late payments incur 1.5% monthly interest. Annual prepayment receives 10% discount.

3. TERMINATION: Either party may terminate with 90 days written notice. Early termination
   by Client requires payment of remaining months at 50% rate. Provider may terminate
   immediately for non-payment exceeding 60 days.

4. LIABILITY: Provider's total liability shall not exceed fees paid in the preceding
   12 months. Neither party is liable for indirect, consequential, or punitive damages.

5. CONFIDENTIALITY: Both parties agree to keep confidential information secret for
   3 years after termination. This includes pricing, technical data, and customer lists.

6. GOVERNING LAW: This agreement is governed by the laws of the State of Delaware.
"""

CLAUSES_TO_EXTRACT = [
    "What are the payment terms and late fees?",
    "How can either party terminate this agreement?",
    "What is the liability cap?",
    "How long does the confidentiality obligation last?",
    "What happens with automatic renewal?",
]


def main():
    client = HttpClient(base_url=BASE_URL, api_key=API_KEY)

    upload = client.rag_upload(content=SAMPLE_CONTRACT, filename="service_agreement.txt")
    print(f"Contract indexed: {upload.chunks_count} chunks\n")

    for q in CLAUSES_TO_EXTRACT:
        resp = client.rag_query(q, top_k=3)
        grounded = resp.get_grounded()
        print(f"Q: {q}")
        print(f"A: {resp.answer}")
        print(f"   grounded={grounded}\n")

    # fact-check a claim against the contract
    print("--- Fact-check ---")
    check = client.fact_check(
        text="Late payments incur 2% monthly interest.",
        source_context=SAMPLE_CONTRACT,
        mode="lexical",
    )
    print(f"Claim: 'Late payments incur 2% monthly interest.'")
    print(f"Verdict: {check.verdict} | Action: {check.action}")
    if check.claims:
        print(f"Reason: {check.claims[0].reason}")


if __name__ == "__main__":
    main()
