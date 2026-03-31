"""RAG workflow: upload a document then query the knowledge base."""

from wauldo import HttpClient

def main() -> None:
    client = HttpClient(base_url="http://localhost:3000")

    # Upload a document for RAG indexing
    doc = (
        "Rust is a systems programming language focused on safety and performance. "
        "It prevents data races at compile time through its ownership system. "
        "The borrow checker enforces memory safety without a garbage collector."
    )
    upload = client.rag_upload(content=doc, filename="rust_intro.txt")
    print(f"Uploaded: id={upload.document_id}, chunks={upload.chunks_count}")

    # Query the knowledge base
    result = client.rag_query("How does Rust ensure memory safety?", top_k=3)
    print(f"\nAnswer: {result.answer}")
    print(f"\nSources ({len(result.sources)}):")
    for src in result.sources:
        print(f"  - [{src.score:.2f}] {src.content[:80]}...")

if __name__ == "__main__":
    main()
