"""CLI for Wauldo — query and verify RAG answers from terminal."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

# handle windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


def _get_client(args: argparse.Namespace) -> Any:
    if args.mock:
        from .mock_client import MockHttpClient
        return MockHttpClient()

    from .http_client import HttpClient

    api_key = os.environ.get("WAULDO_API_KEY")
    base_url = os.environ.get("WAULDO_BASE_URL", "https://api.wauldo.com")

    if not api_key:
        print("error: WAULDO_API_KEY not set", file=sys.stderr)
        print("  export WAULDO_API_KEY=your_key", file=sys.stderr)
        print("  or use --mock for demo mode", file=sys.stderr)
        sys.exit(1)

    return HttpClient(base_url=base_url, api_key=api_key)


def _output(data: dict, args: argparse.Namespace) -> None:
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    if getattr(args, "raw", False):
        if "answer" in data:
            print(data["answer"])
        elif "verdict" in data:
            print(data["verdict"])
        else:
            print(json.dumps(data, ensure_ascii=False))
        return

    # pretty output
    for key, val in data.items():
        if key == "_header":
            print(val)
            print()
        elif key == "_sources":
            print("Sources:")
            for i, src in enumerate(val, 1):
                score = src.get("score", "?")
                doc = src.get("document_id", "unknown")
                content = src.get("content", "")
                preview = content[:80].replace("\n", " ")
                print(f"  [{i}] {doc} (score: {score})")
                if preview:
                    print(f'      "{preview}..."')
            print()
        elif key == "_footer":
            print(val)
        elif isinstance(val, list):
            print(f"{key}:")
            for item in val:
                print(f"  - {item}")
        else:
            print(f"{key}: {val}")


def cmd_upload(args: argparse.Namespace) -> None:
    client = _get_client(args)
    path = args.file

    if not os.path.isfile(path):
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()

    ext = os.path.splitext(path)[1].lower()
    if ext in (".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg"):
        resp = client.upload_file(file_path=path, title=args.title, tags=args.tags)
    else:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        resp = client.rag_upload(content=content, filename=os.path.basename(path))

    elapsed = time.time() - t0

    if getattr(args, "json", False):
        d = resp.model_dump() if hasattr(resp, "model_dump") else {"document_id": resp.document_id, "chunks_count": resp.chunks_count}
        d["latency_s"] = round(elapsed, 2)
        print(json.dumps(d, indent=2, ensure_ascii=False))
    elif getattr(args, "raw", False):
        print(resp.document_id)
    else:
        print(f"\u2713 Uploaded {os.path.basename(path)}")
        print(f"  doc_id: {resp.document_id}")
        print(f"  chunks: {resp.chunks_count}")
        print(f"  time:   {elapsed:.1f}s")


def cmd_query(args: argparse.Namespace) -> None:
    client = _get_client(args)

    t0 = time.time()
    resp = client.rag_query(query=args.question, top_k=args.top_k)
    elapsed = time.time() - t0

    confidence = resp.get_confidence()
    grounded = resp.get_grounded()

    if getattr(args, "json", False):
        d = resp.model_dump()
        d["latency_s"] = round(elapsed, 2)
        print(json.dumps(d, indent=2, ensure_ascii=False))
        return

    if getattr(args, "raw", False):
        print(resp.answer)
        return

    header = "\u2713 Verified Answer" if grounded else "? Answer (unverified)"
    print(header)
    print(resp.answer)
    print()

    if resp.sources:
        print("Sources:")
        for i, s in enumerate(resp.sources, 1):
            preview = s.content[:80].replace("\n", " ")
            print(f"  [{i}] {s.document_id} (score: {s.score})")
            if preview:
                print(f'      "{preview}..."')
        print()

    footer_parts = []
    if confidence is not None:
        footer_parts.append(f"Confidence: {confidence:.2f}")
    if grounded is not None:
        footer_parts.append(f"Grounded: {'yes' if grounded else 'no'}")
    footer_parts.append(f"Latency: {elapsed:.1f}s")
    print(" | ".join(footer_parts))


def cmd_fact_check(args: argparse.Namespace) -> None:
    client = _get_client(args)

    source_text = args.source
    if source_text and os.path.isfile(source_text):
        with open(source_text, "r", encoding="utf-8") as f:
            source_text = f.read()

    if not source_text:
        print("error: --source is required (file path or inline text)", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    resp = client.fact_check(text=args.claim, source_context=source_text, mode=args.mode)
    elapsed = time.time() - t0

    if getattr(args, "json", False):
        d = resp.model_dump()
        d["latency_s"] = round(elapsed, 2)
        print(json.dumps(d, indent=2, ensure_ascii=False))
        return

    if getattr(args, "raw", False):
        print(resp.verdict)
        return

    icon = "\u2713" if resp.verdict == "verified" else "\u2717"
    print(f"{icon} {resp.verdict.upper()} ({resp.action})")
    print(f"  hallucination rate: {resp.hallucination_rate:.1%}")
    print(f"  claims: {resp.supported_claims}/{resp.total_claims} supported")
    print()

    for claim in resp.claims:
        mark = "\u2713" if claim.supported else "\u2717"
        print(f"  {mark} {claim.text}")
        if claim.reason:
            print(f"    reason: {claim.reason}")

    print()
    print(f"Mode: {resp.mode} | Confidence: {resp.confidence:.2f} | Latency: {elapsed:.1f}s")


def cmd_verify(args: argparse.Namespace) -> None:
    client = _get_client(args)

    sources = None
    if args.sources:
        sources = []
        for s in args.sources:
            if os.path.isfile(s):
                with open(s, "r", encoding="utf-8") as f:
                    sources.append({"name": os.path.basename(s), "content": f.read()})
            else:
                sources.append({"name": f"source_{len(sources)+1}", "content": s})

    t0 = time.time()
    resp = client.verify_citation(
        text=args.text,
        sources=sources if sources else None,
        threshold=args.threshold,
    )
    elapsed = time.time() - t0

    if getattr(args, "json", False):
        d = resp.model_dump()
        d["latency_s"] = round(elapsed, 2)
        print(json.dumps(d, indent=2, ensure_ascii=False))
        return

    if getattr(args, "raw", False):
        print(f"{resp.citation_ratio:.2f}")
        return

    icon = "\u2713" if resp.has_sufficient_citations else "\u2717"
    print(f"{icon} Citation ratio: {resp.citation_ratio:.0%} ({resp.citation_count}/{resp.sentence_count} sentences)")

    if resp.uncited_sentences:
        print()
        print("Uncited:")
        for s in resp.uncited_sentences[:5]:
            print(f"  - {s[:100]}")

    if resp.phantom_count and resp.phantom_count > 0:
        print(f"\n  \u26a0 {resp.phantom_count} phantom citation(s) detected")

    print(f"\nLatency: {elapsed:.1f}s")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="wauldo",
        description="Query and verify RAG answers from your terminal",
    )
    parser.add_argument("--mock", action="store_true", help="use mock client (no API key needed)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--raw", action="store_true", help="raw output (just the answer)")
    parser.add_argument("--version", action="version", version="%(prog)s 0.7.0")

    sub = parser.add_subparsers(dest="command")

    # upload
    p_upload = sub.add_parser("upload", help="upload a document for RAG indexing")
    p_upload.add_argument("file", help="path to document (PDF, DOCX, TXT, etc.)")
    p_upload.add_argument("--title", help="document title")
    p_upload.add_argument("--tags", help="comma-separated tags")

    # query
    p_query = sub.add_parser("query", help="ask a question against uploaded docs")
    p_query.add_argument("question", help="your question")
    p_query.add_argument("--top-k", type=int, default=5, help="number of sources (default: 5)")

    # fact-check
    p_fc = sub.add_parser("fact-check", help="verify claims against source text")
    p_fc.add_argument("claim", help="text containing claims to verify")
    p_fc.add_argument("--source", required=True, help="source text or file path to verify against")
    p_fc.add_argument("--mode", default="lexical", choices=["lexical", "hybrid", "semantic"])

    # verify
    p_verify = sub.add_parser("verify", help="check citation coverage in AI-generated text")
    p_verify.add_argument("text", help="AI-generated text to verify")
    p_verify.add_argument("--sources", nargs="+", help="source files or inline text")
    p_verify.add_argument("--threshold", type=float, help="min citation ratio (0.0-1.0)")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "upload": cmd_upload,
        "query": cmd_query,
        "fact-check": cmd_fact_check,
        "verify": cmd_verify,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
