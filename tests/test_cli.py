import tempfile
import os

from wauldo.cli import main


def test_query_mock(capsys):
    main(["--mock", "query", "What is AI?"])
    out = capsys.readouterr().out
    assert "Answer" in out or "answer" in out or "mock response" in out.lower()


def test_guard_mock(capsys):
    main(["--mock", "guard", "The sky is blue", "--source", "The sky is blue on clear days"])
    out = capsys.readouterr().out
    assert "VERIFIED" in out or "verified" in out


def test_upload_mock(capsys, tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    main(["--mock", "upload", str(f)])
    out = capsys.readouterr().out
    assert "Uploaded" in out or "doc_id" in out


def test_upload_pdf_mock(capsys, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake pdf content")
    main(["--mock", "upload", str(f)])
    out = capsys.readouterr().out
    assert "Uploaded" in out or "doc_id" in out


def test_query_json_mock(capsys):
    main(["--mock", "--json", "query", "test question"])
    out = capsys.readouterr().out
    assert '"answer"' in out or '"sources"' in out


def test_query_raw_mock(capsys):
    main(["--mock", "--raw", "query", "test question"])
    out = capsys.readouterr().out
    assert "mock response" in out.lower()
