"""Tests for HttpClient.guard() — hallucination firewall."""

import json
from unittest.mock import MagicMock, patch

import pytest

from wauldo.http_client import HttpClient
from wauldo.http_types import GuardResponse


def _mock_urlopen_response(body: dict, status: int = 200) -> MagicMock:
    raw = json.dumps(body).encode()
    resp = MagicMock()
    resp.read.return_value = raw
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


_VERIFIED_BODY = {
    "verdict": "verified",
    "action": "allow",
    "hallucination_rate": 0.0,
    "mode": "lexical",
    "total_claims": 1,
    "supported_claims": 1,
    "confidence": 1.0,
    "claims": [
        {
            "text": "Paris is in France",
            "supported": True,
            "confidence": 1.0,
            "verdict": "verified",
            "action": "allow",
        }
    ],
}

_REJECTED_BODY = {
    "verdict": "rejected",
    "action": "block",
    "hallucination_rate": 1.0,
    "mode": "lexical",
    "total_claims": 1,
    "supported_claims": 0,
    "confidence": 0.0,
    "claims": [
        {
            "text": "Returns accepted within 60 days",
            "supported": False,
            "confidence": 0.3,
            "verdict": "rejected",
            "action": "block",
            "reason": "numerical_mismatch",
        }
    ],
}

_WEAK_BODY = {
    "verdict": "weak",
    "action": "review",
    "hallucination_rate": 0.5,
    "mode": "lexical",
    "total_claims": 2,
    "supported_claims": 1,
    "confidence": 0.5,
    "claims": [
        {
            "text": "Claim A",
            "supported": True,
            "confidence": 0.8,
            "verdict": "verified",
            "action": "allow",
        },
        {
            "text": "Claim B",
            "supported": False,
            "confidence": 0.2,
            "verdict": "rejected",
            "action": "block",
            "reason": "insufficient_evidence",
        },
    ],
}


class TestGuardVerified:
    """Guard returns verified when claim matches source."""

    @patch("wauldo.http_client.urlopen")
    def test_guard_verified(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_response(_VERIFIED_BODY)
        client = HttpClient(base_url="http://localhost:3000", api_key="test")
        result = client.guard(
            text="Paris is in France",
            source_context="Paris is the capital of France.",
        )
        assert isinstance(result, GuardResponse)
        assert result.verdict == "verified"
        assert result.is_safe is True
        assert result.is_blocked is False
        assert result.confidence == 1.0
        assert result.hallucination_rate == 0.0
        assert len(result.claims) == 1
        assert result.claims[0].supported is True


class TestGuardRejected:
    """Guard returns rejected when claim contradicts source."""

    @patch("wauldo.http_client.urlopen")
    def test_guard_rejected(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_response(_REJECTED_BODY)
        client = HttpClient(base_url="http://localhost:3000", api_key="test")
        result = client.guard(
            text="Returns accepted within 60 days",
            source_context="Our return policy: 14 days.",
        )
        assert result.verdict == "rejected"
        assert result.is_safe is False
        assert result.is_blocked is True
        assert result.confidence == 0.0
        assert result.claims[0].reason == "numerical_mismatch"


class TestGuardWeak:
    """Guard returns weak when claims are mixed."""

    @patch("wauldo.http_client.urlopen")
    def test_guard_weak_review(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_response(_WEAK_BODY)
        client = HttpClient(base_url="http://localhost:3000", api_key="test")
        result = client.guard(text="mixed claims", source_context="source")
        assert result.verdict == "weak"
        assert result.action == "review"
        assert result.is_safe is False
        assert result.is_blocked is False
        assert result.total_claims == 2
        assert result.supported_claims == 1


class TestGuardError:
    """Guard handles HTTP errors gracefully."""

    @patch("wauldo.http_client.urlopen")
    def test_guard_401(self, mock_urlopen):
        from wauldo.exceptions import WauldoError
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="http://localhost:3000/v1/fact-check",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )
        client = HttpClient(base_url="http://localhost:3000", api_key="bad-key")
        with pytest.raises(WauldoError):
            client.guard(text="test", source_context="test")


class TestGuardModes:
    """Guard accepts different verification modes."""

    @patch("wauldo.http_client.urlopen")
    def test_guard_lexical_mode(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen_response(_VERIFIED_BODY)
        client = HttpClient(base_url="http://localhost:3000", api_key="test")
        result = client.guard(text="t", source_context="t", mode="lexical")
        assert result.mode == "lexical"

    @patch("wauldo.http_client.urlopen")
    def test_guard_hybrid_mode(self, mock_urlopen):
        body = {**_VERIFIED_BODY, "mode": "hybrid"}
        mock_urlopen.return_value = _mock_urlopen_response(body)
        client = HttpClient(base_url="http://localhost:3000", api_key="test")
        result = client.guard(text="t", source_context="t", mode="hybrid")
        assert result.mode == "hybrid"
