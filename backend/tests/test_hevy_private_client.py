from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
from urllib.parse import quote

import pytest
from hevy_private_client import HevyPrivateClient, HevyPrivateError, HevyTokens


def _tokens(*, expired: bool = False) -> HevyTokens:
    delta = timedelta(minutes=-1 if expired else 10)
    return HevyTokens("access-secret", "refresh-secret", datetime.now(timezone.utc) + delta)


def test_auth_cookie_is_decoded_without_exposing_credentials() -> None:
    value = quote(json.dumps({
        "access_token": "access-secret",
        "refresh_token": "refresh-secret",
        "expires_at": "2030-01-01T00:00:00Z",
    }))
    parsed = HevyTokens.from_cookie(value)
    assert parsed.access_token == "access-secret"
    assert parsed.refresh_token == "refresh-secret"
    assert parsed.expires_at.tzinfo is not None


def test_invalid_cookie_has_safe_error() -> None:
    with pytest.raises(HevyPrivateError, match="invalid") as caught:
        HevyTokens.from_cookie("not-json-and-no-secret")
    assert "not-json" not in str(caught.value)


def test_expired_session_refreshes_and_rotates_both_tokens() -> None:
    http = Mock()
    http.post.return_value.ok = True
    http.post.return_value.json.return_value = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expires_at": "2030-01-01T00:00:00Z",
    }
    http.request.return_value.ok = True
    http.request.return_value.status_code = 200
    http.request.return_value.json.return_value = {"id": "account-1"}
    rotated: list[HevyTokens] = []
    client = HevyPrivateClient(_tokens(expired=True), http=http, on_tokens_rotated=rotated.append)

    assert client.account() == {"id": "account-1"}
    assert client.tokens.access_token == "new-access"
    assert client.tokens.refresh_token == "new-refresh"
    assert rotated == [client.tokens]
    refresh_call = http.post.call_args.kwargs
    assert refresh_call["json"] == {"refresh_token": "refresh-secret"}


def test_401_refreshes_and_retries_only_once() -> None:
    http = Mock()
    unauthorized = Mock(status_code=401, ok=False)
    success = Mock(status_code=200, ok=True)
    success.json.return_value = {"id": "account-1"}
    http.request.side_effect = [unauthorized, success]
    http.post.return_value.ok = True
    http.post.return_value.json.return_value = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expires_at": "2030-01-01T00:00:00Z",
    }
    client = HevyPrivateClient(_tokens(), http=http)
    assert client.account()["id"] == "account-1"
    assert http.request.call_count == 2


def test_routine_sync_accumulates_updates_and_deletes() -> None:
    client = HevyPrivateClient(_tokens())
    client.request = Mock(side_effect=[
        {"updated": [{"id": "a", "updated_at": "1"}], "deleted": [], "isMore": True},
        {"updated": [{"id": "b", "updated_at": "2"}], "deleted": ["a"], "isMore": False},
    ])
    assert client.routines() == [{"id": "b", "updated_at": "2"}]
    assert client.request.call_count == 2


def test_sync_rejects_malformed_provider_payload() -> None:
    client = HevyPrivateClient(_tokens())
    client.request = Mock(return_value={"updated": {}, "deleted": [], "isMore": False})
    with pytest.raises(HevyPrivateError, match="invalid sync"):
        client.workouts()
