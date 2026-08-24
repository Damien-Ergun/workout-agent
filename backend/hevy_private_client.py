"""Isolated client for Hevy's unsupported web-application protocol.

This adapter deliberately contains every private endpoint and header in one
place.  It is an observed, unstable protocol and is not Hevy's documented
developer API.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import unquote

import requests

BASE_URL = "https://api.hevyapp.com"
PRIVATE_API_KEY = "with_great_power"
AUTH_COOKIE_NAME = "auth2.0-token"
REQUEST_TIMEOUT = 15
MAX_SYNC_PAGES = 100


class HevyPrivateError(RuntimeError):
    """A safe, credential-free private API failure."""


@dataclass(frozen=True)
class HevyTokens:
    """Rotating Hevy web-session credentials."""

    access_token: str
    refresh_token: str
    expires_at: datetime

    @classmethod
    def from_cookie(cls, value: str) -> HevyTokens:
        """Parse the URL-encoded ``auth2.0-token`` cookie value."""
        try:
            payload = json.loads(unquote(value.strip()))
            access_token = str(payload["access_token"]).strip()
            refresh_token = str(payload["refresh_token"]).strip()
            expires_at = _parse_expiry(payload["expires_at"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HevyPrivateError("The Hevy session cookie is invalid.") from exc
        if not access_token or not refresh_token:
            raise HevyPrivateError("The Hevy session cookie is incomplete.")
        return cls(access_token, refresh_token, expires_at)


def _parse_expiry(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        # Accommodate seconds and JavaScript milliseconds.
        timestamp = float(value) / (1000 if float(value) > 10_000_000_000 else 1)
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


class HevyPrivateClient:
    """Small synchronous client with early refresh and one 401 retry."""

    def __init__(
        self,
        tokens: HevyTokens,
        *,
        on_tokens_rotated: Callable[[HevyTokens], None] | None = None,
        http: requests.Session | None = None,
    ) -> None:
        self.tokens = tokens
        self._on_tokens_rotated = on_tokens_rotated
        self.http = http or requests.Session()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.tokens.access_token}",
            "x-api-key": PRIVATE_API_KEY,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def refresh(self) -> None:
        response = self.http.post(
            f"{BASE_URL}/auth/refresh_token",
            headers=self._headers(),
            json={"refresh_token": self.tokens.refresh_token},
            timeout=REQUEST_TIMEOUT,
        )
        if not response.ok:
            raise HevyPrivateError("The Hevy session could not be refreshed.")
        try:
            payload = response.json()
            replacement = HevyTokens(
                access_token=str(payload["access_token"]),
                refresh_token=str(payload["refresh_token"]),
                expires_at=_parse_expiry(payload["expires_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HevyPrivateError("Hevy returned an invalid refresh response.") from exc
        self.tokens = replacement
        if self._on_tokens_rotated:
            self._on_tokens_rotated(replacement)

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        if self.tokens.expires_at <= datetime.now(timezone.utc) + timedelta(seconds=60):
            self.refresh()
        response = self.http.request(
            method,
            f"{BASE_URL}{path}",
            headers=self._headers(),
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
        if response.status_code == 401:
            self.refresh()
            response = self.http.request(
                method,
                f"{BASE_URL}{path}",
                headers=self._headers(),
                timeout=REQUEST_TIMEOUT,
                **kwargs,
            )
        if not response.ok:
            raise HevyPrivateError(f"Hevy request failed with status {response.status_code}.")
        try:
            return response.json()
        except ValueError as exc:
            raise HevyPrivateError("Hevy returned an invalid JSON response.") from exc

    def account(self) -> dict[str, Any]:
        payload = self.request("GET", "/account")
        if not isinstance(payload, dict):
            raise HevyPrivateError("Hevy returned an invalid account response.")
        return payload

    def workout_count(self) -> int | None:
        payload = self.request("GET", "/workout_count")
        value = payload.get("workout_count", payload.get("count")) if isinstance(payload, dict) else payload
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def sync_all(self, resource: str) -> list[dict[str, Any]]:
        """Read all workout or routine deltas, bounded against endless isMore."""
        if resource not in {"workouts", "routines"}:
            raise ValueError("Unsupported Hevy sync resource.")
        versions: dict[str, str] = {}
        records: dict[str, dict[str, Any]] = {}
        for _ in range(MAX_SYNC_PAGES):
            payload = self.request("POST", f"/{resource}_sync_batch", json=versions)
            if not isinstance(payload, dict):
                raise HevyPrivateError("Hevy returned an invalid sync response.")
            updated = payload.get("updated", [])
            deleted = payload.get("deleted", [])
            if not isinstance(updated, list) or not isinstance(deleted, list):
                raise HevyPrivateError("Hevy returned an invalid sync response.")
            for item in updated:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                identifier = str(item["id"])
                records[identifier] = item
                versions[identifier] = str(item.get("updated_at", ""))
            for item in deleted:
                identifier = str(item.get("id") if isinstance(item, dict) else item)
                records.pop(identifier, None)
                versions.pop(identifier, None)
            if not payload.get("isMore"):
                return list(records.values())
        raise HevyPrivateError("Hevy sync exceeded its safety page limit.")

    def routines(self) -> list[dict[str, Any]]:
        return self.sync_all("routines")

    def workouts(self) -> list[dict[str, Any]]:
        return self.sync_all("workouts")
