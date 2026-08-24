"""Internal progressive-overload web app (FastAPI).

A self-hosted dashboard for the workout agent. It reads the same SQLite database
the agent writes and renders it as a rich control centre: today's session and
progressive-overload targets, server-rendered SVG charts of every lift and your
body composition, all-time personal records, training-load trends, a consistency
calendar, the full periodisation plan, and programme check-ins.

Google OAuth login is available when WEB_AUTH_SECRET is configured, providing
user-scoped data isolation. When auth is disabled, it is read-only and meant to
sit behind a reverse proxy on a trusted host (e.g. Apache -> Docker ->
gym.example.com). All motivation is automated; nothing here calls out to an API
on a page view.

Run locally:   uvicorn webapp.app:app --reload
In a container: see Dockerfile.web / the `web` service in docker-compose.yml
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import analytics
import insights
import lifestyle
from ai_provider import AIProvider, resolve_provider
from authlib.integrations.starlette_client import OAuth
from config import Config
from connectors.base import ConnectorContext
from connectors.builtin import build_builtin_registry
from database import (
    clear_all_notifications,
    clear_chat_messages,
    delete_notification,
    delete_user_api_key,
    get_active_programme,
    get_body_metrics,
    get_chat_messages,
    get_checkins,
    get_daily_logs,
    get_dashboard_insight,
    get_exercise_volumes,
    get_meta,
    get_notifications,
    get_or_create_user,
    get_personal_records,
    get_programme_start_date,
    get_progress_history,
    get_reasoning_log,
    get_recent_bests,
    get_session_volumes,
    get_unread_notification_count,
    get_user_api_keys,
    get_user_preferences,
    init_db,
    mark_all_notifications_read,
    mark_notification_read,
    save_chat_message,
    save_notification,
    save_push_subscription,
    save_reasoning_log,
    save_user_api_key,
    save_user_preferences,
    set_active_programme,
    set_meta,
)
from dynamic_programme import (
    MAX_DURATION_WEEKS,
    MIN_DURATION_WEEKS,
    ProgrammeActivationRequest,
    ProgrammePreviewRequest,
    build_programme_preview,
    goal_options,
    serialise_hevy_source,
)
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from google_health_auth import build_authorize_url, exchange_code
from hevy_parser import normalise_name
from hevy_reader import HevyTrainingData
from program import (
    CYCLE_WEEKS,
    week_in_cycle,
)
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from webapp import ai_widgets, charts

DB_PATH = os.environ.get("DATABASE_PATH", "workout_agent.db").strip()
logger = logging.getLogger(__name__)


class HevySessionRequest(BaseModel):
    """URL-encoded value of Hevy's auth2.0-token cookie."""

    auth_cookie: str


@lru_cache(maxsize=1)
def _resolve_provider_for_request(request: Request, config: Config) -> AIProvider:
    user_id = request.session.get("user_id") if hasattr(request, "session") else None
    return resolve_provider(
        user_id=user_id,
        fallback_api_key=config.gemini_api_key,
        fallback_model=config.gemini_model,
    )


def get_config():
    return Config.load()


_RATE_LIMITS: dict[str, list[float]] = {}


def _check_rate_limit(request: Request, limit: int = 10, window: int = 60) -> None:
    now = time.time()
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    elif request.headers.get("x-real-ip"):
        ip = request.headers.get("x-real-ip").strip()  # type: ignore[union-attr]
    else:
        ip = request.client.host if request.client else "unknown"

    if ip not in _RATE_LIMITS:
        _RATE_LIMITS[ip] = []
    _RATE_LIMITS[ip] = [t for t in _RATE_LIMITS[ip] if now - t < window]
    if len(_RATE_LIMITS[ip]) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


WEB_GOOGLE_CLIENT_ID = os.environ.get("WEB_GOOGLE_CLIENT_ID", "").strip()
WEB_GOOGLE_CLIENT_SECRET = os.environ.get("WEB_GOOGLE_CLIENT_SECRET", "").strip()
WEB_AUTH_SECRET = os.environ.get("WEB_AUTH_SECRET", "").strip()
ALLOWED_EMAILS = [
    e.strip() for e in os.environ.get("ALLOWED_EMAILS", "").split(",") if e.strip()
]
ALLOW_ANONYMOUS_WEB = not bool(WEB_GOOGLE_CLIENT_ID)


def _check_api_auth(request: Request) -> str:
    user = request.session.get("user") if hasattr(request, "session") else None
    uid = request.session.get("user_id") if hasattr(request, "session") else None
    if uid:
        return uid
    if user and isinstance(user, dict) and user.get("id"):
        return user["id"]
    if ALLOW_ANONYMOUS_WEB:
        from database import get_legacy_user_id

        return get_legacy_user_id(DB_PATH)
    raise HTTPException(status_code=401, detail="Unauthorized")


# Google Health linking is opt-in: set the OAuth client in the web service's
# environment to enable the "Connect Google Health" button on the Settings page.
GH_CLIENT_ID = os.environ.get("GOOGLE_HEALTH_CLIENT_ID", "").strip()
GH_CLIENT_SECRET = os.environ.get("GOOGLE_HEALTH_CLIENT_SECRET", "").strip()
# Optional explicit redirect URI (recommended behind a reverse proxy so the
# scheme/host match exactly what is registered with Google). When unset it is
# derived from the incoming request.
GH_REDIRECT_URI = os.environ.get("GOOGLE_HEALTH_REDIRECT_URI", "").strip()
_GH_TOKEN_KEY = "google_health_refresh_token"
_GH_STATE_KEY = "google_health_oauth_state"

oauth = OAuth()
if WEB_GOOGLE_CLIENT_ID and WEB_GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=WEB_GOOGLE_CLIENT_ID,
        client_secret=WEB_GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

_BASE_DIR = Path(__file__).resolve().parent

# Cache-busting for static assets. Cloudflare (and browsers) cache /static/*
# aggressively, so a plain "/static/style.css" can serve a stale copy for hours
# after a deploy. We append a short content hash (?v=...) so every change yields
# a brand-new URL the cache has never seen, while unchanged files stay cached.
_ASSET_VERSIONS: dict[str, str] = {}


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Safe even if the agent already created the database (CREATE IF NOT EXISTS).
    init_db(DB_PATH)
    yield


# Determine frontend dist path
# Locally: root/frontend/dist/...
# Docker: /app/frontend/dist/...
if (_BASE_DIR.parent / "frontend/dist/frontend/browser").exists():
    FRONTEND_DIST = _BASE_DIR.parent / "frontend/dist/frontend/browser"  # Docker
else:
    FRONTEND_DIST = (
        _BASE_DIR.parent.parent / "frontend/dist/frontend/browser"
    )  # Local dev
app = FastAPI(title="Workout Agent", docs_url=None, redoc_url=None, lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "static")), name="static")
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST)), name="angular_assets")

# Enable CORS for external/remote frontend hosting
_cors_origins_raw = os.environ.get("ALLOWED_ORIGINS", "").strip()
_cors_origins = (
    [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    if _cors_origins_raw
    else [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:8770",
        "http://127.0.0.1:8770",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_AUTH_SECRET and WEB_GOOGLE_CLIENT_ID:

    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Any) -> Any:
            path = request.url.path
            # Allow static files, assets, JS/CSS/image/font assets, and auth endpoints
            if (
                path.startswith(("/static", "/assets"))
                or path.endswith(
                    (
                        ".js",
                        ".css",
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".svg",
                        ".ico",
                        ".woff",
                        ".woff2",
                        ".ttf",
                        ".map",
                        ".json",
                        ".webmanifest",
                    )
                )
                or path
                in [
                    "/login",
                    "/login/google",
                    "/logout",
                    "/auth",
                    "/google-health/callback",
                    "/favicon.ico",
                    "/sw.js",
                ]
            ):
                return await call_next(request)
            if not request.session.get("user"):
                # API routes: return 401 JSON instead of a redirect so fetch()
                # callers get a clear error instead of the login page HTML.
                if path.startswith("/api/"):
                    return HTMLResponse(
                        '{"detail":"Not authenticated"}',
                        status_code=401,
                        media_type="application/json",
                    )
                return RedirectResponse("/login")
            return await call_next(request)

    app.add_middleware(AuthMiddleware)

# Always mount SessionMiddleware so templates and routes can safely access
# request.session even when web auth is disabled. Without it, any access to
# request.session raises an AssertionError -> 500 Internal Server Error.
# When WEB_AUTH_SECRET is unset we fall back to a per-process random key; this
# is fine because there is nothing sensitive to protect when auth is off.
app.add_middleware(
    SessionMiddleware,
    secret_key=WEB_AUTH_SECRET or secrets.token_hex(32),
)


@app.get("/login/google")
async def login_google(request: Request) -> Any:
    if not WEB_GOOGLE_CLIENT_ID:
        return HTMLResponse("Web auth is not configured.", status_code=500)
    redirect_uri = str(request.url_for("auth"))
    if "localhost" not in redirect_uri and "127.0.0.1" not in redirect_uri:
        redirect_uri = redirect_uri.replace("http://", "https://")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/logout")
async def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse("/login")


@app.get("/api/me")
async def get_current_user_api(request: Request) -> Any:
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse({"user": user})


@app.get("/auth")
async def auth(request: Request) -> Any:
    if not WEB_GOOGLE_CLIENT_ID:
        return RedirectResponse("/")
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:  # noqa: BLE001
        logger.warning("OAuth token exchange failed: %s", e)
        return RedirectResponse("/login")
    user = token.get("userinfo")
    if user:
        email = user.get("email", "")
        if ALLOWED_EMAILS and email not in ALLOWED_EMAILS:
            return HTMLResponse(f"Unauthorized email: {email}", status_code=403)
        # Create or retrieve the user record on first login.
        user_record = get_or_create_user(
            email,
            display_name=user.get("name"),
            db_path=DB_PATH,
        )
        request.session["user"] = email
        request.session["user_id"] = user_record["id"]
    return RedirectResponse("/")


# Automated hype lines. One is shown per day, chosen deterministically from the
# date, so the dashboard always greets you without an API call or a button.
_QUOTES = [
    "Show up today and the bar gets lighter tomorrow. One more rep than last time.",
    "Strength is a skill you practise. Brace, pull, repeat.",
    "Small jumps, stacked weekly, become a competition physique. Keep stacking.",
    "Your future deadlift PR is built on the boring set you log today.",
    "Discipline now, six-pack later. Hit your protein and your top set.",
    "The last clean rep is where the growth lives. Earn it.",
    "Win the morning weigh-in by winning last night's sleep.",
    "Tight waist, wide back, heavy bar. Stay the course.",
    "You do not need motivation, you need the next set. Go.",
    "Consistency is the cheat code. You already have it.",
    "Add a kilo or add a rep. Either way, move forward.",
    "Recovery is training too. Eat, sleep, repeat.",
]


def _daily_quote(today: date | None = None) -> str:
    today = today or datetime.now(tz=timezone.utc).date()
    return _QUOTES[today.toordinal() % len(_QUOTES)]


def _epley_1rm(weight: float | None, reps: int | None) -> float | None:
    return analytics.epley_1rm(weight, reps)


def _weight_on_or_before(weights: list[tuple[str, float]], when: str) -> float | None:
    """Latest bodyweight recorded on or before a date (weights sorted ascending)."""
    result = None
    for when_w, value in weights:
        if when_w <= when:
            result = value
        else:
            break
    return result


def _find_lift_series(
    series: dict[str, Any], *keywords: str
) -> tuple[str | None, list[Any]]:
    """Find an exercise whose name contains all (then any) of the keywords."""
    for name, entries in series.items():
        low = name.lower()
        if all(k in low for k in keywords):
            return name, entries
    for name, entries in series.items():
        low = name.lower()
        if any(k in low for k in keywords):
            return name, entries
    return None, []


def _rep_top(rep_range: str) -> int | None:
    digits = [int(n) for n in rep_range.replace("-", " ").split() if n.isdigit()]
    return digits[-1] if digits else None


def _overload_nudge(planned_rep_range: str, best: dict[str, Any] | None) -> str:
    """Suggest the next progressive-overload step for an exercise."""
    if not best or best.get("top_reps") is None:
        return "Log this lift to start tracking progress."
    top = _rep_top(planned_rep_range)
    reps = best["top_reps"]
    weight = best.get("top_weight_kg")
    if top is not None and reps >= top:
        if weight:
            return "You hit the top of the range. Add 2.5 kg next time."
        return "You hit the top of the range. Add a rep or a little load."
    if top is not None:
        return f"Chase {top} clean reps, then add load."
    return "Keep the reps strict and progress when it feels easy."


def _format_best(best: dict[str, Any] | None) -> str:
    if not best:
        return "No data yet"
    weight = best.get("top_weight_kg")
    reps = best.get("top_reps")
    if weight is not None and reps is not None:
        return f"{weight:g} kg x {reps}"
    if reps is not None:
        return f"{reps} reps"
    return "No data yet"


def _training_levels(user_id: str | None = None) -> dict[str, int]:
    """Map ISO dates to a calendar-heatmap intensity (0-4)."""
    levels: dict[str, int] = {}
    for log in get_daily_logs(limit=400, db_path=DB_PATH, user_id=user_id):
        levels[log["date"]] = 2 if log["day"] is not None else 1
    # A session with logged sets is the strongest signal of a completed workout.
    for session in get_session_volumes(db_path=DB_PATH, user_id=user_id):
        levels[session["date"]] = 4
    return levels


def _current_streak(levels: dict[str, int]) -> int:
    """Count consecutive recent days that had any logged activity."""
    streak = 0
    day = datetime.now(tz=timezone.utc).date()
    # Allow today to be empty (the morning run may not have happened yet).
    if levels.get(day.isoformat(), 0) == 0:
        day -= timedelta(days=1)
    while levels.get(day.isoformat(), 0) > 0:
        streak += 1
        day -= timedelta(days=1)
    return streak


def _dashboard_context(
    today: date | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Build a dashboard from the active Hevy programme, or a setup state."""

    today = today or datetime.now(tz=timezone.utc).date()
    active = get_active_programme(user_id, db_path=DB_PATH) if user_id else None
    definition = active.get("definition") if active else None
    bests = get_recent_bests(DB_PATH, user_id=user_id)
    bests_norm = {normalise_name(name): best for name, best in bests.items()}

    setup_required = not bool(definition)
    rows: list[dict[str, Any]] = []
    focus = "Select routines from Hevy"
    current_day_number: int | None = None
    is_rest_day = True
    cycle_weeks = 0
    week = 0
    block_weeks = 1
    week_in_block = 0
    block: dict[str, Any] = {
        "number": 0,
        "name": "No active programme",
        "focus": "Connect Hevy, select routines and generate your programme.",
        "rep_emphasis": "No prescription yet",
        "target_rir": "",
        "progression": "",
    }

    if definition:
        spec = definition.get("programme_spec") or {}
        cycle_weeks = int(
            spec.get("duration_weeks") or definition.get("cycle_weeks") or 1
        )
        try:
            start = date.fromisoformat(
                str(
                    spec.get("start_date")
                    or get_programme_start_date(DB_PATH, user_id=user_id)
                )
            )
        except ValueError:
            start = get_programme_start_date(DB_PATH, user_id=user_id)
        elapsed_days = max(0, (today - start).days)
        week = min(cycle_weeks, elapsed_days // 7 + 1)

        for candidate in definition.get("blocks", []):
            start_week = int(candidate.get("start_week") or 1)
            end_week = int(candidate.get("end_week") or start_week)
            if start_week <= week <= end_week:
                block = dict(candidate)
                block_weeks = max(1, end_week - start_week + 1)
                week_in_block = week - start_week + 1
                break

        days = definition.get("days") or []
        if days:
            current_day_number = elapsed_days % len(days) + 1
            active_day = days[current_day_number - 1]
            focus = str(active_day.get("focus") or "Training")
            is_rest_day = False
            for exercise in active_day.get("exercises", []):
                name = str(exercise.get("name") or "?")
                rep_range = str(exercise.get("rep_range") or "?")
                best = bests_norm.get(normalise_name(name))
                rows.append(
                    {
                        "name": name,
                        "planned": (
                            exercise.get("scheme")
                            or f"{exercise.get('sets', 0)} × {rep_range}"
                        ),
                        "note": exercise.get("note", ""),
                        "last": _format_best(best),
                        "nudge": _overload_nudge(rep_range, best),
                        "role": exercise.get("role"),
                    }
                )

    metrics = get_body_metrics(db_path=DB_PATH, user_id=user_id)
    latest_weight = metrics[-1]["weight_kg"] if metrics else None
    recovery_like = {"weight_kg": latest_weight} if latest_weight else None
    guidance = lifestyle.daily_guidance(
        current_day_number,
        is_rest_day,
        recovery_like,
    )

    levels = _training_levels(user_id=user_id)
    streak = _current_streak(levels)
    weight_spark = charts.sparkline(
        [metric["weight_kg"] for metric in metrics if metric["weight_kg"]]
    )
    fat_spark = charts.sparkline(
        [metric["body_fat_pct"] for metric in metrics if metric["body_fat_pct"]],
        colour=charts.WARN,
    )
    latest_fat = next(
        (
            metric["body_fat_pct"]
            for metric in reversed(metrics)
            if metric["body_fat_pct"]
        ),
        None,
    )
    review = insights.build_insights(
        get_progress_history(db_path=DB_PATH, user_id=user_id),
        metrics,
        None,
    )

    cycle_progress = week / cycle_weeks * 100 if cycle_weeks else 0
    block_progress = week_in_block / block_weeks * 100 if week_in_block else 0
    block_label = str(block.get("name") or "Setup").split(" ")[0]

    return {
        "active": "today",
        "setup_required": setup_required,
        "active_programme_name": (definition.get("name") if definition else None),
        "quote": _daily_quote(today),
        "week": week,
        "cycle_weeks": cycle_weeks,
        "block": block,
        "week_in_block": week_in_block,
        "block_weeks": block_weeks,
        "focus": focus,
        "is_rest_day": is_rest_day,
        "weekday": today.strftime("%A"),
        "rows": rows,
        "lifestyle": guidance.as_lines(),
        "cycle_ring": charts.progress_ring(
            cycle_progress,
            label=f"Wk {week}" if week else "Setup",
            sub=f"of {cycle_weeks}" if cycle_weeks else "Hevy",
        ),
        "block_ring": charts.progress_ring(
            block_progress,
            label=block_label,
            sub=(
                f"wk {week_in_block}/{block_weeks}" if week_in_block else "not active"
            ),
            colour=charts.ACCENT_2,
        ),
        "streak": streak,
        "calendar": charts.calendar_heatmap(levels),
        "weight": f"{latest_weight:g} kg" if latest_weight else None,
        "weight_spark": weight_spark,
        "body_fat": f"{latest_fat:g} %" if latest_fat else None,
        "fat_spark": fat_spark,
        "review_headline": review.headline,
        "review_recovery": review.recovery.as_text(),
        "review_lifts": review.lifts,
        "dashboard_insight": get_dashboard_insight(db_path=DB_PATH),
    }


@app.get("/")
def dashboard(request: Request):
    return FileResponse(FRONTEND_DIST / "index.html")



@app.get("/api/checkins")
def api_checkins(request: Request):
    user_id = _check_api_auth(request)
    checkins = get_checkins(db_path=DB_PATH, user_id=user_id)
    return JSONResponse(
        jsonable_encoder({
            "checkins": checkins,
            "loading_svg": charts._empty_chart("Check-in cycle in progress", 300, 100) if not checkins else None
        })
    )


@app.get("/api/dashboard")
def api_dashboard(request: Request) -> JSONResponse:
    user_id = _check_api_auth(request)

    from fastapi.encoders import jsonable_encoder

    ctx = _dashboard_context(user_id=user_id)
    return JSONResponse(content=jsonable_encoder(ctx))


def _body_charts(*, user_id: str | None = None) -> dict[str, str | None]:
    readings = get_body_metrics(db_path=DB_PATH, user_id=user_id)

    def _series(key: str, unit: str, colour: str) -> str | None:
        points = [
            {"date": r["date"][5:], "value": r[key], "label": f"{r[key]:g} {unit}"}
            for r in readings
            if r[key] is not None
        ]
        if len(points) < 2:
            return None
        return charts.line_chart(points, unit=unit, colour=colour)

    return {
        "weight": _series("weight_kg", "kg", charts.ACCENT),
        "body_fat": _series("body_fat_pct", "%", charts.WARN),
        "muscle": _series("muscle_pct", "%", charts.ACCENT_2),
        "resting_hr": _series("resting_hr", "bpm", charts.PINK),
    }


def _project_lift(
    label: str, entries: list[Any], target_ordinal: int, *, metric: str = "auto"
) -> dict[str, Any] | None:
    """Build a projection card for a lift at the end of the cycle."""
    points: list[tuple[float, float]] = []
    unit = "kg"
    use_e1rm = metric == "e1rm"
    if metric == "auto":
        use_e1rm = any(e["top_weight_kg"] for e in entries)
    for e in entries:
        value = (
            analytics.epley_1rm(e["top_weight_kg"], e["top_reps"])
            if use_e1rm
            else e["top_reps"]
        )
        if value is None:
            continue
        points.append((float(date.fromisoformat(e["date"]).toordinal()), float(value)))
    if len(points) < 2:
        return None
    if not use_e1rm:
        unit = "reps"
    current = points[-1][1]
    projected = analytics.project(points, target_ordinal)
    if projected is None:
        return None
    projected = max(projected, current)  # never project a regression below today
    return {
        "label": label,
        "current": f"{current:g} {unit}",
        "projected": f"{projected:g} {unit}",
        "metric": "est. 1RM" if use_e1rm else "top reps",
        "gain": round(projected - current, 1),
    }


def _block_lift_str(lift: dict | None) -> str:
    """Format a lift dict as 'sets x rep_range' or empty string."""
    if not lift:
        return ""
    sets = lift.get("sets", 0)
    rep_range = lift.get("rep_range", "")
    if not sets or not rep_range:
        return ""
    return f"{sets} x {rep_range}"


@app.post("/api/programmes/preview")
def preview_programme(
    payload: ProgrammePreviewRequest,
    request: Request,
) -> JSONResponse:
    """Generate a deterministic preview from selected Hevy routines."""

    _check_rate_limit(request, limit=5)
    user_id = _check_api_auth(request)
    training_data = _load_hevy_training_for_user(user_id)
    try:
        preview = build_programme_preview(training_data, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return JSONResponse(jsonable_encoder({"preview": preview}))


@app.post("/api/programmes/activate")
def activate_programme(
    payload: ProgrammeActivationRequest,
    request: Request,
) -> JSONResponse:
    """Reconfirm Hevy source data and activate the generated programme."""

    _check_rate_limit(request, limit=5)
    user_id = _check_api_auth(request)
    training_data = _load_hevy_training_for_user(user_id)
    preview_request = ProgrammePreviewRequest.model_validate(
        payload.model_dump(exclude={"preview_token"})
    )
    try:
        preview = build_programme_preview(training_data, preview_request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if preview["preview_token"] != payload.preview_token:
        raise HTTPException(
            status_code=409,
            detail=(
                "The Hevy routines or programme settings changed after preview. "
                "Generate a fresh preview before activation."
            ),
        )

    set_active_programme(
        user_id,
        source="hevy",
        template_key=f"hevy:{preview['preview_token'][:20]}",
        definition=preview,
        db_path=DB_PATH,
    )
    set_meta(
        "programme_start_date",
        str(preview["programme_spec"]["start_date"]),
        DB_PATH,
        user_id=user_id,
    )
    save_notification(
        user_id,
        title="Hevy programme activated",
        message=(
            f"{preview['name']} is active for "
            f"{preview['cycle_weeks']} weeks using "
            f"{preview['total_days']} selected routines."
        ),
        type="programme",
        link="/plan",
        db_path=DB_PATH,
    )

    return JSONResponse(
        jsonable_encoder(
            {
                "status": "ok",
                "active_programme": get_active_programme(
                    user_id,
                    db_path=DB_PATH,
                ),
            }
        )
    )


@app.post("/api/programmes/select")
async def select_programme(request: Request) -> JSONResponse:
    """Compatibility adapter for clients using the retired template endpoint."""

    _check_rate_limit(request, limit=5)
    user_id = _check_api_auth(request)
    try:
        body = await request.json()
    except (json.JSONDecodeError, TypeError):
        body = {}

    template_key = str(body.get("template_key") or "").strip()
    if template_key == "hybrid_powerbuilding":
        raise HTTPException(
            status_code=410,
            detail=(
                "Hybrid Powerbuilding has been removed. "
                "Select routines from Hevy and generate a programme preview."
            ),
        )
    if template_key != "infer_from_hevy":
        raise HTTPException(
            status_code=410,
            detail=(
                "Static programme selection has been removed. "
                "Use /api/programmes/preview and /api/programmes/activate."
            ),
        )

    training_data = _load_hevy_training_for_user(user_id)
    if not training_data.routines:
        raise HTTPException(
            status_code=422,
            detail="No Hevy routines are available for programme generation.",
        )

    try:
        preview_request = ProgrammePreviewRequest(
            selected_routine_ids=[routine.id for routine in training_data.routines],
            duration_weeks=int(body.get("duration_weeks") or 12),
            goal=body.get("goal") or "general_fitness",
            start_date=body.get("start_date") or datetime.now(tz=timezone.utc).date(),
            sessions_per_week=body.get("sessions_per_week"),
            experience=body.get("experience") or "intermediate",
            max_session_minutes=body.get("max_session_minutes"),
            adaptation_aggressiveness=(
                body.get("adaptation_aggressiveness") or "balanced"
            ),
        )
        preview = build_programme_preview(training_data, preview_request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    set_active_programme(
        user_id,
        source="hevy",
        template_key=f"hevy:{preview['preview_token'][:20]}",
        definition=preview,
        db_path=DB_PATH,
    )
    set_meta(
        "programme_start_date",
        str(preview["programme_spec"]["start_date"]),
        DB_PATH,
        user_id=user_id,
    )
    return JSONResponse(
        jsonable_encoder(
            {
                "status": "ok",
                "source": "hevy",
                "preview_token": preview["preview_token"],
            }
        )
    )


def _load_hevy_training_for_user(
    user_id: str,
    *,
    workout_limit: int = 56,
) -> HevyTrainingData:
    """Fetch the current user's Hevy source data or raise a useful API error."""

    keys = get_user_api_keys(user_id, db_path=DB_PATH)
    hevy_key = keys.get("hevy", {}).get("api_key", "").strip()
    private_record = keys.get("hevy_session", {})
    if not hevy_key and not private_record.get("api_key"):
        raise HTTPException(
            status_code=400,
            detail="No Hevy connection configured. Connect a web session or add an API key in Settings.",
        )

    from hevy_reader import fetch_private_user_training, fetch_user_training

    try:
        if private_record.get("api_key"):
            from hevy_private_client import HevyPrivateClient, HevyTokens, _parse_expiry

            extra = private_record.get("extra") or {}

            def persist_tokens(tokens: HevyTokens) -> None:
                save_user_api_key(
                    user_id,
                    "hevy_session",
                    tokens.access_token,
                    refresh_token=tokens.refresh_token,
                    extra={"expires_at": tokens.expires_at.isoformat()},
                    db_path=DB_PATH,
                )

            tokens = HevyTokens(
                access_token=private_record["api_key"],
                refresh_token=private_record.get("refresh_token") or "",
                expires_at=_parse_expiry(extra.get("expires_at")),
            )
            return fetch_private_user_training(
                HevyPrivateClient(tokens, on_tokens_rotated=persist_tokens),
                workout_limit=workout_limit,
            )
        return fetch_user_training(
            hevy_key,
            workout_limit=workout_limit,
        )
    except Exception as exc:
        logger.exception(
            "Could not fetch Hevy source data for user %s.",
            user_id,
        )
        raise HTTPException(
            status_code=502,
            detail=("Hevy could not be reached. Check the connection and try again."),
        ) from exc


@app.get("/api/xai_reasoning/{context_id}")
def xai_reasoning(context_id: str, request: Request) -> dict[str, Any]:
    _check_rate_limit(request)
    user_id = request.session.get("user_id")

    existing = get_reasoning_log(context_id, db_path=DB_PATH, user_id=user_id)
    if existing:
        return {"reasoning": existing}

    parts = context_id.split("_", 1)
    if len(parts) != 2:
        return {"reasoning": "Invalid context ID"}

    when, ex_name = parts

    config = get_config()
    provider = _resolve_provider_for_request(request, config)

    history = get_progress_history(db_path=DB_PATH, user_id=user_id).get(ex_name, [])

    prompt = f"Why did my volume/performance change for {ex_name} around {when}? Here is my history: {json.dumps(history)}. Provide a clear causal explanation in a few sentences."
    try:
        response = provider.generate(prompt)
        response_text = response if isinstance(response, str) else "".join(response)
        reasoning = (response_text or "Could not determine reasoning.").strip()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Error in xai_reasoning: {exc}")
        reasoning = "Could not determine reasoning."

    save_reasoning_log(context_id, ex_name, reasoning, db_path=DB_PATH, user_id=user_id)
    return {"reasoning": reasoning}


@app.get("/api/project_peak")
def project_peak(request: Request) -> dict[str, Any]:
    _check_rate_limit(request, limit=5)
    user_id = request.session.get("user_id")
    config = get_config()
    provider = _resolve_provider_for_request(request, config)

    series = get_progress_history(db_path=DB_PATH, user_id=user_id)
    dl_entries = series.get("Deadlift", [])
    pu_entries = series.get("Pull-ups", [])

    prompt = f"Analyze this historical progression for Deadlift: {json.dumps(dl_entries)} and Pull-ups: {json.dumps(pu_entries)}. Project the estimated 1RM at the end of the 12-week peaking phase. Adjust the forecast curve if recent sessions look 'bad'. Return JSON: {{'Deadlift_Projected': float, 'Pullups_Projected': float, 'Validation': 'string explanation'}}"
    try:
        response = provider.generate(prompt)
        text = (response if isinstance(response, str) else "".join(response)).strip()
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```")
        return json.loads(text.strip())
    except Exception:  # noqa: BLE001
        return {"error": "Failed to project peak."}


@app.get("/api/chat/history")
def chat_history(request: Request):
    user_id = request.session.get("user_id")
    return get_chat_messages(limit=50, db_path=DB_PATH, user_id=user_id)


@app.post("/api/chat/clear")
def chat_clear(request: Request):
    user_id = request.session.get("user_id")
    clear_chat_messages(db_path=DB_PATH, user_id=user_id)
    return {"status": "ok"}


@app.get("/api/rag_search", response_model=None)
def rag_search(request: Request, q: str = Query(...)) -> Any:
    _check_rate_limit(request, limit=15)
    user_id = request.session.get("user_id")
    config = get_config()
    provider = _resolve_provider_for_request(request, config)

    # Gather training context
    user_id = request.session.get("user_id")
    logs = get_daily_logs(limit=30, db_path=DB_PATH, user_id=user_id)
    history = get_progress_history(db_path=DB_PATH, user_id=user_id)
    biometrics = get_body_metrics(db_path=DB_PATH, user_id=user_id)
    prs = get_personal_records(db_path=DB_PATH)

    context = json.dumps(
        {
            "logs": logs,
            "history": history,
            "biometrics": biometrics[-10:],
            "prs": prs[:10],
        },
        default=str,
    )

    # Build multi-turn conversation from chat history
    chat_history_msgs = get_chat_messages(limit=20, db_path=DB_PATH, user_id=user_id)
    conversation_lines = []
    for msg in chat_history_msgs:
        role_label = "User" if msg["role"] == "user" else "Coach"
        conversation_lines.append(f"{role_label}: {msg['content']}")
    conversation_text = (
        "\n".join(conversation_lines) if conversation_lines else "No previous messages."
    )

    # Save user message
    save_chat_message("user", q, db_path=DB_PATH, user_id=user_id)

    prompt = f"""You are Coach, an elite powerbuilding AI coach embedded in Elgan's training dashboard.
You have full access to his training logs, body composition data, personal records, and programme history.

Your personality:
- Knowledgeable, direct, and encouraging. Like a trusted coach who knows the data.
- Reference specific numbers, dates, and exercises from the context when relevant.
- Keep answers concise but insightful. You can use **bold** for emphasis and bullet lists for structure.
- Use British English.

Conversation so far:
{conversation_text}

Training data context (recent logs, progress history, biometrics, PRs):
{context[:25000]}

User's new message: {q}

Respond naturally as Coach. If the question is about their training data, reference the actual numbers. If it is a general fitness question, answer from expertise but relate it back to their programme where possible."""

    def generate():
        collected: list[str] = []
        try:
            response = provider.generate(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    collected.append(chunk.text)
                    yield chunk.text
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error during Gemini streaming: {e}")
            error_msg = "Sorry, Coach is currently unavailable or encountered an error. Please try again."
            collected.append(error_msg)
            yield error_msg
        finally:
            # Save the full assistant response
            full_response = "".join(collected)
            if full_response:
                save_chat_message(
                    "assistant", full_response, db_path=DB_PATH, user_id=user_id
                )

    return StreamingResponse(generate(), media_type="text/plain")


def _gh_redirect_uri(request: Request) -> str:
    """The OAuth redirect URI, explicit env override or derived from the request."""
    return GH_REDIRECT_URI or str(request.url_for("google_health_callback"))


@app.post("/api/settings/key")
async def save_api_key(request: Request) -> dict[str, str]:
    """Save or update an API key for the current user."""
    _check_rate_limit(request, limit=5)
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    body = await request.json()
    provider = body.get("provider", "").strip().lower()
    api_key = body.get("api_key", "").strip()

    if not provider or not api_key:
        raise HTTPException(status_code=400, detail="Provider and api_key are required")

    valid_providers = {"hevy", "gemini", "claude", "openai", "deepseek"}
    if provider not in valid_providers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider '{provider}'. Use: {', '.join(valid_providers)}",
        )

    # Optional model override for AI providers.
    extra = {}
    model = body.get("model", "").strip()
    if model:
        extra["model"] = model

    save_user_api_key(
        user_id,
        provider,
        api_key,
        extra=extra or None,
        db_path=DB_PATH,
    )
    return {"status": "ok", "provider": provider}


@app.post("/api/settings/key/delete")
async def remove_api_key(request: Request) -> dict[str, str]:
    """Remove a stored API key for the current user."""
    _check_rate_limit(request, limit=5)
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    body = await request.json()
    provider = body.get("provider", "").strip().lower()
    if not provider:
        raise HTTPException(status_code=400, detail="Provider is required")

    delete_user_api_key(user_id, provider, db_path=DB_PATH)
    return {"status": "ok", "provider": provider}


@app.post("/api/settings/verify-hevy")
async def verify_hevy_key(request: Request) -> dict[str, Any]:
    """Test a Hevy API key by calling /v1/workouts/count."""
    _check_rate_limit(request, limit=5)
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    body = await request.json()
    api_key = body.get("api_key", "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")

    from hevy_client import get_workout_count

    count = get_workout_count(api_key)
    if count is not None:
        return {"status": "ok", "workout_count": count}
    return {
        "status": "error",
        "detail": "Could not connect to Hevy. Check the API key.",
    }


@app.post("/api/settings/hevy-session")
def save_hevy_session(
    payload: HevySessionRequest,
    request: Request,
) -> dict[str, Any]:
    """Validate and securely persist an unofficial Hevy web session."""
    _check_rate_limit(request, limit=3)
    user_id = _check_api_auth(request)

    from hevy_private_client import HevyPrivateClient, HevyPrivateError, HevyTokens

    try:
        tokens = HevyTokens.from_cookie(payload.auth_cookie)
        client = HevyPrivateClient(tokens)
        account = client.account()
    except HevyPrivateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # A request may rotate credentials; persist only the client's final triplet.
    tokens = client.tokens
    save_user_api_key(
        user_id,
        "hevy_session",
        tokens.access_token,
        refresh_token=tokens.refresh_token,
        extra={"expires_at": tokens.expires_at.isoformat()},
        db_path=DB_PATH,
    )
    return {
        "status": "ok",
        "username": account.get("username"),
        "detail": "Hevy web session connected. This unsupported protocol may change without notice.",
    }


@app.post("/api/settings/preferences")
async def save_preferences(request: Request) -> dict[str, str]:
    """Save user training preferences."""
    _check_rate_limit(request, limit=5)
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    body = await request.json()
    save_user_preferences(
        user_id,
        goals=body.get("goals"),
        constraints=body.get("constraints"),
        experience_level=body.get("experience_level"),
        coaching_style=body.get("coaching_style"),
        preferred_ai=body.get("preferred_ai"),
        ai_model=body.get("ai_model"),
        custom_rules=body.get("custom_rules"),
        db_path=DB_PATH,
    )
    return {"status": "ok"}


@app.post("/api/settings/push-subscribe")
async def save_push_subscription_route(request: Request) -> dict[str, str]:
    """Save a push subscription for the current user."""
    _check_rate_limit(request, limit=5)
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    body = await request.json()
    endpoint = body.get("endpoint")
    keys = body.get("keys", {})
    p256dh = keys.get("p256dh")
    auth = keys.get("auth")

    if not endpoint or not p256dh or not auth:
        raise HTTPException(status_code=400, detail="Invalid push subscription data")

    save_push_subscription(
        user_id, endpoint=endpoint, p256dh=p256dh, auth=auth, db_path=DB_PATH
    )
    return {"status": "ok"}


@app.get("/google-health/connect")
def google_health_connect(request: Request) -> RedirectResponse:
    """Start the Google Health OAuth flow and redirect to the consent screen."""
    _check_rate_limit(request, limit=5)
    if not (GH_CLIENT_ID and GH_CLIENT_SECRET):
        return RedirectResponse("/settings?gh=unconfigured", status_code=303)
    user_id = request.session.get("user_id")
    state = secrets.token_urlsafe(16)
    set_meta(_GH_STATE_KEY, state, DB_PATH, user_id=user_id)
    url = build_authorize_url(
        GH_CLIENT_ID,
        state,
        redirect_uri=_gh_redirect_uri(request),
    )
    return RedirectResponse(url, status_code=303)


@app.get("/google-health/callback", name="google_health_callback")
def google_health_callback(request: Request) -> RedirectResponse:
    """Receive Google's redirect, swap the code for a refresh token, store it."""
    if not (GH_CLIENT_ID and GH_CLIENT_SECRET):
        return RedirectResponse("/settings?gh=unconfigured", status_code=303)
    if request.query_params.get("error"):
        return RedirectResponse("/settings?gh=denied", status_code=303)
    user_id = request.session.get("user_id")
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    expected = get_meta(_GH_STATE_KEY, DB_PATH, user_id=user_id)
    set_meta(_GH_STATE_KEY, "", DB_PATH, user_id=user_id)  # one-time use
    if not code or not state or not expected or state != expected:
        return RedirectResponse("/settings?gh=error", status_code=303)
    tokens = exchange_code(
        GH_CLIENT_ID,
        GH_CLIENT_SECRET,
        code,
        redirect_uri=_gh_redirect_uri(request),
    )
    if not tokens or not tokens.get("refresh_token"):
        return RedirectResponse("/settings?gh=error", status_code=303)
    set_meta(_GH_TOKEN_KEY, tokens["refresh_token"], DB_PATH, user_id=user_id)
    return RedirectResponse("/settings?gh=connected", status_code=303)


@app.post("/google-health/disconnect")
def google_health_disconnect(request: Request) -> RedirectResponse:
    """Forget the stored refresh token so the agent stops syncing."""
    _check_rate_limit(request, limit=5)
    user_id = request.session.get("user_id")
    set_meta(_GH_TOKEN_KEY, "", DB_PATH, user_id=user_id)
    return RedirectResponse("/settings?gh=disconnected", status_code=303)


@app.get("/sw.js", include_in_schema=False)
def service_worker() -> FileResponse:
    """Serve the service worker from the root so it can control every page.

    A worker only controls URLs within its own path, so it must be served from
    "/" (with the Service-Worker-Allowed header) rather than from /static.
    """
    return FileResponse(
        _BASE_DIR / "static" / "sw.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )


# --- AUTO-GENERATED API ROUTES FOR ANGULAR ---
from fastapi.encoders import jsonable_encoder


@app.get("/api/progress")
def api_progress(request: Request):
    user_id = _check_api_auth(request)
    series = get_progress_history(db_path=DB_PATH, user_id=user_id)
    charts_data = []
    for name in sorted(series):
        entries = series[name]
        points = [
            {
                "date": e["date"][5:],
                "value": e["top_weight_kg"],
                "label": _format_best(e),
            }
            for e in entries
        ]
        e1rms = [_epley_1rm(e["top_weight_kg"], e["top_reps"]) for e in entries]
        e1rms_filtered = [v for v in e1rms if v is not None]
        best_e1rm = max(e1rms_filtered) if e1rms_filtered else None
        charts_data.append(
            {
                "name": name,
                "svg": charts.line_chart(points, unit="kg"),
                "best_e1rm": best_e1rm,
                "sessions": len(entries),
            }
        )
    return JSONResponse(
        jsonable_encoder({"charts": charts_data, "body": _body_charts(user_id=user_id)})
    )


@app.get("/api/stats")
def api_stats(request: Request):
    user_id = _check_api_auth(request)
    volumes = get_session_volumes(db_path=DB_PATH, user_id=user_id)
    prs = get_personal_records(db_path=DB_PATH, user_id=user_id)
    logs = get_daily_logs(limit=400, db_path=DB_PATH, user_id=user_id)
    start = get_programme_start_date(db_path=DB_PATH, user_id=user_id)
    series = get_progress_history(db_path=DB_PATH, user_id=user_id)
    today = datetime.now(tz=timezone.utc).date()
    week = week_in_cycle(start, today)

    total_sessions = len(volumes)
    total_volume = sum(v["volume"] for v in volumes)
    days_on_programme = (today - start).days

    focus_counts: dict[str, int] = {}
    for log in logs:
        f = log.get("focus") if isinstance(log, dict) else getattr(log, "focus", None)
        d = log.get("day") if isinstance(log, dict) else getattr(log, "day", None)
        if d is not None and f:
            focus_counts[f] = focus_counts.get(f, 0) + 1
    distribution = (
        charts.donut(
            [{"label": k, "value": v} for k, v in sorted(focus_counts.items())]
        )
        if focus_counts
        else None
    )

    groups = analytics.group_volumes(
        get_exercise_volumes(db_path=DB_PATH, user_id=user_id),
    )
    muscle_donut = (
        charts.donut(
            [
                {"label": g, "value": v}
                for g, v in sorted(groups.items(), key=lambda kv: -kv[1])
            ]
        )
        if groups
        else None
    )

    recent = volumes[-14:]
    volume_bars = (
        charts.bar_chart(
            [
                {"label": v["date"][5:], "value": v["volume"], "caption": v["date"]}
                for v in recent
            ],
            unit="kg",
        )
        if recent
        else None
    )

    biometrics = get_body_metrics(db_path=DB_PATH, user_id=user_id)
    block_phase_svg = ai_widgets.block_phase_tracker(volumes)
    recovery_grid_svg = ai_widgets.systemic_recovery_correlation(biometrics, volumes)
    vol_dist_svg = ai_widgets.volume_distribution(groups)

    weights = [(m["date"], m["weight_kg"]) for m in biometrics if m["weight_kg"]]
    _, dl_entries = _find_lift_series(series, "deadlift")
    dots_points, ratio_points = [], []
    for e in dl_entries:
        e1rm = analytics.epley_1rm(e["top_weight_kg"], e["top_reps"])
        bw = _weight_on_or_before(weights, e["date"])
        if not e1rm or not bw:
            continue
        score = analytics.dots_score(bw, e1rm)
        if score:
            dots_points.append(
                {"date": e["date"][5:], "value": score, "label": f"{score:g}"}
            )
        ratio_points.append(
            {
                "date": e["date"][5:],
                "value": round(e1rm / bw, 2),
                "label": f"{e1rm / bw:.2f}x",
            }
        )
    dots_chart = (
        charts.line_chart(dots_points, colour=charts.PURPLE)
        if len(dots_points) > 1
        else None
    )
    ratio_chart = (
        charts.line_chart(ratio_points, unit="x", colour=charts.ACCENT_2)
        if len(ratio_points) > 1
        else None
    )

    target_ordinal = today.toordinal() + max(0, (CYCLE_WEEKS - week)) * 7
    projections = [
        _project_lift("Deadlift", dl_entries, target_ordinal, metric="e1rm"),
        _project_lift("Pull-ups", _find_lift_series(series, "pull")[1], target_ordinal),
    ]
    projections = [p for p in projections if p]

    pr_rows = [
        {
            "exercise": pr["exercise"],
            "e1rm": f"{pr['e1rm']:g}",
            "detail": f"{pr['weight_kg']:g} kg x {pr['reps']}",
            "date": pr["date"],
        }
        for pr in prs
    ]

    return JSONResponse(
        jsonable_encoder(
            {
                "total_sessions": total_sessions,
                "total_volume": total_volume,
                "days_on_programme": days_on_programme,
                "exercises_tracked": len(series),
                "has_distribution": bool(focus_counts),
                "distribution": distribution,
                "has_muscle": bool(groups),
                "muscle_donut": muscle_donut,
                "has_volume": bool(recent),
                "volume_bars": volume_bars,
                "block_phase_svg": block_phase_svg,
                "recovery_grid_svg": recovery_grid_svg,
                "vol_dist_svg": vol_dist_svg,
                "dots_chart": dots_chart,
                "ratio_chart": ratio_chart,
                "projections": projections,
                "prs": pr_rows,
            }
        )
    )


@app.get("/api/history")
def api_history(request: Request):
    user_id = _check_api_auth(request)
    logs = get_daily_logs(limit=30, db_path=DB_PATH, user_id=user_id)
    return JSONResponse(jsonable_encoder({"logs": logs}))


@app.get("/api/plan")
def api_plan(request: Request) -> JSONResponse:
    """Render only an activated Hevy-native programme; never a static fallback."""

    user_id = _check_api_auth(request)
    active = get_active_programme(user_id, db_path=DB_PATH)
    raw_definition = active.get("definition") if active else None
    definition: dict[str, Any] = (
        raw_definition if isinstance(raw_definition, dict) else {}
    )
    source = active.get("source") if active else None

    if source != "hevy" or not definition:
        return JSONResponse(
            jsonable_encoder(
                {
                    "setup_required": True,
                    "is_active_programme": False,
                    "source": None,
                    "active_programme": None,
                    "split_name": "No active programme",
                    "week": 0,
                    "cycle_weeks": 0,
                    "current_block": None,
                    "blocks": [],
                    "days": [],
                    "rules": [],
                    "analysis": {},
                    "warnings": [],
                    "programme_spec": {},
                }
            )
        )

    spec = definition.get("programme_spec") or {}
    cycle_weeks = max(
        1,
        int(spec.get("duration_weeks") or definition.get("cycle_weeks") or 1),
    )
    try:
        start_date = date.fromisoformat(str(spec.get("start_date")))
    except (TypeError, ValueError):
        start_date = get_programme_start_date(DB_PATH, user_id=user_id)

    today = datetime.now(tz=timezone.utc).date()
    elapsed_days = max(0, (today - start_date).days)
    week = min(cycle_weeks, elapsed_days // 7 + 1)

    blocks: list[dict[str, Any]] = []
    current_block: dict[str, Any] | None = None
    for raw_block in definition.get("blocks") or []:
        block = dict(raw_block)
        start_week = int(block.get("start_week") or 1)
        end_week = int(block.get("end_week") or start_week)
        block["is_current"] = start_week <= week <= end_week
        if block["is_current"]:
            current_block = block
        blocks.append(block)

    return JSONResponse(
        jsonable_encoder(
            {
                "setup_required": False,
                "is_active_programme": True,
                "source": "hevy",
                "active_programme": active,
                "split_name": (
                    definition.get("name")
                    or (active or {}).get("name")
                    or "Hevy programme"
                ),
                "week": week,
                "cycle_weeks": cycle_weeks,
                "current_block": (current_block.get("name") if current_block else None),
                "blocks": blocks,
                "days": definition.get("days") or [],
                "rules": definition.get("rules") or [],
                "analysis": definition.get("analysis") or {},
                "warnings": definition.get("warnings") or [],
                "programme_spec": spec,
            }
        )
    )


@app.get("/api/programmes")
def api_programmes(request: Request) -> JSONResponse:
    """Return the user's live Hevy routine library and active programme."""

    user_id = _check_api_auth(request)
    active = get_active_programme(user_id, db_path=DB_PATH)
    keys = get_user_api_keys(user_id, db_path=DB_PATH)
    hevy_key = keys.get("hevy", {}).get("api_key", "").strip()
    has_private_session = bool(keys.get("hevy_session", {}).get("api_key"))

    empty_source: dict[str, Any] = {
        "username": None,
        "workout_count": 0,
        "routine_count": 0,
        "recent_workout_count": 0,
        "folders": [],
        "routines": [],
    }
    connection: dict[str, Any]
    source = empty_source

    if not hevy_key and not has_private_session:
        connection = {
            "state": "disconnected",
            "detail": "Connect a Hevy web session or add an API key in Settings to import routines.",
            "username": None,
        }
    else:
        try:
            training_data = _load_hevy_training_for_user(user_id)
            source = serialise_hevy_source(training_data)
            connection = {
                "state": "connected",
                "detail": None,
                "username": training_data.username,
            }
        except HTTPException as exc:
            connection = {
                "state": "error",
                "detail": str(exc.detail),
                "username": None,
            }

    return JSONResponse(
        jsonable_encoder(
            {
                "active_programme": active,
                "current_state": active,
                "templates": [],
                "connection": connection,
                "source": source,
                "duration": {
                    "minimum_weeks": MIN_DURATION_WEEKS,
                    "maximum_weeks": MAX_DURATION_WEEKS,
                    "default_weeks": 12,
                },
                "goals": goal_options(),
            }
        )
    )


@app.get("/api/settings")
def api_settings(request: Request):
    user_id = _check_api_auth(request)

    prefs = get_user_preferences(user_id, DB_PATH)
    keys = get_user_api_keys(
        user_id, DB_PATH
    )  # dict mapping provider_name -> record dict

    user_keys = {}
    known_providers = ["hevy", "gemini", "claude", "openai", "deepseek"]
    for p in known_providers:
        k_data = keys.get(p, {})
        key_str = k_data.get("api_key", "")
        if key_str:
            masked = f"••••••••{key_str[-4:]}" if len(key_str) >= 4 else "••••••••"
            user_keys[p] = {"has_key": True, "masked": masked}
        else:
            user_keys[p] = {"has_key": False, "masked": None}
    user_keys["hevy_session"] = {
        "has_key": bool(keys.get("hevy_session", {}).get("api_key")),
        "masked": None,
    }

    ai_providers = [
        {"id": "gemini", "name": "Google Gemini", "default_model": "gemini-2.5-flash"},
        {
            "id": "claude",
            "name": "Anthropic Claude",
            "default_model": "claude-3-5-sonnet-20241022",
        },
        {"id": "openai", "name": "OpenAI", "default_model": "gpt-4o"},
        {"id": "deepseek", "name": "DeepSeek", "default_model": "deepseek-chat"},
    ]

    gh_connected = bool(get_meta(_GH_TOKEN_KEY, DB_PATH, user_id=user_id))
    gh_configured = bool(GH_CLIENT_ID and GH_CLIENT_SECRET)
    gh_status = request.query_params.get("gh")

    registry = build_builtin_registry()
    ctx = ConnectorContext(user_id=user_id)
    connectors_info = []
    for connector in registry.all():
        try:
            status = connector.status(ctx)
            if status.state == "pending":
                continue
            connectors_info.append({
                "provider": connector.provider,
                "name": connector.provider.title().replace("_", " "),
                "state": status.state,
                "message": status.message,
                "authorize_supported": connector.capabilities.authorize,
            })
        except Exception as exc:  # noqa: BLE001 - isolate optional connectors
            logger.warning("Could not read connector status for %s: %s", connector.provider, exc)

    return JSONResponse(
        jsonable_encoder(
            {
                "user_prefs": prefs,
                "user_keys": user_keys,
                "ai_providers": ai_providers,
                "gh_connected": gh_connected,
                "gh_configured": gh_configured,
                "gh_status": gh_status,
                "connectors_info": connectors_info,
                "vapid_public_key": os.environ.get("VAPID_PUBLIC_KEY"),
                "key_status": request.query_params.get("key_status"),
                "pref_status": request.query_params.get("pref_status"),
            }
        )
    )


@app.post("/api/settings/sync-history")
async def sync_history_endpoint(request: Request) -> dict[str, Any]:
    """Rebuild user's workout history from Hevy."""
    _check_rate_limit(request, limit=2)
    user_id = _check_api_auth(request)

    keys = get_user_api_keys(user_id, DB_PATH)
    hevy_key = keys.get("hevy", {}).get("api_key", "").strip()
    if not hevy_key:
        raise HTTPException(
            status_code=400,
            detail="No Hevy API key configured. Save your Hevy API key in Settings first.",
        )

    from scripts.sync_history import sync_all

    result = sync_all(hevy_key, DB_PATH, user_id=user_id)
    if "error" in result:
        err_msg = str(result["error"])
        save_notification(
            user_id,
            title="⚠️ Hevy Sync Failed",
            message=f"Could not sync history: {err_msg}",
            type="sync",
            link="/settings",
            db_path=DB_PATH,
        )
        raise HTTPException(status_code=400, detail=err_msg)

    workouts_found = int(result.get("workouts_found", 0))
    processed = int(result.get("processed", 0))
    save_notification(
        user_id,
        title="✓ Hevy History Synced",
        message=f"Successfully synced and rebuilt {processed} of {workouts_found} workouts from Hevy.",
        type="sync",
        link="/history",
        db_path=DB_PATH,
    )
    return {"status": "ok", **result}


@app.get("/api/notifications")
def api_get_notifications(
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    unread_only: bool = Query(False),
):
    user_id = _check_api_auth(request)
    notifs = get_notifications(
        user_id, limit=limit, unread_only=unread_only, db_path=DB_PATH
    )
    unread_count = get_unread_notification_count(user_id, db_path=DB_PATH)
    return JSONResponse(
        jsonable_encoder(
            {
                "notifications": notifs,
                "unread_count": unread_count,
            }
        )
    )


@app.post("/api/notifications/{notification_id}/read")
def api_mark_notification_read(notification_id: int, request: Request):
    user_id = _check_api_auth(request)
    success = mark_notification_read(notification_id, user_id, db_path=DB_PATH)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    unread_count = get_unread_notification_count(user_id, db_path=DB_PATH)
    return {"status": "ok", "unread_count": unread_count}


@app.post("/api/notifications/read-all")
def api_mark_all_notifications_read(request: Request):
    user_id = _check_api_auth(request)
    updated = mark_all_notifications_read(user_id, db_path=DB_PATH)
    return {"status": "ok", "updated": updated, "unread_count": 0}


@app.delete("/api/notifications/{notification_id}")
def api_delete_notification(notification_id: int, request: Request):
    user_id = _check_api_auth(request)
    success = delete_notification(notification_id, user_id, db_path=DB_PATH)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    unread_count = get_unread_notification_count(user_id, db_path=DB_PATH)
    return {"status": "ok", "unread_count": unread_count}


@app.post("/api/notifications/clear")
@app.delete("/api/notifications")
def api_clear_notifications(request: Request):
    user_id = _check_api_auth(request)
    cleared = clear_all_notifications(user_id, db_path=DB_PATH)
    return {"status": "ok", "cleared": cleared, "unread_count": 0}


@app.post("/api/notifications/test-coach")
def api_test_coach_notification(request: Request):
    user_id = _check_api_auth(request)
    nid = save_notification(
        user_id,
        title="✨ Coach Status Update",
        message="Coach analysis active: Keep recovery high and log your sets accurately today!",
        type="coach",
        link="/dashboard",
        db_path=DB_PATH,
    )
    unread_count = get_unread_notification_count(user_id, db_path=DB_PATH)
    return {"status": "ok", "id": nid, "unread_count": unread_count}


@app.get("/api/chat-history")
def api_chat_history(request: Request):
    user_id = _check_api_auth(request)
    return JSONResponse(
        jsonable_encoder(
            {"messages": get_chat_messages(db_path=DB_PATH, user_id=user_id)}
        )
    )


@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
async def catch_all(full_path: str, request: Request):
    if full_path.startswith(
        ("api/", "static/", "login/google", "auth", "google-health/")
    ):
        raise HTTPException(status_code=404, detail="Not found")

    if full_path:
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)

    return FileResponse(FRONTEND_DIST / "index.html")
