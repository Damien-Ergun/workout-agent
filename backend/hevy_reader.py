"""Read-first Hevy integration: understand what the user actually trains.

Instead of pushing a static programme TO Hevy (the old ``hevy_sync.py`` flow),
this module pulls the user's routines, recent workouts, and exercise templates
FROM Hevy, then distills them into a structured ``HevyTrainingData`` object
that the rest of the system can reason about.

The data is fetched per-user using their stored API key (from Sprint 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime, timezone

from hevy_client import (
    get_exercise_templates,
    get_recent_workouts,
    get_routine_folders,
    get_routines,
    get_user_info,
    get_workout_count,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ExerciseTemplate:
    """Metadata for a single exercise from the Hevy template library."""

    id: str
    title: str
    exercise_type: str  # "weight_reps", "bodyweight_reps", "duration", etc.
    primary_muscle_group: str
    secondary_muscle_groups: list[str] = field(default_factory=list)
    equipment: str | None = None
    is_custom: bool = False


@dataclass
class RoutineExercise:
    """A single exercise within a routine template."""

    template_id: str
    title: str  # Resolved from the template
    sets: int
    rest_seconds: int
    notes: str | None = None
    superset_id: int | None = None
    # Planned targets (from the routine, not logged values)
    target_weight_kg: float | None = None
    target_reps: int | None = None
    target_rep_range: tuple[int, int] | None = None
    # Exact provider set payloads are retained for source preview and hashing.
    set_targets: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Routine:
    """A routine template as it exists in Hevy."""

    id: str
    title: str
    folder_id: int | None = None
    exercises: list[RoutineExercise] = field(default_factory=list)
    updated_at: str | None = None


@dataclass
class WorkoutExercise:
    """A single exercise from a completed workout."""

    template_id: str
    title: str
    sets: list[dict[str, Any]] = field(default_factory=list)
    # Derived top set
    top_weight_kg: float | None = None
    top_reps: int | None = None
    total_sets: int = 0


@dataclass
class CompletedWorkout:
    """A completed (logged) workout from Hevy."""

    id: str
    title: str
    start_time: str | None = None
    end_time: str | None = None
    duration_seconds: int | None = None
    exercises: list[WorkoutExercise] = field(default_factory=list)
    total_volume_kg: float = 0.0


@dataclass
class HevyTrainingData:
    """Everything we know about a user's training from their Hevy account.

    This is the read-first data structure that replaces the static ``program.py``
    for determining what a user trains, how often, and at what intensity.
    """

    # Account info
    username: str | None = None
    workout_count: int | None = None

    # Exercise library (template_id -> metadata)
    exercise_templates: dict[str, ExerciseTemplate] = field(default_factory=dict)

    # Routine templates (what the user plans to do)
    routines: list[Routine] = field(default_factory=list)

    # Recently completed workouts (what they actually did)
    recent_workouts: list[CompletedWorkout] = field(default_factory=list)

    # Routine folders
    folders: dict[int, str] = field(default_factory=dict)  # folder_id -> title

    def routine_titles(self) -> list[str]:
        """Return all routine titles (the user's training days)."""
        return [r.title for r in self.routines]

    def exercises_for_routine(self, title: str) -> list[RoutineExercise]:
        """Return exercises for a named routine."""
        for r in self.routines:
            if r.title == title:
                return r.exercises
        return []

    def muscle_group_for(self, template_id: str) -> str:
        """Return the primary muscle group for an exercise template."""
        tmpl = self.exercise_templates.get(template_id)
        return tmpl.primary_muscle_group if tmpl else "other"

    def exercise_name(self, template_id: str) -> str:
        """Human-readable name for a template ID."""
        tmpl = self.exercise_templates.get(template_id)
        return tmpl.title if tmpl else template_id

    def latest_workout(self) -> CompletedWorkout | None:
        """Return the most recently completed workout."""
        return self.recent_workouts[0] if self.recent_workouts else None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_exercise_template(raw: dict[str, Any]) -> ExerciseTemplate:
    return ExerciseTemplate(
        id=str(raw.get("id", "")),
        title=raw.get("title", "Unknown"),
        exercise_type=raw.get("type", "weight_reps"),
        primary_muscle_group=raw.get("primary_muscle_group", "other"),
        secondary_muscle_groups=raw.get("secondary_muscle_groups") or [],
        equipment=raw.get("equipment"),
        is_custom=raw.get("is_custom", False),
    )


def _parse_routine(
    raw: dict[str, Any],
    templates: dict[str, ExerciseTemplate],
) -> Routine:
    exercises: list[RoutineExercise] = []
    for ex_raw in raw.get("exercises", []):
        tid = str(ex_raw.get("exercise_template_id", ""))
        tmpl = templates.get(tid)
        title = tmpl.title if tmpl else tid

        sets_raw = ex_raw.get("sets", [])
        num_sets = len(sets_raw)

        # Extract planned targets from the first set (routines usually
        # have uniform targets across sets).
        target_weight = None
        target_reps = None
        target_range = None
        if sets_raw:
            first_set = sets_raw[0] if isinstance(sets_raw[0], dict) else {}
            target_weight = first_set.get("weight_kg")
            target_reps = first_set.get("reps")
            rep_range = first_set.get("rep_range")
            if isinstance(rep_range, dict):
                target_range = (
                    rep_range.get("start", 0),
                    rep_range.get("end", 0),
                )

        exercises.append(
            RoutineExercise(
                template_id=tid,
                title=title,
                sets=num_sets,
                rest_seconds=ex_raw.get("rest_seconds", 90),
                notes=ex_raw.get("notes"),
                superset_id=ex_raw.get("superset_id"),
                target_weight_kg=target_weight,
                target_reps=target_reps,
                target_rep_range=target_range,
                set_targets=[
                    dict(item) if isinstance(item, dict) else {"value": item}
                    for item in sets_raw
                ],
            ),
        )

    return Routine(
        id=str(raw.get("id", "")),
        title=raw.get("title", "Untitled"),
        folder_id=raw.get("folder_id"),
        exercises=exercises,
        updated_at=raw.get("updated_at"),
    )


def _top_set_from_sets(
    sets_raw: list[dict[str, Any]],
) -> tuple[float | None, int | None]:
    """Return (top_weight_kg, top_reps) from a list of set dicts."""
    best_weight: float | None = None
    best_reps: int | None = None
    best_key = (-1.0, -1)

    for s in sets_raw:
        if not isinstance(s, dict):
            continue
        w = s.get("weight_kg")
        r = s.get("reps")
        key = (float(w) if w is not None else -1.0, int(r) if r is not None else -1)
        if key > best_key:
            best_key = key
            best_weight = float(w) if w is not None else None
            best_reps = int(r) if r is not None else None

    return best_weight, best_reps

def _parse_hevy_datetime(value: Any) -> datetime | None:
    """Parse Hevy timestamps from ISO strings or Unix seconds/milliseconds."""
    if value is None:
        return None

    try:
        if isinstance(value, (int, float)):
            timestamp = float(value)

            # Hevy's web API may return JavaScript timestamps in milliseconds.
            if abs(timestamp) > 10_000_000_000:
                timestamp /= 1000.0

            return datetime.fromtimestamp(timestamp, tz=timezone.utc)

        text = str(value).strip()
        if not text:
            return None

        # Also tolerate numeric timestamps encoded as strings.
        try:
            timestamp = float(text)
        except ValueError:
            timestamp = None

        if timestamp is not None:
            if abs(timestamp) > 10_000_000_000:
                timestamp /= 1000.0
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)

        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    except (ValueError, TypeError, OverflowError, OSError):
        return None
    
    
def _parse_workout(
    raw: dict[str, Any],
    templates: dict[str, ExerciseTemplate],
) -> CompletedWorkout:
    exercises: list[WorkoutExercise] = []
    total_volume = 0.0

    for ex_raw in raw.get("exercises", []):
        tid = str(ex_raw.get("exercise_template_id", ""))
        tmpl = templates.get(tid)
        title = ex_raw.get("title") or (tmpl.title if tmpl else tid)

        sets_raw = ex_raw.get("sets", [])
        top_w, top_r = _top_set_from_sets(sets_raw)

        # Calculate volume from all sets.
        for s in sets_raw:
            if isinstance(s, dict):
                w = s.get("weight_kg") or 0
                r = s.get("reps") or 0
                total_volume += w * r

        exercises.append(
            WorkoutExercise(
                template_id=tid,
                title=title,
                sets=sets_raw,
                top_weight_kg=top_w,
                top_reps=top_r,
                total_sets=len(sets_raw),
            ),
        )

    # Duration and timestamp normalization.
    raw_start = raw.get("start_time")
    raw_end = raw.get("end_time")

    start_dt = _parse_hevy_datetime(raw_start)
    end_dt = _parse_hevy_datetime(raw_end)

    start = (
        start_dt.isoformat().replace("+00:00", "Z")
        if start_dt is not None
        else None
    )
    end = (
        end_dt.isoformat().replace("+00:00", "Z")
        if end_dt is not None
        else None
    )

    duration = None
    if start_dt is not None and end_dt is not None:
        duration = max(0, int((end_dt - start_dt).total_seconds()))

    return CompletedWorkout(
        id=str(raw.get("id", "")),
        title=raw.get("title", "Workout"),
        start_time=start,
        end_time=end,
        duration_seconds=duration,
        exercises=exercises,
        total_volume_kg=round(total_volume, 1),
    )


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------


def fetch_user_training(api_key: str, *, workout_limit: int = 15) -> HevyTrainingData:
    """Pull everything we need from Hevy to understand what this user trains.

    This is the single entry point for the read-first flow. It fetches:
    1. Exercise templates (the exercise library with muscle groups)
    2. Routine templates (what the user plans to do)
    3. Recent completed workouts (what they actually did)
    4. Routine folders (organisational structure)
    5. Account info (username, workout count)

    All calls are best-effort: a failure in one does not block the others.
    """
    data = HevyTrainingData()

    # 1. Exercise templates (needed to resolve names in routines/workouts).
    raw_templates = get_exercise_templates(api_key)
    if raw_templates:
        for raw in raw_templates:
            tmpl = _parse_exercise_template(raw)
            data.exercise_templates[tmpl.id] = tmpl
        logger.info(
            "Fetched %d exercise templates from Hevy.", len(data.exercise_templates)
        )
    else:
        logger.warning("Could not fetch exercise templates from Hevy.")

    # 2. Routines.
    raw_routines = get_routines(api_key)
    if raw_routines:
        data.routines = [
            _parse_routine(r, data.exercise_templates) for r in raw_routines
        ]
        logger.info("Fetched %d routines from Hevy.", len(data.routines))

    # 3. Recent workouts.
    raw_workouts = get_recent_workouts(api_key, limit=workout_limit)
    if raw_workouts:
        data.recent_workouts = [
            _parse_workout(w, data.exercise_templates) for w in raw_workouts
        ]
        logger.info("Fetched %d recent workouts from Hevy.", len(data.recent_workouts))

    # 4. Folders.
    raw_folders = get_routine_folders(api_key)
    if raw_folders:
        data.folders = {
            int(f["id"]): f.get("title", "") for f in raw_folders if "id" in f
        }

    # 5. Account info.
    user_info = get_user_info(api_key)
    if user_info:
        data.username = user_info.get("username")
    count = get_workout_count(api_key)
    if count is not None:
        data.workout_count = count

    return data


def fetch_private_user_training(
    client: Any,
    *,
    workout_limit: int = 15,
) -> HevyTrainingData:
    """Normalize data fetched through the unsupported Hevy web protocol.

    The private API does not expose the documented exercise-template catalogue,
    so template names are derived from synchronized routine/workout exercises.
    """
    data = HevyTrainingData()
    raw_routines = client.routines()
    raw_workouts = client.workouts()

    for source in (*raw_routines, *raw_workouts):
        for raw_exercise in source.get("exercises", []):
            if not isinstance(raw_exercise, dict):
                continue
            template_id = str(raw_exercise.get("exercise_template_id", ""))
            if not template_id or template_id in data.exercise_templates:
                continue
            data.exercise_templates[template_id] = ExerciseTemplate(
                id=template_id,
                title=str(raw_exercise.get("title") or template_id),
                exercise_type=str(raw_exercise.get("type") or "weight_reps"),
                primary_muscle_group=str(
                    raw_exercise.get("primary_muscle_group") or "other"
                ),
                secondary_muscle_groups=list(
                    raw_exercise.get("secondary_muscle_groups") or []
                ),
                equipment=raw_exercise.get("equipment"),
                is_custom=bool(raw_exercise.get("is_custom", False)),
            )

    data.routines = [
        _parse_routine(routine, data.exercise_templates) for routine in raw_routines
    ]
    data.recent_workouts = [
        _parse_workout(workout, data.exercise_templates)
        for workout in raw_workouts[:workout_limit]
    ]
    account = client.account()
    data.username = account.get("username")
    data.workout_count = client.workout_count()
    return data
