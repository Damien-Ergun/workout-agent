"""Tests for hevy_reader — the read-first Hevy integration module."""

from __future__ import annotations

from unittest.mock import patch

import hevy_reader



# ---------------------------------------------------------------------------
# Parse helper tests (pure functions, no mocking needed)
# ---------------------------------------------------------------------------




def test_parse_hevy_datetime_accepts_iso_string():
    dt = hevy_reader._parse_hevy_datetime("2026-08-24T12:30:00Z")

    assert dt is not None
    assert dt.isoformat() == "2026-08-24T12:30:00+00:00"


def test_parse_hevy_datetime_accepts_unix_seconds():
    dt = hevy_reader._parse_hevy_datetime(1756038600)

    assert dt is not None
    assert dt.tzinfo is not None


def test_parse_hevy_datetime_accepts_unix_milliseconds():
    dt = hevy_reader._parse_hevy_datetime(1756038600000)

    assert dt is not None
    assert dt.tzinfo is not None


def test_parse_hevy_datetime_rejects_invalid_value():
    assert hevy_reader._parse_hevy_datetime("not-a-date") is None
    
    
def test_parse_exercise_template_minimal() -> None:
    raw: dict[str, object] = {
        "id": "001",
        "title": "Bench Press",
        "type": "weight_reps",
        "primary_muscle_group": "chest",
        "secondary_muscle_groups": ["shoulders", "triceps"],
    }
    tmpl = hevy_reader._parse_exercise_template(raw)
    assert tmpl.id == "001"
    assert tmpl.title == "Bench Press"
    assert tmpl.exercise_type == "weight_reps"
    assert tmpl.primary_muscle_group == "chest"
    assert tmpl.secondary_muscle_groups == ["shoulders", "triceps"]
    assert tmpl.equipment is None
    assert tmpl.is_custom is False


def test_parse_exercise_template_custom_and_equipment() -> None:
    raw: dict[str, object] = {
        "id": "CUST01",
        "title": "My Curl",
        "type": "bodyweight_reps",
        "primary_muscle_group": "biceps",
        "is_custom": True,
        "equipment": "dumbbell",
    }
    tmpl = hevy_reader._parse_exercise_template(raw)
    assert tmpl.id == "CUST01"
    assert tmpl.equipment == "dumbbell"
    assert tmpl.is_custom is True
    assert tmpl.secondary_muscle_groups == []


def test_parse_exercise_template_defaults() -> None:
    raw: dict[str, object] = {}
    tmpl = hevy_reader._parse_exercise_template(raw)
    assert tmpl.id == ""
    assert tmpl.title == "Unknown"
    assert tmpl.exercise_type == "weight_reps"
    assert tmpl.primary_muscle_group == "other"


def test_top_set_from_sets_empty() -> None:
    w, r = hevy_reader._top_set_from_sets([])
    assert w is None
    assert r is None


def test_top_set_from_sets_skips_non_dict() -> None:
    w, r = hevy_reader._top_set_from_sets([{"weight_kg": 50, "reps": 8}, "junk"])  # type: ignore[list-item]
    assert w == 50.0
    assert r == 8


def test_top_set_from_sets_picks_highest_weight() -> None:
    sets: list[dict[str, object]] = [
        {"weight_kg": 60, "reps": 12},
        {"weight_kg": 80, "reps": 6},
        {"weight_kg": 70, "reps": 10},
    ]
    w, r = hevy_reader._top_set_from_sets(sets)
    assert w == 80.0
    assert r == 6


def test_top_set_from_sets_none_values_handled() -> None:
    sets: list[dict[str, object]] = [
        {"weight_kg": None, "reps": None},
        {"weight_kg": 40, "reps": 10},
    ]
    w, r = hevy_reader._top_set_from_sets(sets)
    assert w == 40.0
    assert r == 10


def test_parse_routine_with_targets() -> None:
    templates: dict[str, hevy_reader.ExerciseTemplate] = {}
    raw: dict[str, object] = {
        "id": "R1",
        "title": "Push Day",
        "folder_id": 2,
        "updated_at": "2025-01-01T00:00:00Z",
        "exercises": [
            {
                "exercise_template_id": "001",
                "rest_seconds": 120,
                "notes": "Go heavy",
                "superset_id": 1,
                "sets": [
                    {"weight_kg": 80, "reps": 10, "rep_range": {"start": 8, "end": 12}},
                ],
            },
        ],
    }
    routine = hevy_reader._parse_routine(raw, templates)
    assert routine.id == "R1"
    assert routine.title == "Push Day"
    assert routine.folder_id == 2
    assert routine.updated_at == "2025-01-01T00:00:00Z"
    assert len(routine.exercises) == 1
    ex = routine.exercises[0]
    assert ex.template_id == "001"
    assert ex.title == "001"  # not resolved since template not in dict
    assert ex.sets == 1
    assert ex.rest_seconds == 120
    assert ex.notes == "Go heavy"
    assert ex.superset_id == 1
    assert ex.target_weight_kg == 80
    assert ex.target_reps == 10
    assert ex.target_rep_range == (8, 12)


def test_parse_routine_resolves_template_title() -> None:
    templates = {
        "001": hevy_reader.ExerciseTemplate(
            id="001",
            title="Bench Press",
            exercise_type="weight_reps",
            primary_muscle_group="chest",
        ),
    }
    raw: dict[str, object] = {
        "id": "R2",
        "title": "Push Day",
        "exercises": [
            {"exercise_template_id": "001", "rest_seconds": 90, "sets": []},
        ],
    }
    routine = hevy_reader._parse_routine(raw, templates)
    assert routine.exercises[0].title == "Bench Press"
    assert routine.exercises[0].sets == 0


def test_parse_workout_single_exercise() -> None:
    raw: dict[str, object] = {
        "id": "W1",
        "title": "Morning session",
        "start_time": "2025-06-01T08:00:00Z",
        "end_time": "2025-06-01T09:00:00Z",
        "exercises": [
            {
                "exercise_template_id": "001",
                "title": "Bench Press",
                "sets": [
                    {"weight_kg": 80, "reps": 10},
                    {"weight_kg": 85, "reps": 8},
                ],
            },
        ],
    }
    tmpl = hevy_reader.ExerciseTemplate(
        id="001",
        title="Bench Press",
        exercise_type="weight_reps",
        primary_muscle_group="chest",
    )
    templates = {"001": tmpl}
    workout = hevy_reader._parse_workout(raw, templates)
    assert workout.id == "W1"
    assert workout.title == "Morning session"
    assert workout.start_time == "2025-06-01T08:00:00Z"
    assert workout.duration_seconds == 3600
    assert workout.total_volume_kg == 80 * 10 + 85 * 8
    assert len(workout.exercises) == 1
    ex = workout.exercises[0]
    assert ex.template_id == "001"
    assert ex.title == "Bench Press"
    assert ex.top_weight_kg == 85.0
    assert ex.top_reps == 8
    assert ex.total_sets == 2


def test_parse_workout_no_duration_when_missing_times() -> None:
    raw: dict[str, object] = {
        "id": "W2",
        "title": "Quickie",
        "exercises": [],
    }
    workout = hevy_reader._parse_workout(raw, {})
    assert workout.duration_seconds is None


def test_parse_workout_volume_handles_missing_weight_reps() -> None:
    raw: dict[str, object] = {
        "id": "W3",
        "title": "Bodyweight",
        "exercises": [
            {
                "exercise_template_id": "BW001",
                "sets": [{"reps": 20}, {"weight_kg": None, "reps": None}],
            },
        ],
    }
    workout = hevy_reader._parse_workout(raw, {})
    assert workout.total_volume_kg == 0.0


# ---------------------------------------------------------------------------
# HevyTrainingData methods (pure)
# ---------------------------------------------------------------------------


def make_training_data() -> hevy_reader.HevyTrainingData:
    data = hevy_reader.HevyTrainingData()
    data.exercise_templates = {
        "001": hevy_reader.ExerciseTemplate(
            id="001",
            title="Bench Press",
            exercise_type="weight_reps",
            primary_muscle_group="chest",
        ),
        "002": hevy_reader.ExerciseTemplate(
            id="002",
            title="Squat",
            exercise_type="weight_reps",
            primary_muscle_group="quads",
        ),
    }
    data.routines = [
        hevy_reader.Routine(
            id="R1",
            title="Push Day",
            exercises=[
                hevy_reader.RoutineExercise(
                    template_id="001",
                    title="Bench Press",
                    sets=3,
                    rest_seconds=90,
                ),
            ],
        ),
        hevy_reader.Routine(id="R2", title="Leg Day", exercises=[]),
    ]
    data.recent_workouts = [
        hevy_reader.CompletedWorkout(id="W1", title="Push workout"),
        hevy_reader.CompletedWorkout(id="W2", title="Leg workout"),
    ]
    return data


def test_routine_titles() -> None:
    data = make_training_data()
    assert data.routine_titles() == ["Push Day", "Leg Day"]


def test_exercises_for_routine() -> None:
    data = make_training_data()
    assert len(data.exercises_for_routine("Push Day")) == 1
    assert data.exercises_for_routine("Push Day")[0].template_id == "001"
    assert data.exercises_for_routine("Leg Day") == []
    assert data.exercises_for_routine("Nonexistent") == []


def test_muscle_group_for() -> None:
    data = make_training_data()
    assert data.muscle_group_for("001") == "chest"
    assert data.muscle_group_for("002") == "quads"
    assert data.muscle_group_for("unknown") == "other"


def test_exercise_name() -> None:
    data = make_training_data()
    assert data.exercise_name("001") == "Bench Press"
    assert data.exercise_name("unknown") == "unknown"


def test_latest_workout() -> None:
    data = make_training_data()
    assert data.latest_workout() is not None
    assert data.latest_workout().id == "W1"  # type: ignore[union-attr]

    empty = hevy_reader.HevyTrainingData()
    assert empty.latest_workout() is None


# ---------------------------------------------------------------------------
# fetch_user_training (orchestration — all network calls mocked)
# ---------------------------------------------------------------------------


def _fake_templates(api_key: str) -> list[dict[str, object]]:
    return [
        {
            "id": "001",
            "title": "Bench Press",
            "type": "weight_reps",
            "primary_muscle_group": "chest",
        },
    ]


def _fake_routines(api_key: str) -> list[dict[str, object]]:
    return [
        {"id": "R1", "title": "Push Day", "exercises": []},
    ]


def _fake_recent_workouts(api_key: str, limit: int = 10) -> list[dict[str, object]]:
    return [
        {
            "id": "W1",
            "title": "Push session",
            "start_time": "2025-06-01T08:00:00Z",
            "end_time": "2025-06-01T09:00:00Z",
            "exercises": [],
        },
    ]


def _fake_folders(api_key: str) -> list[dict[str, object]]:
    return [{"id": 1, "title": "My Folder"}]


def _fake_user_info(api_key: str) -> dict[str, object]:
    return {"username": "testuser"}


def test_fetch_user_training_success() -> None:
    with (
        patch("hevy_reader.get_exercise_templates", _fake_templates),
        patch("hevy_reader.get_routines", _fake_routines),
        patch("hevy_reader.get_recent_workouts", _fake_recent_workouts),
        patch("hevy_reader.get_routine_folders", _fake_folders),
        patch("hevy_reader.get_user_info", _fake_user_info),
        patch("hevy_reader.get_workout_count", lambda api_key: 42),
    ):
        data = hevy_reader.fetch_user_training("fake-key")
        assert data.username == "testuser"
        assert data.workout_count == 42
        assert len(data.exercise_templates) == 1
        assert len(data.routines) == 1
        assert len(data.recent_workouts) == 1
        assert data.folders == {1: "My Folder"}


def test_fetch_user_training_graceful_degradation() -> None:
    """None of the Hevy endpoints respond, but we still get an empty object."""

    def _none(*args: object, **kwargs: object) -> None:
        return None

    with (
        patch("hevy_reader.get_exercise_templates", _none),
        patch("hevy_reader.get_routines", _none),
        patch("hevy_reader.get_recent_workouts", _none),
        patch("hevy_reader.get_routine_folders", _none),
        patch("hevy_reader.get_user_info", _none),
        patch("hevy_reader.get_workout_count", _none),
    ):
        data = hevy_reader.fetch_user_training("fake-key")
        assert data.username is None
        assert data.workout_count is None
        assert data.exercise_templates == {}
        assert data.routines == []
        assert data.recent_workouts == []
        assert data.folders == {}


def test_fetch_user_training_partial_failure() -> None:
    """Some endpoints work, some don't — data should be partial."""

    def _none(*args: object, **kwargs: object) -> None:
        return None

    with (
        patch("hevy_reader.get_exercise_templates", _fake_templates),
        patch("hevy_reader.get_routines", _none),
        patch("hevy_reader.get_recent_workouts", _fake_recent_workouts),
        patch("hevy_reader.get_routine_folders", _none),
        patch("hevy_reader.get_user_info", _none),
        patch("hevy_reader.get_workout_count", lambda api_key: 5),
    ):
        data = hevy_reader.fetch_user_training("fake-key")
        assert data.workout_count == 5
        assert len(data.exercise_templates) == 1
        assert data.routines == []  # routines failed
        assert len(data.recent_workouts) == 1
        assert data.folders == {}
        assert data.username is None
