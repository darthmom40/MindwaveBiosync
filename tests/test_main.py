import math

import pytest

from main import run, validate_state


def test_pipeline_returns_expected_structure():
    result = run(verbose=False)
    assert set(result["bsam_state"]) == {"B", "S", "A", "M"}
    assert set(result["gemini_vector"]) == {"B", "S", "A", "M"}
    assert result["dominant_dimension"] in {"B", "S", "A", "M"}
    assert 0.0 <= result["resonance_score"] <= 1.0
    assert 0.0 <= result["gemini_coherence"] <= 1.0
    assert all(math.isfinite(value) for value in result["bsam_state"].values())


def test_pipeline_accepts_custom_starting_state():
    result = run({"B": 0.2, "S": 0.4, "A": 0.6, "M": 0.8}, verbose=False)
    assert set(result["state_vector"]) == {"B", "S", "A", "M"}


@pytest.mark.parametrize(
    "state",
    [
        {"B": 1, "S": 1, "A": 1},
        {"B": 1, "S": 1, "A": 1, "M": 1, "X": 0},
        {"B": -1, "S": 1, "A": 1, "M": 1},
        {"B": float("nan"), "S": 1, "A": 1, "M": 1},
        {"B": float("inf"), "S": 1, "A": 1, "M": 1},
        {"B": "not-a-number", "S": 1, "A": 1, "M": 1},
    ],
)
def test_invalid_initial_state_is_rejected(state):
    with pytest.raises(ValueError):
        validate_state(state)


def test_validation_does_not_mutate_caller_state():
    state = {"B": 0.1, "S": 0.2, "A": 0.3, "M": 0.4}
    original = dict(state)
    validate_state(state)
    assert state == original
