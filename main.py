"""CLI entry point for the experimental MindWave BioSync pipeline."""
from math import isfinite

from activation_engine import ActivationEngine
from bsam_model import BSAMModel
from gemini_engine import GeminiEngine
from mindwave_symbolic import ACTIVATION_LAYERS
from resonance_engine import ResonanceEngine
from state_engine import StateEngine

DEFAULT_STATE = {"B": 1.0, "S": 1.0, "A": 1.0, "M": 1.2}
STATE_KEYS = ("B", "S", "A", "M")


def validate_state(initial_state=None):
    """Validate starting dimensions and return a fresh dictionary."""
    state = dict(DEFAULT_STATE if initial_state is None else initial_state)
    if set(state) != set(STATE_KEYS):
        raise ValueError("initial_state must contain exactly B, S, A, and M")
    result = {}
    for key in STATE_KEYS:
        try:
            value = float(state[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a finite, non-negative number") from exc
        if not isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite, non-negative number")
        result[key] = value
    return result


def run(initial_state=None, *, verbose=True):
    """Run the local pipeline and return structured results."""
    bsam = BSAMModel(**validate_state(initial_state))
    bsam = ActivationEngine().apply_all(bsam, ACTIVATION_LAYERS)
    bsam = GeminiEngine().apply_gemini(bsam)
    bsam = ResonanceEngine().calculate_resonance(bsam)
    output = StateEngine().generate_output(
        bsam.vector(), resonance_score=bsam.resonance_score
    )
    output["bsam_state"] = bsam.get_state()
    output["gemini_vector"] = dict(bsam.gemini_vector)
    output["gemini_coherence"] = float(bsam.gemini_coherence)
    if verbose:
        print("BSAM State:", output["bsam_state"])
        print("Gemini Vector:", output["gemini_vector"])
        print("Gemini Coherence:", output["gemini_coherence"])
        print("Resonance Score:", output["resonance_score"])
        print("System Output:", output)
    return output


if __name__ == "__main__":
    run()
