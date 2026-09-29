"""Example entry point demonstrating resonance calculation."""

from activation_engine import ActivationEngine
from bsam_model import BSAMModel
from gemini_engine import GeminiEngine
from mindwave_symbolic import ACTIVATION_LAYERS
from resonance_engine import ResonanceEngine
from state_engine import StateEngine


def run():
    bsam = BSAMModel(B=1.0, S=1.0, A=1.0, M=1.2)
    bsam = ActivationEngine().apply_all(bsam, ACTIVATION_LAYERS)
    bsam = GeminiEngine().apply_gemini(bsam)
    bsam = ResonanceEngine().calculate_resonance(bsam)

    output = StateEngine().generate_output(
        bsam.vector(), resonance_score=bsam.resonance_score
    )
    print("Resonance Score:", bsam.resonance_score)
    print("System Output:", output)
    return output


if __name__ == "__main__":
    run()
