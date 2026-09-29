"""Example entry point demonstrating the Gemini layer."""

from activation_engine import ActivationEngine
from bsam_model import BSAMModel
from gemini_engine import GeminiEngine
from mindwave_symbolic import ACTIVATION_LAYERS


def run():
    bsam = BSAMModel(B=1.0, S=1.0, A=1.0, M=1.2)
    bsam = ActivationEngine().apply_all(bsam, ACTIVATION_LAYERS)
    bsam = GeminiEngine().apply_gemini(bsam)

    print("After Activation:", bsam.get_state())
    print("Gemini Vector:", bsam.gemini_vector)
    print("Gemini Coherence:", bsam.gemini_coherence)
    return bsam


if __name__ == "__main__":
    run()
