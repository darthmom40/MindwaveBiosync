from bsam_model import BSAMModel
from gemini_engine import GeminiEngine
from mindwave_symbolic import ACTIVATION_LAYERS
from resonance_engine import ResonanceEngine
from state_engine import StateEngine


def test_core_pipeline_runs():
    bsam = BSAMModel(B=1.0, S=1.0, A=1.0, M=1.2)
    bsam.normalize()
    bsam = GeminiEngine().apply_gemini(bsam)
    bsam = ResonanceEngine().calculate_resonance(bsam)
    output = StateEngine().generate_output(bsam.vector(), bsam.resonance_score)

    assert set(bsam.vector()) == {"B", "S", "A", "M"}
    assert 0.0 <= bsam.resonance_score <= 1.0
    assert output["dominant_dimension"] in bsam.vector()


def test_activation_layer_catalog():
    assert [layer.name for layer in ACTIVATION_LAYERS] == [
        "Heart Activation",
        "Soul Activation",
        "Neural/Biofeedback",
        "Cosmic/Energetic Alignment",
    ]
