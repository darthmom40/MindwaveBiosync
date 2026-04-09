from bsam_model import BSAMModel
from state_engine import StateEngine
from activation_engine import ActivationEngine
from gemini_engine import GeminiEngine
from resonance_engine import ResonanceEngine
from mindwave_symbolic import ACTIVATION_LAYERS

def run():
    # 1️⃣ Initialize Systems
    bsam = BSAMModel(B=1.0, S=1.0, A=1.0, M=1.2)
    engine = StateEngine()
    activator = ActivationEngine()
    gemini = GeminiEngine()
    resonance = ResonanceEngine()

    # 2️⃣ Apply Activation Layers
    bsam = activator.apply_all(bsam, ACTIVATION_LAYERS)
    print("After Activation:", bsam.get_state())

    # 3️⃣ Apply Gemini
    bsam = gemini.apply_gemini(bsam)
    print("Gemini Vector:", bsam.gemini_vector)
    print("Gemini Coherence:", bsam.gemini_coherence)

    # 4️⃣ Calculate Resonance
    bsam = resonance.calculate_resonance(bsam)
    print("Resonance Score:", bsam.resonance_score)

    # 5️⃣ Generate Output
    vector = bsam.vector()
    output = engine.generate_output(vector, resonance_score=bsam.resonance_score)
    print("\n--- MindWave BioSync Output ---")
    print("BSAM State:", bsam.get_state())
    print("System Output:", output)

if __name__ == "__main__":
    run()