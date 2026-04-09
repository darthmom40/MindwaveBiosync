# --------------------------------------------
# MindWave BioSync™ – Main System Loop
# Activation → Gemini → State Engine
# --------------------------------------------

from bsam_model import BSAMModel
from state_engine import StateEngine
from activation_engine import ActivationEngine
from gemini_engine import GeminiEngine
from mindwave_symbolic import ACTIVATION_LAYERS  # your symbolic activation layers

def run():
    # -------------------------------
    # 1️⃣ Initialize Core Systems
    # -------------------------------
    bsam = BSAMModel(B=1.0, S=1.0, A=1.0, M=1.2)
    engine = StateEngine()
    activator = ActivationEngine()
    gemini = GeminiEngine()

    # -------------------------------
    # 2️⃣ Apply Activation Layers
    # -------------------------------
    bsam = activator.apply_all(bsam, ACTIVATION_LAYERS)
    print("After Activation Layers:", bsam.get_state())

    # -------------------------------
    # 3️⃣ Apply Gemini / Twin-State Logic
    # -------------------------------
    bsam = gemini.apply