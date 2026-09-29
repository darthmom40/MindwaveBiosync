"""Circadian-aware output layer for the prototype."""


class StateEngine:
    def generate_output(self, vector, resonance_score=0.0):
        dominant = max(vector, key=vector.get)
        return {
            "dominant_dimension": dominant,
            "state_vector": vector,
            "resonance_score": float(resonance_score),
        }
