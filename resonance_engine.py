"""Resonance/coherence calculations for the local prototype."""


class ResonanceEngine:
    def calculate_resonance(self, bsam):
        values = list(bsam.vector().values())
        mean = sum(values) / len(values)
        deviation = sum(abs(value - mean) for value in values) / len(values)
        bsam.resonance_score = max(0.0, 1.0 - deviation)
        return bsam
