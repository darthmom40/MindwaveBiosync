"""Core BSAM (Bio-Signal Adaptive Matrix) state model."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BSAMModel:
    """Stores normalized biological, sensory, affective, and mental state."""

    B: float = 1.0
    S: float = 1.0
    A: float = 1.0
    M: float = 1.0
    extras: Dict[str, float] = field(default_factory=dict)

    def update(self, changes: Dict[str, float]) -> None:
        for key, value in changes.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown BSAM dimension: {key}")
            setattr(self, key, float(getattr(self, key)) + float(value))

    def normalize(self) -> "BSAMModel":
        values = {"B": self.B, "S": self.S, "A": self.A, "M": self.M}
        maximum = max(values.values(), default=1.0)
        if maximum > 1.0:
            for key in values:
                setattr(self, key, getattr(self, key) / maximum)
        return self

    def vector(self) -> Dict[str, float]:
        return {"B": self.B, "S": self.S, "A": self.A, "M": self.M}

    def get_state(self) -> Dict[str, float]:
        return self.vector()
