from types import SimpleNamespace

from activation_engine import ActivationEngine


class FakeBSAM:
    def __init__(self):
        self.values = {"A": 0.0, "M": 0.0, "B": 0.0, "S": 0.0}
        self.normalized = False

    def update(self, changes):
        for key, value in changes.items():
            self.values[key] += value

    def normalize(self):
        self.normalized = True


def test_apply_all_activation_layers():
    bsam = FakeBSAM()
    layers = [
        SimpleNamespace(name="Heart Activation"),
        SimpleNamespace(name="Soul Activation"),
        SimpleNamespace(name="Neural/Biofeedback"),
        SimpleNamespace(name="Cosmic/Energetic Alignment"),
    ]

    result = ActivationEngine().apply_all(bsam, layers)

    assert result is bsam
    assert bsam.values == {"A": 0.2, "M": 0.15, "B": 0.1, "S": 0.1}
    assert bsam.normalized is True


def test_unknown_layer_is_ignored():
    bsam = FakeBSAM()

    ActivationEngine().apply_layer(
        bsam, SimpleNamespace(name="Unknown Layer")
    )

    assert bsam.values == {"A": 0.0, "M": 0.0, "B": 0.0, "S": 0.0}
