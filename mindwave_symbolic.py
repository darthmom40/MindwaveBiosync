"""Symbolic activation-layer definitions used by MindWave BioSync."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ActivationLayer:
    name: str
    description: str = ""


ACTIVATION_LAYERS = [
    ActivationLayer("Heart Activation", "Emotional and relational state."),
    ActivationLayer("Soul Activation", "Purpose and reflective pattern alignment."),
    ActivationLayer("Neural/Biofeedback", "Biological and regulation state."),
    ActivationLayer("Cosmic/Energetic Alignment", "Sensory and timing alignment."),
]
