class ActivationEngine:
    """
    Converts symbolic activation layers into real BSAM state changes
    """

    def __init__(self):
        pass

    def apply_layer(self, bsam, layer):
        """
        Applies a single activation layer to BSAM model
        """

        if layer.name == "Heart Activation":
            # Emotional / relational boost
            bsam.update({"A": 0.2})

        elif layer.name == "Soul Activation":
            # Pattern + purpose alignment (mental clarity)
            bsam.update({"M": 0.15})

        elif layer.name == "Neural/Biofeedback":
            # Physical / brain regulation
            bsam.update({"B": 0.1})

        elif layer.name == "Cosmic/Energetic Alignment":
            # Sensory + timing alignment
            bsam.update({"S": 0.1})

    def apply_all(self, bsam, layers):
        """
        Apply multiple activation layers
        """
        for layer in layers:
            self.apply_layer(bsam, layer)

        # Normalize after all changes
        bsam.normalize()

        return bsam