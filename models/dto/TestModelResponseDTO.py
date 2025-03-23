

class TestModelResponseDTO:
    def __init__(self, prediction: str,confidence: float, proability_breakdown: dict[str, float]):
        self.prediction = prediction
        self.confidence = confidence
        self.proability_breakdown = proability_breakdown