from typing import Dict, Any, List


class ExplainabilityService:
    """Service interface for feature importance, quantum sensitivity and decomposed confidence."""

    @staticmethod
    def calculate_confidence_decomposition(
        probability: float,
        data_quality: float = 1.0,
        ood_score: float = 0.0
    ) -> Dict[str, Any]:
        confidence = float(probability * data_quality * (1.0 - ood_score))
        return {
            "model_probability": probability,
            "data_quality_score": data_quality,
            "ood_status": "LOW" if ood_score < 0.2 else ("MODERATE" if ood_score < 0.6 else "HIGH"),
            "model_confidence_score": round(confidence, 4)
        }


explainability_service = ExplainabilityService()
