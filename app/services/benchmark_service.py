from typing import Dict, Any, List


class BenchmarkService:
    """Service interface for objective Classical vs QML utility benchmarking."""

    @staticmethod
    def compute_quantum_utility(
        classical_metrics: Dict[str, float],
        quantum_metrics: Dict[str, float],
        threshold: float = 0.02
    ) -> str:
        """Determines objective utility category without dogmatic bias."""
        c_bal_acc = classical_metrics.get("balanced_accuracy", 0.0)
        q_bal_acc = quantum_metrics.get("balanced_accuracy", 0.0)
        diff = q_bal_acc - c_bal_acc

        if diff > threshold:
            return "QML Superior"
        elif diff < -threshold:
            return "Classical ML Superior"
        else:
            return "Comparable Performance"


benchmark_service = BenchmarkService()
