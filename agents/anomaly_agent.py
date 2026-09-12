"""Public anomaly-agent service boundary."""
from src.anomaly_agent import check_thresholds, calculate_severity


class AnomalyAgent:
    def analyze(self, predictions):
        return check_thresholds(predictions)
