import unittest

from agents.ai_harness import AIHarness


class FakeAnomalyAgent:
    def __init__(self, status):
        self.status = status

    def analyze(self, predictions):
        return dict(self.status)


class FakeRootCauseAgent:
    def __init__(self):
        self.called = False
        self.use_ollama = None

    def analyze(self, anomaly, use_ollama=True):
        self.called = True
        self.use_ollama = use_ollama
        return {
            "status": "ANOMALY",
            "root_causes": [],
            "recommended_actions": [],
            "qwen_reasoning": "RCA explanation",
        }


class AIHarnessTests(unittest.TestCase):
    def test_normal_run_skips_root_cause_agent(self):
        root = FakeRootCauseAgent()
        harness = AIHarness(
            anomaly_agent=FakeAnomalyAgent({
                "status": "NORMAL",
                "severity": "LOW",
                "anomalies": [],
            }),
            root_cause_agent=root,
            explainer=lambda status, context: "Network healthy",
        )

        result = harness.run([], "24 hours")

        self.assertEqual(result["status"]["status"], "NORMAL")
        self.assertIsNone(result["root_cause_analysis"])
        self.assertFalse(root.called)
        self.assertEqual(result["agent_text"], "Network healthy")
        self.assertTrue(result["run_id"])
        self.assertEqual(result["harness_trace"][0]["status"], "COMPLETED")
        self.assertEqual(result["harness_trace"][1]["status"], "SKIPPED")

    def test_anomaly_run_calls_root_cause_agent(self):
        root = FakeRootCauseAgent()
        harness = AIHarness(
            anomaly_agent=FakeAnomalyAgent({
                "status": "ANOMALY",
                "severity": "HIGH",
                "anomalies": [{"metric": "latency_ms", "value": 90}],
                "maximum_latency": 90,
                "minimum_throughput": 100,
                "maximum_jitter": 20,
            }),
            root_cause_agent=root,
        )

        result = harness.run([], "2 hours", use_ollama=True)

        self.assertTrue(root.called)
        self.assertTrue(root.use_ollama)
        self.assertEqual(result["root_cause_analysis"]["status"], "ANOMALY")
        self.assertEqual(result["agent_text"], "RCA explanation")
        self.assertEqual(result["harness_trace"][1]["stage"], "root_cause_analysis")

    def test_anomaly_run_can_disable_ollama(self):
        root = FakeRootCauseAgent()
        harness = AIHarness(
            anomaly_agent=FakeAnomalyAgent({
                "status": "ANOMALY",
                "severity": "HIGH",
                "anomalies": [{"metric": "latency_ms", "value": 90}],
            }),
            root_cause_agent=root,
        )

        result = harness.run([], "2 hours", use_ollama=False)

        self.assertTrue(root.called)
        self.assertFalse(root.use_ollama)
        self.assertEqual(result["harness_trace"][2]["status"], "SKIPPED")


if __name__ == "__main__":
    unittest.main()
