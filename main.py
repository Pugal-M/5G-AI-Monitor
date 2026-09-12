"""Convenient project entry point: python main.py --scenario anomaly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from main import run_pipeline

if __name__ == "__main__":
    import argparse
    from agents.root_cause_agent import format_rca_report

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("normal", "anomaly"), default="normal")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(args.scenario, args.generate)
    print(result["status"])
    if result["root_cause_analysis"]:
        print(format_rca_report(result["root_cause_analysis"]))
