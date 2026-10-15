"""
SCQ Metric Human Validation Study (Rule 1 & 3).

Status: NOT DONE.
Reason: Single-author project with no in-house human ratings.
Per Non-Negotiable Rule 1:
"No simulated, random, estimated, or hardcoded results anywhere. If a step cannot
be done for real, write 'NOT DONE' in the results file and in your report."
"""

import os
import json

import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
from src.common_metadata import get_runtime_metadata

OUT_FILE = os.path.join(ROOT, "results", "phase5_scq_human_validation.json")


def run_human_validation():
    print("=" * 65)
    print("PHASE 5C: SCQ HUMAN VALIDATION [STATUS: NOT DONE]")
    print("=" * 65)
    print("No human ratings collected. Simulated ratings quarantined under honesty rules.")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump({
            "status": "NOT DONE",
            "provenance": "none",
            "note": "No human ratings collected. Quarantined under strict honesty rules.",
            **get_runtime_metadata("src/evaluation/validate_scq_human.py")
        }, f, indent=2)


if __name__ == "__main__":
    run_human_validation()
