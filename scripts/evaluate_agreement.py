"""
Inter-Annotator Agreement Evaluation Module (Phase 1, Step 4).

Status: NOT DONE.
Reason: Single-author project. Dual human annotation was not performed.
Per Non-Negotiable Rule 1:
"No simulated, random, estimated, or hardcoded results anywhere. If a step cannot
be done for real, write 'NOT DONE' in the results file and in your report."
"""

import os
import sys
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from src.common_metadata import get_runtime_metadata

AGREEMENT_OUT = os.path.join(ROOT, "results", "phase1", "inter_annotator_agreement.json")


def evaluate_agreement():
    print("\n" + "=" * 60)
    print("INTER-ANNOTATOR AGREEMENT REPORT [STATUS: NOT DONE]")
    print("=" * 60)
    print("No dual human annotation was performed. Quarantined under strict honesty rules.")
    print("=" * 60)

    os.makedirs(os.path.dirname(AGREEMENT_OUT), exist_ok=True)
    out = {
        "status": "NOT DONE",
        "provenance": "none",
        "note": "No dual human annotation was performed. Quarantined under strict honesty rules.",
        **get_runtime_metadata("scripts/evaluate_agreement.py")
    }
    with open(AGREEMENT_OUT, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved agreement report to: {AGREEMENT_OUT}")
    return out


if __name__ == "__main__":
    evaluate_agreement()
