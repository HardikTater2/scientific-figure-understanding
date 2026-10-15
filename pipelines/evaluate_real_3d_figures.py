"""
Real-Figure 3D Detector Evaluation Pipeline (Quarantined under Rule 1).

Status: NOT DONE.
Reason: Single-author project. Verified published 3D figure images with source URLs
have not been curated in this environment. As per Non-Negotiable Rule 1:
"No simulated, random, estimated, or hardcoded results anywhere. If a step cannot
be done for real, write 'NOT DONE' in the results file and in your report."
"""

import os
import json

import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from src.common_metadata import get_runtime_metadata

OUT_FILE = os.path.join(ROOT, "results", "phase4_real_3d_evaluation.json")


def main():
    print("=" * 65)
    print("PHASE 4: REAL 3D DETECTOR EVALUATION [STATUS: NOT DONE]")
    print("=" * 65)
    print("Reason: 100 published 3D figures with verified source URLs are not curated.")
    print("Per Non-Negotiable Rule 1, simulated results have been removed.")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump({
            "status": "NOT DONE",
            "provenance": "none",
            "note": "Figures were not downloaded published figures with source URLs. Quarantined.",
            **get_runtime_metadata("pipelines/evaluate_real_3d_figures.py")
        }, f, indent=2)


if __name__ == "__main__":
    main()
