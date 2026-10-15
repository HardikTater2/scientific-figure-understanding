"""
Zero-Shot Vision-Language Model Baseline Pipeline (Rule 1 & 2).

Status: PENDING REAL LOCAL MODEL EXECUTION (Task 4: mlx-vlm).
Rule: Every model output must come from an actual model call.
Simulations have been removed.
"""

import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_FILE = os.path.join(ROOT, "results", "phase2", "zero_shot_vlm_baseline.json")


def evaluate_vlm():
    print("=" * 65)
    print("PHASE 2: ZERO-SHOT VLM GROUNDING BASELINE [STATUS: PENDING REAL MLX-VLM]")
    print("=" * 65)
    print("Simulated results removed under project honesty rules.")
    print("To run for real, mlx-vlm will be called in Task 4.")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump({
            "status": "NOT DONE",
            "provenance": "silver_heuristic",
            "note": "Zero-shot VLM simulation removed. Real local model call required."
        }, f, indent=2)


if __name__ == "__main__":
    evaluate_vlm()
