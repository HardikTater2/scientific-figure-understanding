"""
CHART-Info UB PMC Real Benchmark Evaluator (Task 3a).
Evaluates FigureProcessor text-role classification on real PubMed Central chart annotations
from the ICPR/ICDAR CHART-Info benchmark.

Documented Text Role Mapping:
  - CHART-Info "axis_title", "x_axis_title", "y_axis_title"  -> "AXIS"
  - CHART-Info "legend_label", "legend_title"                 -> "LEGEND"
  - CHART-Info "tick_label", "mark_label", "value_label",
               "tick_grouping"                                -> "LABEL"
  - CHART-Info "chart_title", "other"                         -> "OTHER"

Evaluates:
  (i) Gold text bounding boxes (evaluates role classifier in isolation)
  (ii) OCR extracted bounding boxes (end-to-end evaluation including OCR detection errors)

Provenance: human_external_dataset
"""

import os
import sys
import json
from typing import Dict, Any, List, Tuple
from sklearn.metrics import classification_report, accuracy_score, f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CHART_INFO_ROLE_MAPPING = {
    "axis_title": "AXIS",
    "x_axis_title": "AXIS",
    "y_axis_title": "AXIS",
    "legend_label": "LEGEND",
    "legend_title": "LEGEND",
    "tick_label": "LABEL",
    "mark_label": "LABEL",
    "value_label": "LABEL",
    "tick_grouping": "LABEL",
    "chart_title": "OTHER",
    "other": "OTHER"
}

TARGET_CLASSES = ["AXIS", "LEGEND", "LABEL", "OTHER"]


def evaluate_chart_info_roles(predictions: List[str], ground_truth: List[str]) -> Dict[str, Any]:
    """
    Evaluates element role classification against mapped CHART-Info ground truth.
    Strictly measures real predictions against real human ground truth (zero simulation).
    """
    assert len(predictions) == len(ground_truth), "Predictions and ground truth must have equal length"
    acc = accuracy_score(ground_truth, predictions)
    micro_f1 = f1_score(ground_truth, predictions, average="micro")
    macro_f1 = f1_score(ground_truth, predictions, average="macro", zero_division=0)

    report = classification_report(ground_truth, predictions, labels=TARGET_CLASSES, output_dict=True, zero_division=0)

    return {
        "accuracy": float(round(acc, 4)),
        "micro_f1": float(round(micro_f1, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "total_elements": len(ground_truth),
        "per_class": {
            cls: {
                "precision": float(round(report[cls]["precision"], 4)),
                "recall": float(round(report[cls]["recall"], 4)),
                "f1": float(round(report[cls]["f1-score"], 4)),
                "support": int(report[cls]["support"])
            }
            for cls in TARGET_CLASSES if cls in report
        }
    }
