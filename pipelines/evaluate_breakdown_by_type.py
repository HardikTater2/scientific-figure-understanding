"""
Fine-Grained Breakdown by Figure and Element Type (Phase 6).
Computes grounding performance broken down by:
  1. Element Type: LEGEND, AXIS, LABEL
  2. Figure Type: Line Plot, Bar Chart, Scatter Plot, Heatmap, Diagram

Flags any subcategory with N < 20 figures, computes 95% bootstrap confidence intervals,
and saves results to results/phase6_finegrained_breakdown.json.
"""

import os
import sys
import json
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.evaluation.bootstrap import paired_bootstrap
from src.grounding.grounding_features import pair_features, build_grounding_graph, TEXT_ELEMENT_TYPES
from sklearn.linear_model import LogisticRegression
from src.common_metadata import get_runtime_metadata

BENCHMARK_PATH = os.path.join(ROOT, "data", "scicap_subset", "silver_grounding_v0.json")
OUT_FILE = os.path.join(ROOT, "results", "appendix_circular", "phase6_finegrained_breakdown.json")


def infer_figure_type(fig: Dict[str, Any]) -> str:
    cap = fig.get("caption", "").lower()
    para = fig.get("paragraph_text", "").lower()
    combined = f"{cap} {para}"

    if any(k in combined for k in ["bar", "histogram", "column"]):
        return "bar_chart"
    elif any(k in combined for k in ["scatter", "correlation", "embedding", "distribution"]):
        return "scatter_plot"
    elif any(k in combined for k in ["heatmap", "matrix", "confusion matrix", "attention"]):
        return "heatmap"
    elif any(k in combined for k in ["diagram", "architecture", "schematic", "workflow"]):
        return "diagram"
    else:
        return "line_plot"  # majority category in SciCap


def compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)

    # 95% bootstrap CI on F1
    n = len(y_true)
    if n > 1 and tp + fn > 0:
        def stat_fn(idx):
            yt = y_true[idx]
            yp = y_pred[idx]
            c_tp = np.sum((yp == 1) & (yt == 1))
            c_fp = np.sum((yp == 1) & (yt == 0))
            c_fn = np.sum((yp == 0) & (yt == 1))
            cp = c_tp / max(c_tp + c_fp, 1)
            cr = c_tp / max(c_tp + c_fn, 1)
            return 2 * cp * cr / max(cp + cr, 1e-9)

        diffs = paired_bootstrap(stat_fn, n, n_resamples=2000, seed=42)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
    else:
        lo, hi = f1, f1

    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "ci95_low": round(float(lo), 4),
        "ci95_high": round(float(hi), 4),
        "tp": tp, "fp": fp, "fn": fn,
        "num_pairs": n
    }


def main():
    print("=" * 65)
    print("PHASE 6: FINE-GRAINED BREAKDOWN BY ELEMENT & FIGURE TYPE")
    print("=" * 65)

    with open(BENCHMARK_PATH) as f:
        data = json.load(f)

    val_figs = [f for f in data["figures"] if f.get("split") == "val"]
    test_figs = [f for f in data["figures"] if f.get("split") == "test"]

    # 1. Fit reference Logistic Regression model on validation pairs
    X_train, y_train = [], []
    for f in val_figs:
        para = f.get("paragraph_text", "")
        links = f.get("silver_links") or f.get("gold_links", [])
        gold = {(l["element_id"], l["entity"]) for l in links}
        elements = [e for e in f.get("elements", []) if e.get("type") in TEXT_ELEMENT_TYPES]
        entities = f.get("entities", [])
        g = build_grounding_graph(elements, entities)
        for el in elements:
            for en in entities:
                rule_edge = g.has_edge(f"elem:{el['id']}", f"ent:{en['text']}")
                X_train.append(pair_features(el, en, para, rule_edge))
                y_train.append(int((el["id"], en["text"]) in gold))

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(np.array(X_train), np.array(y_train))
    threshold = 0.950  # Tuned validation threshold from Phase 2

    # 2. Extract test pairs with metadata
    by_elem_type = defaultdict(lambda: {"true": [], "pred": []})
    by_fig_type = defaultdict(lambda: {"true": [], "pred": [], "figs": set()})

    for f in test_figs:
        ftype = infer_figure_type(f)
        by_fig_type[ftype]["figs"].add(f["figure_id"])

        para = f.get("paragraph_text", "")
        links = f.get("silver_links") or f.get("gold_links", [])
        gold = {(l["element_id"], l["entity"]) for l in links}
        elements = [e for e in f.get("elements", []) if e.get("type") in TEXT_ELEMENT_TYPES]
        entities = f.get("entities", [])
        g = build_grounding_graph(elements, entities)

        for el in elements:
            etype = el.get("type", "LABEL")
            for en in entities:
                rule_edge = g.has_edge(f"elem:{el['id']}", f"ent:{en['text']}")
                feat = pair_features(el, en, para, rule_edge)
                prob = lr.predict_proba([feat])[0, 1]
                pred = int(prob >= threshold)
                is_gold = int((el["id"], en["text"]) in gold)

                by_elem_type[etype]["true"].append(is_gold)
                by_elem_type[etype]["pred"].append(pred)

                by_fig_type[ftype]["true"].append(is_gold)
                by_fig_type[ftype]["pred"].append(pred)

    # 3. Summarize breakdown by Element Type
    print("\n--- Breakdown by Element Role ---")
    elem_results = {}
    for etype in ["AXIS", "LEGEND", "LABEL"]:
        yt = np.array(by_elem_type[etype]["true"])
        yp = np.array(by_elem_type[etype]["pred"])
        m = compute_group_metrics(yt, yp)
        elem_results[etype] = m
        print(f"  {etype:<10} F1: {m['f1']:.4f} [{m['ci95_low']}, {m['ci95_high']}] (P={m['precision']:.4f}, R={m['recall']:.4f}, N_pairs={m['num_pairs']})")

    # 4. Summarize breakdown by Figure Type
    print("\n--- Breakdown by Figure Type ---")
    fig_results = {}
    for ftype, d in by_fig_type.items():
        yt = np.array(d["true"])
        yp = np.array(d["pred"])
        n_figs = len(d["figs"])
        m = compute_group_metrics(yt, yp)
        m["num_figures"] = n_figs
        m["small_sample_flag"] = bool(n_figs < 20)
        fig_results[ftype] = m
        flag = " [WARNING: N < 20]" if n_figs < 20 else ""
        print(f"  {ftype:<15} (N={n_figs:2d}{flag}): F1: {m['f1']:.4f} [{m['ci95_low']}, {m['ci95_high']}] (P={m['precision']:.4f}, R={m['recall']:.4f})")

    out_data = {
        "provenance": "silver_heuristic",
        "breakdown_by_element_role": elem_results,
        "breakdown_by_figure_type": fig_results,
        **get_runtime_metadata("pipelines/evaluate_breakdown_by_type.py")
    }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nSaved fine-grained breakdown report to: {OUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
