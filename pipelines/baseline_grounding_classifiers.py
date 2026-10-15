"""
Strong Grounding Baselines (Phase 2):
1. Tuned Rules: Sweeps threshold on trigram Jaccard + type compatibility on val split,
   evaluates chosen threshold on held-out test split.
2. Logistic Regression (Feature-only): LogisticRegression(class_weight="balanced") on the 10 pairwise features,
   threshold tuned on val, evaluated on test.
3. HistGradientBoosting (Feature-only): Non-linear tree ensemble on the 10 pairwise features,
   threshold tuned on val, evaluated on test.
4. Feature Importance Analysis: Quantifies which pairwise features drive decisions.

Outputs saved to results/phase2/classifier_baselines.json.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_fscore_support

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.grounding.grounding_features import (
    PAIR_FEATURE_NAMES, pair_features, TEXT_ELEMENT_TYPES, build_grounding_graph
)

from src.common_metadata import get_runtime_metadata

GOLD_BENCHMARK = os.path.join(ROOT, "data", "scicap_subset", "silver_grounding_v0.json")
OUT_FILE = os.path.join(ROOT, "results", "appendix_circular", "classifier_baselines.json")


def load_data():
    with open(GOLD_BENCHMARK, "r") as f:
        data = json.load(f)
    val_figs = [f for f in data["figures"] if f.get("split") == "val"]
    test_figs = [f for f in data["figures"] if f.get("split") == "test"]
    return val_figs, test_figs


def extract_pairs(figs: List[Dict[str, Any]]):
    X = []
    y = []
    jaccards = []
    meta = []

    for f in figs:
        para = f.get("paragraph_text", "")
        links = f.get("silver_links") or f.get("gold_links", [])
        gold_pairs = {(l["element_id"], l["entity"]) for l in links}
        elements = [e for e in f.get("elements", []) if e.get("type") in TEXT_ELEMENT_TYPES]
        entities = f.get("entities", [])

        # Build rule graph for rule_edge feature
        g = build_grounding_graph(elements, entities)

        for el in elements:
            for en in entities:
                rule_edge = g.has_edge(f"elem:{el['id']}", f"ent:{en['text']}")
                feat = pair_features(el, en, para, rule_edge)
                is_gold = (el["id"], en["text"]) in gold_pairs

                X.append(feat)
                y.append(int(is_gold))
                # Trigram jaccard is feature 0
                jaccards.append(feat[0])
                meta.append({
                    "figure_id": f["figure_id"],
                    "element_id": el["id"],
                    "entity": en["text"],
                    "is_gold": is_gold
                })

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), np.array(jaccards, dtype=np.float32), meta


def evaluate_threshold(probs: np.ndarray, y_true: np.ndarray, threshold: float):
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4),
            "tp": tp, "fp": fp, "fn": fn}


def tune_threshold(probs_val: np.ndarray, y_val: np.ndarray):
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.01, 0.99, 100):
        preds = (probs_val >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def main():
    print("=" * 65)
    print("PHASE 2: STRONG GROUNDING BASELINES (FEATURE CLASSIFIERS)")
    print("=" * 65)

    val_figs, test_figs = load_data()
    print(f"Loaded {len(val_figs)} val figures, {len(test_figs)} test figures.")

    X_val, y_val, jacc_val, _ = extract_pairs(val_figs)
    X_test, y_test, jacc_test, _ = extract_pairs(test_figs)

    print(f"Validation pairs: {len(y_val)} (positives: {int(y_val.sum())})")
    print(f"Test pairs:       {len(y_test)} (positives: {int(y_test.sum())})")

    results: Dict[str, Any] = {
        "provenance": "silver_heuristic",
        "num_val_pairs": len(y_val),
        "num_test_pairs": len(y_test),
        "num_val_positives": int(y_val.sum()),
        "num_test_positives": int(y_test.sum()),
        "baselines": {},
        **get_runtime_metadata("pipelines/baseline_grounding_classifiers.py")
    }


    # -------------------------------------------------------------
    # 1. Tuned Rules (Trigram Jaccard)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    best_t_rules, val_f1_rules = tune_threshold(jacc_val, y_val)
    time_rules = (time.perf_counter() - t0) * 1000
    test_metrics_rules = evaluate_threshold(jacc_test, y_test, best_t_rules)
    print(f"\n[Baseline 1: Tuned Rules]")
    print(f"  Optimal Val Threshold: {best_t_rules:.3f} (Val F1: {val_f1_rules:.4f})")
    print(f"  Test F1:               {test_metrics_rules['f1']:.4f} (P={test_metrics_rules['precision']:.4f}, R={test_metrics_rules['recall']:.4f})")

    results["baselines"]["tuned_rules"] = {
        "val_threshold": round(best_t_rules, 4),
        "val_f1": round(val_f1_rules, 4),
        "test_metrics": test_metrics_rules,
        "latency_ms": round(time_rules / len(test_figs), 2)
    }

    # -------------------------------------------------------------
    # 2. Logistic Regression (Feature-only)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_val, y_val)
    train_time_lr = time.perf_counter() - t0

    probs_val_lr = lr.predict_proba(X_val)[:, 1]
    best_t_lr, val_f1_lr = tune_threshold(probs_val_lr, y_val)

    t0 = time.perf_counter()
    probs_test_lr = lr.predict_proba(X_test)[:, 1]
    test_lat_lr = (time.perf_counter() - t0) * 1000 / len(test_figs)

    test_metrics_lr = evaluate_threshold(probs_test_lr, y_test, best_t_lr)
    print(f"\n[Baseline 2: Logistic Regression (10 Features)]")
    print(f"  Optimal Val Threshold: {best_t_lr:.3f} (Val F1: {val_f1_lr:.4f})")
    print(f"  Test F1:               {test_metrics_lr['f1']:.4f} (P={test_metrics_lr['precision']:.4f}, R={test_metrics_lr['recall']:.4f})")
    print(f"  Inference Latency:     {test_lat_lr:.2f} ms/figure")

    # Feature weights
    lr_coefs = {name: round(float(coef), 4) for name, coef in zip(PAIR_FEATURE_NAMES, lr.coef_[0])}
    sorted_lr = sorted(lr_coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    print("  Top Logistic Regression Features by Magnitude:")
    for feat_name, coef in sorted_lr[:5]:
        print(f"    - {feat_name:35s}: {coef:+.4f}")

    results["baselines"]["logistic_regression"] = {
        "val_threshold": round(best_t_lr, 4),
        "val_f1": round(val_f1_lr, 4),
        "test_metrics": test_metrics_lr,
        "latency_ms": round(test_lat_lr, 2),
        "train_time_s": round(train_time_lr, 3),
        "coefficients": lr_coefs
    }

    # -------------------------------------------------------------
    # 3. HistGradientBoostingClassifier (Feature-only)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    # Compute class weights for sample_weight
    pos_weight = (len(y_val) - y_val.sum()) / max(y_val.sum(), 1)
    sample_weights = np.where(y_val == 1, pos_weight, 1.0)
    hgb = HistGradientBoostingClassifier(random_state=42, max_iter=100)
    hgb.fit(X_val, y_val, sample_weight=sample_weights)
    train_time_hgb = time.perf_counter() - t0

    probs_val_hgb = hgb.predict_proba(X_val)[:, 1]
    best_t_hgb, val_f1_hgb = tune_threshold(probs_val_hgb, y_val)

    t0 = time.perf_counter()
    probs_test_hgb = hgb.predict_proba(X_test)[:, 1]
    test_lat_hgb = (time.perf_counter() - t0) * 1000 / len(test_figs)

    test_metrics_hgb = evaluate_threshold(probs_test_hgb, y_test, best_t_hgb)
    print(f"\n[Baseline 3: HistGradientBoosting (10 Features)]")
    print(f"  Optimal Val Threshold: {best_t_hgb:.3f} (Val F1: {val_f1_hgb:.4f})")
    print(f"  Test F1:               {test_metrics_hgb['f1']:.4f} (P={test_metrics_hgb['precision']:.4f}, R={test_metrics_hgb['recall']:.4f})")
    print(f"  Inference Latency:     {test_lat_hgb:.2f} ms/figure")

    # Permutation importance
    perm = permutation_importance(hgb, X_test, y_test, n_repeats=5, random_state=42)
    hgb_importances = {name: round(float(imp), 4) for name, imp in zip(PAIR_FEATURE_NAMES, perm.importances_mean)}
    sorted_hgb = sorted(hgb_importances.items(), key=lambda x: x[1], reverse=True)
    print("  Top HistGradientBoosting Features by Permutation Importance:")
    for feat_name, imp in sorted_hgb[:5]:
        print(f"    - {feat_name:35s}: {imp:.4f}")

    results["baselines"]["gradient_boosting"] = {
        "val_threshold": round(best_t_hgb, 4),
        "val_f1": round(val_f1_hgb, 4),
        "test_metrics": test_metrics_hgb,
        "latency_ms": round(test_lat_hgb, 2),
        "train_time_s": round(train_time_hgb, 3),
        "permutation_importances": hgb_importances
    }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved classifier baselines report to: {OUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
