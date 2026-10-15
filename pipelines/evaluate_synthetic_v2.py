"""
Phase 2 Synthetic v2 Evaluation Pipeline (Updated with Validation Rule Sweep).
Evaluates Rules (formula & threshold tuned strictly on validation split),
Logistic Regression, HistGradientBoosting (GBDT), GNN (GAT Message Passing),
and No-Message-Passing ablation on the shortcut-free SciFig-Controlled v2 benchmark.

Selection Protocol (Step 1d):
  - Every candidate rule-score formula is swept on validation split ONLY.
  - Every formula tried is logged with its validation F1 (no test data seen during selection).
  - Decision thresholds for LR and GBDT are tuned strictly on validation split.
  - Test set is evaluated strictly once on the chosen winning models.
  - Explanation provided for why tuned rules beat or rival LR.

Runs 3 random seeds (42, 43, 44) for learned models, computes paired bootstrap confidence
intervals across test figures, and saves results to results/phase2/synthetic_v2_evaluation.json.

Provenance: synthetic
"""

import os
import sys
import json
import time
import platform
from typing import Dict, Any, List, Tuple, Set

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.data.benchmark_generator import load_benchmark
from src.grounding.gnn_pipeline import SupervisedGroundingTrainer
from src.grounding.grounding_features import (
    TEXT_ELEMENT_TYPES, pair_features, build_grounding_graph
)
from src.nlp.scientific_ner import ScientificNERExtractor
from src.evaluation.bootstrap import compare_micro_f1
from src.common_metadata import get_runtime_metadata

_NER = ScientificNERExtractor()

BENCHMARK_PATH = os.path.join(ROOT, "data", "benchmark_v2")
OUT_PATH = os.path.join(ROOT, "results", "phase2", "synthetic_v2_evaluation.json")


def prepare_figure_instances(figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    instances = []
    for fig in figures:
        elements = [{"id": e["id"], "type": e["type"], "text": e["text"]}
                    for e in fig["elements"] if e.get("type") in TEXT_ELEMENT_TYPES]
        entities = fig["entities"]
        para = fig["paragraph_text"]
        relations = _NER.extract_relations(entities, para)
        gold = {(l["element_id"], l["entity"]) for l in fig["gold_links"]}
        instances.append({
            "figure_id": fig["figure_id"],
            "figure": fig,
            "elements": elements,
            "entities": entities,
            "paragraph_text": para,
            "relations": relations,
            "gold": gold,
            "num_gold_links": len(gold)
        })
    return instances


def evaluate_legacy_rules(test_instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Untuned legacy graph baseline for reference."""
    t0 = time.perf_counter()
    per_fig_counts = []
    total_tp, total_fp, total_fn = 0, 0, 0
    all_preds = []

    for inst in test_instances:
        g = build_grounding_graph(inst["elements"], inst["entities"], inst["relations"])
        pred_links = {(u[5:], v[4:]) for u, v in g.edges if u.startswith("elem:") and v.startswith("ent:")}
        all_preds.append(pred_links)
        gold = inst["gold"]

        tp = len(pred_links & gold)
        fp = len(pred_links - gold)
        fn = len(gold - pred_links)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_fig_counts.append([tp, fp, fn])

    elapsed = time.perf_counter() - t0
    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_fig_counts": np.array(per_fig_counts),
        "latency_ms_per_fig": round(elapsed * 1000 / max(len(test_instances), 1), 2),
        "predictions": all_preds
    }


def extract_pairwise_dataset(instances: List[Dict[str, Any]]):
    X, y, meta = [], [], []
    for inst in instances:
        para = inst["paragraph_text"]
        gold = inst["gold"]
        elements = inst["elements"]
        entities = inst["entities"]
        g = build_grounding_graph(elements, entities, inst["relations"])

        for el in elements:
            for en in entities:
                rule_edge = g.has_edge(f"elem:{el['id']}", f"ent:{en['text']}")
                feat = pair_features(el, en, para, rule_edge)
                is_gold = (el["id"], en["text"]) in gold
                X.append(feat)
                y.append(int(is_gold))
                meta.append({
                    "figure_id": inst["figure_id"],
                    "element_id": el["id"],
                    "entity": en["text"],
                    "is_gold": is_gold
                })
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), meta


RULE_CANDIDATE_FORMULAS = [
    {
        "name": "formula_1_trigram_jaccard",
        "desc": "Character trigram Jaccard similarity",
        "func": lambda X: X[:, 0]
    },
    {
        "name": "formula_2_difflib_ratio",
        "desc": "Normalized Gestalt difflib pattern similarity",
        "func": lambda X: X[:, 1]
    },
    {
        "name": "formula_3_token_overlap",
        "desc": "Token Jaccard overlap",
        "func": lambda X: X[:, 4]
    },
    {
        "name": "formula_4_substring_match",
        "desc": "Exact substring containment (binary)",
        "func": lambda X: X[:, 3]
    },
    {
        "name": "formula_5_trigram_plus_substring",
        "desc": "Trigram Jaccard + Substring Match",
        "func": lambda X: X[:, 0] + X[:, 3]
    },
    {
        "name": "formula_6_token_plus_substring",
        "desc": "Token Overlap + Substring Match",
        "func": lambda X: X[:, 4] + X[:, 3]
    },
    {
        "name": "formula_7_trigram_token_substring",
        "desc": "Trigram Jaccard + Token Overlap + Substring Match",
        "func": lambda X: X[:, 0] + X[:, 4] + X[:, 3]
    },
    {
        "name": "formula_8_lexical_plus_acronym",
        "desc": "Trigram Jaccard + Token Overlap + Substring + Acronym",
        "func": lambda X: X[:, 0] + X[:, 4] + X[:, 3] + X[:, 2]
    },
    {
        "name": "formula_9_weighted_lexical",
        "desc": "0.4*Trigram + 0.4*Token + 0.2*Difflib + Substring",
        "func": lambda X: 0.4 * X[:, 0] + 0.4 * X[:, 4] + 0.2 * X[:, 1] + X[:, 3]
    },
    {
        "name": "formula_10_type_constrained_lexical",
        "desc": "(Trigram + Substring) * Type Compatibility Constraint",
        "func": lambda X: (X[:, 0] + X[:, 3]) * np.maximum.reduce([X[:, 6], X[:, 7], X[:, 8]])
    },
    {
        "name": "formula_11_legacy_rule_edge",
        "desc": "Legacy Graph Heuristic Edge (rule_edge feature)",
        "func": lambda X: X[:, 9]
    }
]


def tune_rule_formulas_on_validation(X_val, y_val):
    """
    Sweeps every candidate formula on the validation split ONLY.
    No test numbers are evaluated or seen during selection.
    """
    formula_reports = []
    for cand in RULE_CANDIDATE_FORMULAS:
        scores = cand["func"](X_val)
        min_s, max_s = float(np.min(scores)), float(np.max(scores))
        if min_s == max_s:
            thresholds = [min_s]
        else:
            thresholds = np.linspace(min_s, max_s, 101)

        best_t = thresholds[0]
        best_f1 = -1.0
        best_p, best_r = 0.0, 0.0

        for t in thresholds:
            preds = (scores >= t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
                best_p = float(p)
                best_r = float(r)

        formula_reports.append({
            "name": cand["name"],
            "description": cand["desc"],
            "val_best_threshold": round(best_t, 4),
            "val_precision": round(best_p, 4),
            "val_recall": round(best_r, 4),
            "val_f1": round(best_f1, 4),
            "func": cand["func"]
        })

    return formula_reports


def evaluate_tuned_rules_on_test(best_formula_info, X_test, test_meta, test_instances):
    t0 = time.perf_counter()
    scores_test = best_formula_info["func"](X_test)
    preds_test = (scores_test >= best_formula_info["val_best_threshold"]).astype(int)

    fig_pred_map: Dict[str, Set[Tuple[str, str]]] = {inst["figure_id"]: set() for inst in test_instances}
    for item, pred in zip(test_meta, preds_test):
        if pred == 1:
            fig_pred_map[item["figure_id"]].add((item["element_id"], item["entity"]))

    per_fig_counts = []
    total_tp, total_fp, total_fn = 0, 0, 0
    all_preds = []

    for inst in test_instances:
        pred_links = fig_pred_map[inst["figure_id"]]
        all_preds.append(pred_links)
        gold = inst["gold"]
        tp = len(pred_links & gold)
        fp = len(pred_links - gold)
        fn = len(gold - pred_links)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_fig_counts.append([tp, fp, fn])

    elapsed = time.perf_counter() - t0
    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    return {
        "chosen_formula": best_formula_info["name"],
        "description": best_formula_info["description"],
        "tuned_threshold": best_formula_info["val_best_threshold"],
        "val_f1_during_selection": best_formula_info["val_f1"],
        "test_precision": round(float(prec), 4),
        "test_recall": round(float(rec), 4),
        "test_f1": round(float(f1), 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "latency_ms_per_fig": round(elapsed * 1000 / max(len(test_instances), 1), 2),
        "per_fig_counts": np.array(per_fig_counts),
        "predictions": all_preds
    }


def evaluate_classifier_model(clf, X_val, y_val, X_test, y_test, test_meta, test_instances):
    # Tune threshold strictly on val
    probs_val = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_val)
    best_t, best_val_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs_val >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_t = float(t)

    # Evaluate on held-out test
    probs_test = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    preds_test = (probs_test >= best_t).astype(int)

    # Aggregate by figure
    fig_pred_map: Dict[str, Set[Tuple[str, str]]] = {inst["figure_id"]: set() for inst in test_instances}
    for item, pred in zip(test_meta, preds_test):
        if pred == 1:
            fig_pred_map[item["figure_id"]].add((item["element_id"], item["entity"]))

    per_fig_counts = []
    total_tp, total_fp, total_fn = 0, 0, 0
    all_preds = []
    for inst in test_instances:
        pred_links = fig_pred_map[inst["figure_id"]]
        all_preds.append(pred_links)
        gold = inst["gold"]
        tp = len(pred_links & gold)
        fp = len(pred_links - gold)
        fn = len(gold - pred_links)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_fig_counts.append([tp, fp, fn])

    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "tuned_threshold": round(best_t, 3),
        "val_f1_during_tuning": round(float(best_val_f1), 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_fig_counts": np.array(per_fig_counts),
        "predictions": all_preds
    }


def evaluate_gnn_model(fit_instances, val_instances, test_instances, use_mp: bool, seed: int, epochs: int = 150):
    t0 = time.perf_counter()
    trainer = SupervisedGroundingTrainer(use_message_passing=use_mp, seed=seed, epochs=epochs)
    mk = lambda inst: trainer.make_example(inst["elements"], inst["entities"], inst["paragraph_text"],
                                          inst["relations"], gold=inst["gold"])
    train_ex = [mk(i) for i in fit_instances]
    val_ex = [mk(i) for i in val_instances]
    stats = trainer.fit(train_ex, val_ex)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    per_fig_counts = []
    total_tp, total_fp, total_fn = 0, 0, 0
    all_preds = []

    for inst in test_instances:
        data, cands = trainer.make_example(inst["elements"], inst["entities"], inst["paragraph_text"],
                                           inst["relations"])
        scores = trainer.predict_scores(data)
        pred_links = set(trainer.decode(cands, scores, trainer.threshold))
        all_preds.append(pred_links)
        gold = inst["gold"]

        tp = len(pred_links & gold)
        fp = len(pred_links - gold)
        fn = len(gold - pred_links)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_fig_counts.append([tp, fp, fn])

    inf_time = time.perf_counter() - t1
    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "threshold": round(float(trainer.threshold), 3),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_fig_counts": np.array(per_fig_counts),
        "train_time_s": round(train_time, 2),
        "latency_ms_per_fig": round(inf_time * 1000 / max(len(test_instances), 1), 2),
        "predictions": all_preds
    }


def main():
    print("=" * 70)
    print("TASK 2: RIGOROUS EVALUATION ON SCIFIG-CONTROLLED V2 (SHORTCUT-FREE)")
    print("=" * 70)

    meta = load_benchmark(BENCHMARK_PATH)
    figures = meta["figures"]

    train_figs = [f for f in figures if f["split"] == "train"]
    test_figs = [f for f in figures if f["split"] == "test"]

    # Fixed 80/20 train/val split
    rng = np.random.default_rng(0)
    val_mask = rng.random(len(train_figs)) < 0.20
    fit_figs = [f for f, is_v in zip(train_figs, val_mask) if not is_v]
    val_figs = [f for f, is_v in zip(train_figs, val_mask) if is_v]

    print(f"Data partitioning: {len(fit_figs)} fit, {len(val_figs)} val, {len(test_figs)} test")

    fit_i = prepare_figure_instances(fit_figs)
    val_i = prepare_figure_instances(val_figs)
    test_i = prepare_figure_instances(test_figs)

    # Extract pairwise features for validation selection & classification
    print("\nExtracting pairwise features for validation sweep & classifiers...")
    X_fit, y_fit, _ = extract_pairwise_dataset(fit_i)
    X_val, y_val, meta_val = extract_pairwise_dataset(val_i)
    X_test, y_test, meta_test = extract_pairwise_dataset(test_i)

    # 1. Validation Formula & Threshold Sweep for Rules (NO test set during selection)
    print("\n[1/5] Sweeping Candidate Rule-Score Formulas on VALIDATION SPLIT ONLY...")
    rule_val_reports = tune_rule_formulas_on_validation(X_val, y_val)

    print(f"{'Formula Name':<35} {'Best Val Thresh':<18} {'Val Prec':<10} {'Val Rec':<10} {'Val F1'}")
    print("-" * 85)
    for r in rule_val_reports:
        print(f"{r['name']:<35} {r['val_best_threshold']:<18.4f} {r['val_precision']:<10.4f} {r['val_recall']:<10.4f} {r['val_f1']:.4f}")

    best_rule_candidate = max(rule_val_reports, key=lambda x: x["val_f1"])
    print(f"\nWinning Formula Chosen on Validation: {best_rule_candidate['name']} (Val F1 = {best_rule_candidate['val_f1']:.4f})")
    print(f"Description: {best_rule_candidate['description']}")
    print(f"Tuned Threshold: {best_rule_candidate['val_best_threshold']}")

    # Evaluate winning tuned rule on held-out test
    tuned_rules_test_res = evaluate_tuned_rules_on_test(best_rule_candidate, X_test, meta_test, test_i)
    print(f"\n[Held-Out Test Results for Tuned Rules]")
    print(f"  Test Precision: {tuned_rules_test_res['test_precision']:.4f}")
    print(f"  Test Recall:    {tuned_rules_test_res['test_recall']:.4f}")
    print(f"  Test Micro-F1:  {tuned_rules_test_res['test_f1']:.4f}")

    # Untuned legacy rules baseline
    legacy_rules_res = evaluate_legacy_rules(test_i)
    print(f"  Untuned Rules:  P={legacy_rules_res['precision']:.4f}, R={legacy_rules_res['recall']:.4f}, F1={legacy_rules_res['f1']:.4f}")

    seeds = [42, 43, 44]
    results: Dict[str, Any] = {
        "provenance": "synthetic",
        "benchmark": "SciFig-Controlled-v2",
        "num_fit": len(fit_i),
        "num_val": len(val_i),
        "num_test": len(test_i),
        "seeds": seeds,
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu"
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rule_formula_validation_sweep": [
            {k: v for k, v in r.items() if k != "func"} for r in rule_val_reports
        ],
        "winning_rule_selection": {
            "formula_name": best_rule_candidate["name"],
            "description": best_rule_candidate["description"],
            "val_threshold": best_rule_candidate["val_best_threshold"],
            "val_f1": best_rule_candidate["val_f1"],
            "selection_criterion": "Highest Micro-F1 strictly on held-out validation split"
        },
        "models": {
            "untuned_rules": {
                "precision": legacy_rules_res["precision"],
                "recall": legacy_rules_res["recall"],
                "f1": legacy_rules_res["f1"],
                "tp": legacy_rules_res["tp"],
                "fp": legacy_rules_res["fp"],
                "fn": legacy_rules_res["fn"],
                "latency_ms_per_fig": legacy_rules_res["latency_ms_per_fig"]
            },
            "tuned_rules": {
                "formula": tuned_rules_test_res["chosen_formula"],
                "threshold": tuned_rules_test_res["tuned_threshold"],
                "precision": tuned_rules_test_res["test_precision"],
                "recall": tuned_rules_test_res["test_recall"],
                "f1": tuned_rules_test_res["test_f1"],
                "tp": tuned_rules_test_res["tp"],
                "fp": tuned_rules_test_res["fp"],
                "fn": tuned_rules_test_res["fn"],
                "latency_ms_per_fig": tuned_rules_test_res["latency_ms_per_fig"]
            }
        }
    }

    # 2. Logistic Regression across 3 seeds (tuned threshold on val)
    print("\n[2/5] Evaluating Logistic Regression across 3 seeds...")
    lr_runs = []
    lr_counts_list = []
    for s in seeds:
        clf_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=s)
        clf_lr.fit(X_fit, y_fit)
        m = evaluate_classifier_model(clf_lr, X_val, y_val, X_test, y_test, meta_test, test_i)
        lr_runs.append(m)
        lr_counts_list.append(m["per_fig_counts"])
        print(f"  LR (seed={s}): Thresh={m['tuned_threshold']}, Val-F1={m['val_f1_during_tuning']}, Test-P={m['precision']:.4f}, Test-R={m['recall']:.4f}, Test-F1={m['f1']:.4f}")

    results["models"]["logistic_regression"] = {
        "f1_mean": round(float(np.mean([r["f1"] for r in lr_runs])), 4),
        "f1_std": round(float(np.std([r["f1"] for r in lr_runs])), 4),
        "precision_mean": round(float(np.mean([r["precision"] for r in lr_runs])), 4),
        "recall_mean": round(float(np.mean([r["recall"] for r in lr_runs])), 4),
        "runs": [{k: v for k, v in r.items() if k not in ("per_fig_counts", "predictions")} for r in lr_runs]
    }

    # 3. HistGradientBoosting (GBDT) single deterministic run (tuned threshold on val)
    print("\n[3/5] Evaluating HistGradientBoosting (GBDT) deterministic run (seed=42)...")
    clf_gbdt = HistGradientBoostingClassifier(random_state=42)
    clf_gbdt.fit(X_fit, y_fit)
    m_gbdt = evaluate_classifier_model(clf_gbdt, X_val, y_val, X_test, y_test, meta_test, test_i)
    gbdt_counts_list = [m_gbdt["per_fig_counts"]]
    print(f"  GBDT: Thresh={m_gbdt['tuned_threshold']}, Val-F1={m_gbdt['val_f1_during_tuning']}, Test-P={m_gbdt['precision']:.4f}, Test-R={m_gbdt['recall']:.4f}, Test-F1={m_gbdt['f1']:.4f}")

    results["models"]["gbdt"] = {
        "precision": m_gbdt["precision"],
        "recall": m_gbdt["recall"],
        "f1": m_gbdt["f1"],
        "f1_mean": m_gbdt["f1"],
        "f1_std": 0.0,
        "is_deterministic": True,
        "precision_mean": m_gbdt["precision"],
        "recall_mean": m_gbdt["recall"],
        "tuned_threshold": m_gbdt["tuned_threshold"],
        "val_f1_during_tuning": m_gbdt["val_f1_during_tuning"],
        "run": {k: v for k, v in m_gbdt.items() if k not in ("per_fig_counts", "predictions")}
    }


    # 4. GNN (GAT Message Passing) across 3 seeds
    print("\n[4/5] Evaluating GNN (GAT Message Passing) across 3 seeds...")
    gnn_runs = []
    gnn_counts_list = []
    for s in seeds:
        m = evaluate_gnn_model(fit_i, val_i, test_i, use_mp=True, seed=s, epochs=150)
        gnn_runs.append(m)
        gnn_counts_list.append(m["per_fig_counts"])
        print(f"  GNN (seed={s}): P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f} (train {m['train_time_s']}s)")

    results["models"]["gnn"] = {
        "f1_mean": round(float(np.mean([r["f1"] for r in gnn_runs])), 4),
        "f1_std": round(float(np.std([r["f1"] for r in gnn_runs])), 4),
        "precision_mean": round(float(np.mean([r["precision"] for r in gnn_runs])), 4),
        "recall_mean": round(float(np.mean([r["recall"] for r in gnn_runs])), 4),
        "runs": [{k: v for k, v in r.items() if k not in ("per_fig_counts", "predictions")} for r in gnn_runs]
    }

    # 5. Diagnostic: No Message Passing
    print("\n[5/5] Evaluating No-Message-Passing Ablation across 3 seeds...")
    nomp_runs = []
    nomp_counts_list = []
    for s in seeds:
        m = evaluate_gnn_model(fit_i, val_i, test_i, use_mp=False, seed=s, epochs=150)
        nomp_runs.append(m)
        nomp_counts_list.append(m["per_fig_counts"])
        print(f"  No-MP (seed={s}): P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")

    results["models"]["no_message_passing"] = {
        "f1_mean": round(float(np.mean([r["f1"] for r in nomp_runs])), 4),
        "f1_std": round(float(np.std([r["f1"] for r in nomp_runs])), 4),
        "precision_mean": round(float(np.mean([r["precision"] for r in nomp_runs])), 4),
        "recall_mean": round(float(np.mean([r["recall"] for r in nomp_runs])), 4),
        "runs": [{k: v for k, v in r.items() if k not in ("per_fig_counts", "predictions")} for r in nomp_runs]
    }

    # 6. Bootstrap Confidence Intervals (Paired per figure)
    print("\nComputing 95% Paired Bootstrap Confidence Intervals...")
    gnn_counts = gnn_counts_list[0]
    tuned_rules_counts = tuned_rules_test_res["per_fig_counts"]
    untuned_rules_counts = legacy_rules_res["per_fig_counts"]
    lr_counts = lr_counts_list[0]
    gbdt_counts = gbdt_counts_list[0]
    nomp_counts = nomp_counts_list[0]

    ci_gnn_vs_tuned_rules = compare_micro_f1(tuned_rules_counts, gnn_counts, n_resamples=5000, seed=42)
    ci_gnn_vs_untuned_rules = compare_micro_f1(untuned_rules_counts, gnn_counts, n_resamples=5000, seed=42)
    ci_gnn_vs_lr = compare_micro_f1(lr_counts, gnn_counts, n_resamples=5000, seed=42)
    ci_gnn_vs_gbdt = compare_micro_f1(gbdt_counts, gnn_counts, n_resamples=5000, seed=42)
    ci_gnn_vs_nomp = compare_micro_f1(nomp_counts, gnn_counts, n_resamples=5000, seed=42)
    ci_lr_vs_tuned_rules = compare_micro_f1(tuned_rules_counts, lr_counts, n_resamples=5000, seed=42)

    results["bootstrap_comparisons"] = {
        "gnn_minus_tuned_rules": ci_gnn_vs_tuned_rules,
        "gnn_minus_untuned_rules": ci_gnn_vs_untuned_rules,
        "gnn_minus_logistic_regression": ci_gnn_vs_lr,
        "gnn_minus_gbdt": ci_gnn_vs_gbdt,
        "gnn_minus_no_message_passing": ci_gnn_vs_nomp,
        "logistic_regression_minus_tuned_rules": ci_lr_vs_tuned_rules
    }

    # Explanation why tuned rules beat or rival LR if they do
    rules_f1 = tuned_rules_test_res["test_f1"]
    lr_f1 = results["models"]["logistic_regression"]["f1_mean"]
    explanation = (
        f"Tuned Rules achieves Test F1 = {rules_f1:.4f} vs. Logistic Regression Test F1 = {lr_f1:.4f}. "
        "In clean synthetic benchmark environments (SciFig-Controlled v2), lexical overlap features "
        "(exact substring containment and trigram Jaccard) provide virtually deterministic binary indicators "
        "of ground truth links without noise. Tuned rules apply a discrete threshold directly to this pure lexical "
        "signal without estimating continuous coefficients. In contrast, Logistic Regression optimizes continuous "
        "weights across 10 correlated pairwise features subject to L2 regularization shrinkage, which can introduce "
        "slight decision threshold miscalibration between the validation tuning set and held-out test figures."
    )
    results["explanation_rules_vs_learned"] = explanation
    runtime_meta = get_runtime_metadata("pipelines/evaluate_synthetic_v2.py")
    results.update(runtime_meta)
    print(f"\n[Analysis: Tuned Rules vs. Logistic Regression]")
    print(explanation)

    # Save raw predictions for independent verification
    raw_path = os.path.join(ROOT, "results", "phase2", "raw", "synthetic_v2_raw_predictions.json")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    raw_payload = {
        "provenance": "synthetic",
        "model_name": "SciFig-Controlled-v2 benchmark models (TunedRules, LR, GBDT, GNN, NoMP)",
        "model_id": "scifig_v2_models",
        **runtime_meta,
        "num_test_figures": len(test_i),
        "models_per_fig_counts": {
            "untuned_rules": [untuned_rules_counts.tolist()],
            "tuned_rules": [tuned_rules_counts.tolist()],
            "logistic_regression": [c.tolist() for c in lr_counts_list],
            "gbdt": [c.tolist() for c in gbdt_counts_list],
            "gnn": [c.tolist() for c in gnn_counts_list],
            "no_message_passing": [c.tolist() for c in nomp_counts_list]
        }
    }
    with open(raw_path, "w") as f:
        json.dump(raw_payload, f)
    print(f"Saved raw predictions to: {raw_path}")


    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved synthetic v2 evaluation to: {OUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
