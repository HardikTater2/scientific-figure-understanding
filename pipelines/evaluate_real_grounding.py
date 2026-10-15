"""
Real-Figure Grounding Evaluator (Phase 1).
Evaluates grounding methods on the real-world SciCap-Grounding-Gold test set:
  A. Rule-Based Grounding (EntityGroundingGraph / lexical overlap heuristics)
  B. GNN Grounding (PairGroundingModel with GAT message passing)
  C. Diagnostic No-Message-Passing (Identical architecture with MP disabled)

Evaluated under two settings:
  1. Oracle Elements: Evaluates pure grounding ability given gold visual elements and gold entities.
  2. End-to-End: Evaluates realistic end-to-end performance using figure OCR and NER extractions.

Output is saved to results/phase1/real_grounding_results.json.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Set, Tuple

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.grounding.grounding_features import TEXT_ELEMENT_TYPES, build_grounding_graph
from src.grounding.gnn_pipeline import SupervisedGroundingTrainer
from src.evaluation.figure_extraction_eval import match_elements
from src.nlp.modern_scientific_ner import ModernScientificNER
from src.vision.ocr_engine import OCREngine
from src.vision.figure_processor import FigureProcessor
from src.evaluation.bootstrap import compare_micro_f1, compare_means
from src.common_metadata import get_runtime_metadata

GOLD_BENCHMARK = os.path.join(ROOT, "data", "scicap_subset", "silver_grounding_v0.json")
OCR_CACHE = os.path.join(ROOT, "data", "scicap_subset", "cache", "ocr.json")
OUT_FILE = os.path.join(ROOT, "results", "appendix_circular", "real_grounding_results.json")


def load_dataset():
    with open(GOLD_BENCHMARK, "r") as f:
        data = json.load(f)
    val_figs = [f for f in data["figures"] if f.get("split") == "val"]
    test_figs = [f for f in data["figures"] if f.get("split") == "test"]
    return val_figs, test_figs


def build_instance(fig: Dict[str, Any], setting: str, processor: FigureProcessor, ner: ModernScientificNER):
    links = fig.get("silver_links") or fig.get("gold_links", [])
    gold_by_elem = {l["element_id"]: l["entity"] for l in links}
    num_gold_links = len(links)
    para = fig.get("paragraph_text", "")
    cap = fig.get("caption", "")
    full_text = f"{cap} {para}"

    if setting == "oracle":
        elements = fig.get("elements", [])
        entities = fig.get("entities", [])
        gold_pairs = {(eid, ent) for eid, ent in gold_by_elem.items()}
    else:
        # End-to-end: Extract elements via OCR and entities via NER
        img_path = os.path.join(ROOT, "data", "scicap_subset", fig["image_path"])
        fig_out = processor.process_figure(img_path)
        elements = [e for e in fig_out["elements"] if e["type"] in TEXT_ELEMENT_TYPES]
        raw_ents = ner.extract_entities(full_text)
        entities = [{"id": f"ent_{i}", "text": e["text"], "type": e["type"]} for i, e in enumerate(raw_ents)]

        gold_pairs = set()
        for i, j, _ in match_elements(elements, fig.get("elements", [])):
            gold_elem_id = fig["elements"][j]["id"]
            if gold_elem_id in gold_by_elem:
                gold_pairs.add((elements[i]["id"], gold_by_elem[gold_elem_id]))

    relations = ner.extract_relations(entities, full_text)
    return {
        "figure_id": fig["figure_id"],
        "paragraph_text": para,
        "elements": elements,
        "entities": entities,
        "relations": relations,
        "gold": gold_pairs,
        "num_gold_links": num_gold_links
    }


def _expected_rank(scores: Dict[str, float], gold_ent: str) -> float:
    g = scores.get(gold_ent, 0.0)
    greater = sum(1 for s in scores.values() if s > g)
    ties = sum(1 for s in scores.values() if s == g)
    return greater + (ties + 1) / 2.0


def score_predictions(instances: List[Dict[str, Any]], pred_links: List[Set[Tuple[str, str]]],
                      score_maps: List[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    tp = fp = 0
    total_gold = sum(i["num_gold_links"] for i in instances)
    acc1_list = []
    mrr_list = []

    for inst, preds, sc in zip(instances, pred_links, score_maps):
        tp += len(preds & inst["gold"])
        fp += len(preds - inst["gold"])

        for eid, ent in inst["gold"]:
            scores_for_el = sc.get(eid, {})
            r = _expected_rank(scores_for_el, ent)
            ties = sum(1 for s in scores_for_el.values() if s == scores_for_el.get(ent, 0.0))
            greater = sum(1 for s in scores_for_el.values() if s > scores_for_el.get(ent, 0.0))
            acc1_list.append((1.0 / ties) if (greater == 0 and ties > 0) else 0.0)
            mrr_list.append(1.0 / r)

        missing = inst["num_gold_links"] - len(inst["gold"])
        acc1_list.extend([0.0] * missing)
        mrr_list.extend([0.0] * missing)

    fn = total_gold - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_gold, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "grounding_accuracy@1": round(float(np.mean(acc1_list)), 4) if acc1_list else 0.0,
        "mrr": round(float(np.mean(mrr_list)), 4) if mrr_list else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "total_gold": int(total_gold)
    }


def evaluate_rules(instances: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
    preds = []
    score_maps = []
    latencies = []

    for inst in instances:
        t0 = time.perf_counter()
        g = build_grounding_graph(inst["elements"], inst["entities"], inst["relations"])
        links = {(u[5:], v[4:]) for u, v in g.edges if u.startswith("elem:") and v.startswith("ent:")}
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        preds.append(links)

        sc = {el["id"]: {en["text"]: float((el["id"], en["text"]) in links) for en in inst["entities"]}
              for el in inst["elements"]}
        score_maps.append(sc)

    metrics = score_predictions(instances, preds, score_maps)
    return metrics, float(np.median(latencies))


def evaluate_learned(train_insts: List[Dict[str, Any]], val_insts: List[Dict[str, Any]],
                     test_insts: List[Dict[str, Any]], use_mp: bool, seed: int = 42, epochs: int = 100):
    trainer = SupervisedGroundingTrainer(use_message_passing=use_mp, seed=seed, epochs=epochs)
    mk = lambda inst: trainer.make_example(inst["elements"], inst["entities"], inst["paragraph_text"],
                                          inst["relations"], gold=inst["gold"])
    train_ex = [mk(i) for i in train_insts]
    val_ex = [mk(i) for i in val_insts]
    stats = trainer.fit(train_ex, val_ex)

    preds = []
    score_maps = []
    latencies = []

    for inst in test_insts:
        t0 = time.perf_counter()
        data, cands = trainer.make_example(inst["elements"], inst["entities"], inst["paragraph_text"],
                                          inst["relations"])
        scores = trainer.predict_scores(data)
        links = set(trainer.decode(cands, scores, trainer.threshold))
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        preds.append(links)

        sc: Dict[str, Dict[str, float]] = {}
        for (eid, ent), score_val in zip(cands, scores):
            sc.setdefault(eid, {})[ent] = float(score_val)
        score_maps.append(sc)

    metrics = score_predictions(test_insts, preds, score_maps)
    return metrics, stats, float(np.median(latencies))


def run_all_evaluations():
    print("=" * 65)
    print("PHASE 1 REAL-WORLD GROUNDING EVALUATION (SCICAP-GROUNDING-GOLD)")
    print("=" * 65)

    ocr = OCREngine(cache_path=OCR_CACHE)
    processor = FigureProcessor(ocr_engine=ocr)
    ner = ModernScientificNER()

    val_figs, test_figs = load_dataset()
    print(f"Loaded {len(val_figs)} val figures, {len(test_figs)} test figures.")

    # We use val split figures for training + validation in the real benchmark
    train_val_split = int(len(val_figs) * 0.70)
    train_figs = val_figs[:train_val_split]
    tuning_figs = val_figs[train_val_split:]

    results: Dict[str, Any] = {
        "dataset": "SciCap-Grounding-Silver-v0",
        "provenance": "silver_heuristic",
        "num_val_figures": len(val_figs),
        "num_test_figures": len(test_figs),
        "total_test_gold_links": sum(len(f.get("silver_links") or f.get("gold_links", [])) for f in test_figs),
        "settings": {},
        **get_runtime_metadata("pipelines/evaluate_real_grounding.py")
    }


    for setting in ["oracle", "end_to_end"]:
        print(f"\n--- Setting: {setting.upper()} ---")
        train_i = [build_instance(f, setting, processor, ner) for f in train_figs]
        val_i = [build_instance(f, setting, processor, ner) for f in tuning_figs]
        test_i = [build_instance(f, setting, processor, ner) for f in test_figs]

        # 1. Rules Baseline
        rules_res, rules_lat = evaluate_rules(test_i)
        print(f"Rules:        F1 = {rules_res['f1']:.4f} (P={rules_res['precision']:.4f}, R={rules_res['recall']:.4f}, Acc@1={rules_res['grounding_accuracy@1']:.4f}, MRR={rules_res['mrr']:.4f}, Lat={rules_lat:.1f}ms)")

        # 2. GNN (with message passing) across 3 seeds
        gnn_runs = []
        for seed in [42, 43, 44]:
            m, _, lat = evaluate_learned(train_i, val_i, test_i, use_mp=True, seed=seed, epochs=80)
            m["latency_ms"] = lat
            gnn_runs.append(m)

        gnn_f1s = [r["f1"] for r in gnn_runs]
        gnn_mrr = [r["mrr"] for r in gnn_runs]
        gnn_lat = [r["latency_ms"] for r in gnn_runs]
        print(f"GNN:          F1 = {np.mean(gnn_f1s):.4f} +/- {np.std(gnn_f1s):.4f} (MRR={np.mean(gnn_mrr):.4f}, Lat={np.median(gnn_lat):.1f}ms)")

        # 3. Diagnostic No-Message-Passing across 3 seeds
        nomp_runs = []
        for seed in [42, 43, 44]:
            m, _, lat = evaluate_learned(train_i, val_i, test_i, use_mp=False, seed=seed, epochs=80)
            m["latency_ms"] = lat
            nomp_runs.append(m)

        nomp_f1s = [r["f1"] for r in nomp_runs]
        nomp_mrr = [r["mrr"] for r in nomp_runs]
        nomp_lat = [r["latency_ms"] for r in nomp_runs]
        print(f"No-MP GNN:    F1 = {np.mean(nomp_f1s):.4f} +/- {np.std(nomp_f1s):.4f} (MRR={np.mean(nomp_mrr):.4f}, Lat={np.median(nomp_lat):.1f}ms)")

        results["settings"][setting] = {
            "rules": {"metrics": rules_res, "latency_ms": rules_lat},
            "gnn": {
                "f1_mean": round(float(np.mean(gnn_f1s)), 4),
                "f1_std": round(float(np.std(gnn_f1s)), 4),
                "mrr_mean": round(float(np.mean(gnn_mrr)), 4),
                "latency_ms": float(np.median(gnn_lat)),
                "runs": gnn_runs
            },
            "no_message_passing": {
                "f1_mean": round(float(np.mean(nomp_f1s)), 4),
                "f1_std": round(float(np.std(nomp_f1s)), 4),
                "mrr_mean": round(float(np.mean(nomp_mrr)), 4),
                "latency_ms": float(np.median(nomp_lat)),
                "runs": nomp_runs
            }
        }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved real-world grounding results to: {OUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    run_all_evaluations()
