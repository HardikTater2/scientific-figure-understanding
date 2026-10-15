"""
Task 3a (Updated): Real Human Data Evaluation on CHART-Info UB PMC Test Split.
Evaluates FigureProcessor's text-role classification on real PubMed Central charts
using:
  (i) Gold human-annotated text bounding boxes (classifier in isolation, all 3,280 charts)
  (ii) Real OCR bounding boxes (end-to-end pipeline with OCR detection, 250 chart sample)

Scoring Rules & Fixes Applied (Step 1c):
  1. Adjusted OCR credit: credit "correctly dropped" ONLY for gold numeric ticks
     matched to a raw OCR box (IoU >= 0.30) that the processor then dropped.
     Unmatched gold boxes where OCR failed to detect anything remain misses ("OTHER").
  2. OCR Detection Recall: reported separately per role (AXIS, LEGEND, LABEL, TICK, OTHER).
  3. Macro-F1 over non-TICK roles (AXIS, LEGEND, LABEL, OTHER) reported for both gold and OCR.
  4. Bootstrap 95% confidence intervals resampled by chart, not by box.

Role Mapping Table:
  - "axis_title", "x_axis_title", "y_axis_title"  -> "AXIS"
  - "legend_label", "legend_title"                 -> "LEGEND"
  - "tick_label", "tick_grouping"                  -> "TICK"
  - "mark_label", "value_label"                    -> "LABEL"
  - "chart_title", "other"                         -> "OTHER"

Provenance: human_external_dataset
Outputs saved to results/phase3/chart_info_evaluation.json.
"""

import os
import sys
import json
import glob
import time
from typing import Dict, Any, List, Tuple
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.vision.figure_processor import FigureProcessor, is_numeric_text
from src.vision.ocr_engine import OCREngine
from src.common_metadata import get_runtime_metadata

TEST_ANN_DIR = os.path.join(ROOT, "data", "external", "chart_info", "CHARTINFO_2024_Test", "annotations_JSON")
TEST_IMG_DIR = os.path.join(ROOT, "data", "external", "chart_info", "CHARTINFO_2024_Test", "images")
CACHE_FILE = os.path.join(ROOT, "data", "external", "chart_info", "ocr_cache.json")
OUT_PATH = os.path.join(ROOT, "results", "phase3", "chart_info_evaluation.json")

TARGET_CLASSES = ["AXIS", "LEGEND", "LABEL", "TICK", "OTHER"]
NON_TICK_CLASSES = ["AXIS", "LEGEND", "LABEL", "OTHER"]


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = areaA + areaB - inter
    return inter / denom if denom > 0 else 0.0


def polygon_to_bbox(poly: Dict[str, int]) -> List[int]:
    xs = [poly[k] for k in ["x0", "x1", "x2", "x3"] if k in poly]
    ys = [poly[k] for k in ["y0", "y1", "y2", "y3"] if k in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def classify_chart_boxes(processor: FigureProcessor, boxes: List[Dict[str, Any]], W: int, H: int) -> List[str]:
    out = ["OTHER"] * len(boxes)
    non_numeric_indices = []
    non_numeric_boxes = []

    for i, b in enumerate(boxes):
        if is_numeric_text(b["text"]):
            out[i] = "TICK"
        else:
            non_numeric_indices.append(i)
            non_numeric_boxes.append(b)

    if non_numeric_boxes:
        classified = processor._classify(non_numeric_boxes, W, H)
        for orig_idx, role, b in zip(non_numeric_indices, classified, non_numeric_boxes):
            if b["bbox"][3] < processor.top_band * H and role == "LABEL":
                out[orig_idx] = "OTHER"
            else:
                out[orig_idx] = role

    return out


def calc_metrics(preds: List[str], trues: List[str]) -> Dict[str, Any]:
    if not trues:
        return {"accuracy": 0.0, "macro_f1": 0.0, "macro_f1_non_tick": 0.0, "total_boxes": 0, "per_class": {}}
    acc = accuracy_score(trues, preds)
    macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)
    macro_f1_non_tick = f1_score(trues, preds, labels=NON_TICK_CLASSES, average="macro", zero_division=0)
    rep = classification_report(trues, preds, labels=TARGET_CLASSES, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_f1_non_tick": round(float(macro_f1_non_tick), 4),
        "total_boxes": len(trues),
        "per_class": {
            c: {
                "precision": round(float(rep[c]["precision"]), 4),
                "recall": round(float(rep[c]["recall"]), 4),
                "f1": round(float(rep[c]["f1-score"]), 4),
                "support": int(rep[c]["support"])
            }
            for c in TARGET_CLASSES if c in rep
        }
    }


def bootstrap_gold_by_chart(chart_records: List[Dict[str, Any]], n_boot: int = 1000) -> Dict[str, Any]:
    """Bootstrap resampling by chart for gold boxes evaluation."""
    n_charts = len(chart_records)
    rng = np.random.default_rng(42)
    boot_accs, boot_f1s, boot_non_tick = [], [], []

    for _ in range(n_boot):
        sample_idxs = rng.choice(n_charts, size=n_charts, replace=True)
        boot_preds, boot_trues = [], []
        for idx in sample_idxs:
            rec = chart_records[idx]
            boot_preds.extend(rec["preds"])
            boot_trues.extend(rec["trues"])
        if boot_trues:
            boot_accs.append(accuracy_score(boot_trues, boot_preds))
            boot_f1s.append(f1_score(boot_trues, boot_preds, average="macro", zero_division=0))
            boot_non_tick.append(f1_score(boot_trues, boot_preds, labels=NON_TICK_CLASSES, average="macro", zero_division=0))

    return {
        "accuracy_ci95": [round(float(np.percentile(boot_accs, 2.5)), 4), round(float(np.percentile(boot_accs, 97.5)), 4)],
        "macro_f1_ci95": [round(float(np.percentile(boot_f1s, 2.5)), 4), round(float(np.percentile(boot_f1s, 97.5)), 4)],
        "macro_f1_non_tick_ci95": [round(float(np.percentile(boot_non_tick, 2.5)), 4), round(float(np.percentile(boot_non_tick, 97.5)), 4)]
    }


def bootstrap_ocr_by_chart(chart_records: List[Dict[str, Any]], n_boot: int = 1000) -> Dict[str, Any]:
    """Bootstrap resampling by chart for OCR boxes evaluation."""
    n_charts = len(chart_records)
    rng = np.random.default_rng(42)

    boot_std_f1, boot_std_nontick, boot_std_acc = [], [], []
    boot_adj_f1, boot_adj_nontick, boot_adj_acc = [], [], []
    boot_det_recalls = {c: [] for c in TARGET_CLASSES}

    for _ in range(n_boot):
        sample_idxs = rng.choice(n_charts, size=n_charts, replace=True)
        trues = []
        preds_std = []
        preds_adj = []
        gold_counts = {c: 0 for c in TARGET_CLASSES}
        det_counts = {c: 0 for c in TARGET_CLASSES}

        for idx in sample_idxs:
            rec = chart_records[idx]
            trues.extend(rec["trues"])
            preds_std.extend(rec["preds_std"])
            preds_adj.extend(rec["preds_adj"])
            for c in TARGET_CLASSES:
                gold_counts[c] += rec["gold_role_counts"].get(c, 0)
                det_counts[c] += rec["det_role_counts"].get(c, 0)

        if trues:
            boot_std_acc.append(accuracy_score(trues, preds_std))
            boot_std_f1.append(f1_score(trues, preds_std, average="macro", zero_division=0))
            boot_std_nontick.append(f1_score(trues, preds_std, labels=NON_TICK_CLASSES, average="macro", zero_division=0))

            boot_adj_acc.append(accuracy_score(trues, preds_adj))
            boot_adj_f1.append(f1_score(trues, preds_adj, average="macro", zero_division=0))
            boot_adj_nontick.append(f1_score(trues, preds_adj, labels=NON_TICK_CLASSES, average="macro", zero_division=0))

        for c in TARGET_CLASSES:
            rec_val = det_counts[c] / gold_counts[c] if gold_counts[c] > 0 else 0.0
            boot_det_recalls[c].append(rec_val)

    return {
        "standard_macro_f1_ci95": [round(float(np.percentile(boot_std_f1, 2.5)), 4), round(float(np.percentile(boot_std_f1, 97.5)), 4)],
        "standard_macro_f1_non_tick_ci95": [round(float(np.percentile(boot_std_nontick, 2.5)), 4), round(float(np.percentile(boot_std_nontick, 97.5)), 4)],
        "standard_accuracy_ci95": [round(float(np.percentile(boot_std_acc, 2.5)), 4), round(float(np.percentile(boot_std_acc, 97.5)), 4)],
        "adjusted_macro_f1_ci95": [round(float(np.percentile(boot_adj_f1, 2.5)), 4), round(float(np.percentile(boot_adj_f1, 97.5)), 4)],
        "adjusted_macro_f1_non_tick_ci95": [round(float(np.percentile(boot_adj_nontick, 2.5)), 4), round(float(np.percentile(boot_adj_nontick, 97.5)), 4)],
        "adjusted_accuracy_ci95": [round(float(np.percentile(boot_adj_acc, 2.5)), 4), round(float(np.percentile(boot_adj_acc, 97.5)), 4)],
        "ocr_detection_recall_ci95": {
            c: [round(float(np.percentile(boot_det_recalls[c], 2.5)), 4), round(float(np.percentile(boot_det_recalls[c], 97.5)), 4)]
            for c in TARGET_CLASSES
        }
    }


def main():
    print("=" * 70)
    print("TASK 3a (UPDATED): CHART-INFO REAL DATA EVALUATION (STRICT ADJUSTED RULES)")
    print("=" * 70)

    mapping_doc = {
        "axis_title": "AXIS",
        "x_axis_title": "AXIS",
        "y_axis_title": "AXIS",
        "legend_label": "LEGEND",
        "legend_title": "LEGEND",
        "tick_label": "TICK",
        "tick_grouping": "TICK",
        "mark_label": "LABEL",
        "value_label": "LABEL",
        "chart_title": "OTHER",
        "other": "OTHER"
    }

    ocr_engine = OCREngine(cache_path=CACHE_FILE)
    processor = FigureProcessor(ocr_engine=ocr_engine)
    ann_files = sorted(glob.glob(os.path.join(TEST_ANN_DIR, "*.json")))

    selected_charts = []
    for fp in ann_files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        img_path = os.path.join(TEST_IMG_DIR, stem + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(TEST_IMG_DIR, stem + ".png")
        if not os.path.exists(img_path):
            continue

        with open(fp) as f:
            d = json.load(f)

        t2_out = (d.get("task2") or {}).get("output")
        t3_out = (d.get("task3") or {}).get("output")

        if isinstance(t2_out, dict) and isinstance(t3_out, dict):
            blocks = t2_out.get("text_blocks", [])
            roles = t3_out.get("text_roles", [])
            if blocks and roles and len(blocks) == len(roles):
                selected_charts.append((stem, img_path, fp, blocks, roles))

    total_valid = len(selected_charts)
    print(f"Found {total_valid} valid test charts with Task 2+3 annotations in full test split.")

    # 1. EVALUATION ON GOLD BOXES ACROSS FULL TEST SPLIT (ALL 3,280 CHARTS)
    print(f"\n[1/2] Evaluating gold text boxes across FULL test split ({total_valid} charts)...")
    gold_chart_records = []
    all_gold_preds = []
    all_gold_trues = []
    t_gold0 = time.perf_counter()

    for idx, (stem, img_path, ann_path, blocks, roles) in enumerate(selected_charts):
        role_by_id = {r["id"]: r["role"] for r in roles if isinstance(r, dict)}
        bboxes = [polygon_to_bbox(b["polygon"]) for b in blocks if "polygon" in b and b["id"] in role_by_id]
        if not bboxes:
            continue
        W = max(b[2] for b in bboxes) + 20
        H = max(b[3] for b in bboxes) + 20

        chart_boxes = []
        chart_trues = []
        for b in blocks:
            b_id = b["id"]
            if b_id not in role_by_id or "polygon" not in b:
                continue
            bbox = polygon_to_bbox(b["polygon"])
            bw = max(bbox[2] - bbox[0], 1)
            bh = max(bbox[3] - bbox[1], 1)
            chart_boxes.append({
                "bbox": bbox,
                "text": b.get("text", ""),
                "orientation": "vertical" if bh > 1.5 * bw else "horizontal"
            })
            chart_trues.append(mapping_doc.get(role_by_id[b_id], "OTHER"))

        pred_roles = classify_chart_boxes(processor, chart_boxes, W, H)
        all_gold_preds.extend(pred_roles)
        all_gold_trues.extend(chart_trues)
        gold_chart_records.append({"preds": pred_roles, "trues": chart_trues})

    gold_time = round(time.perf_counter() - t_gold0, 2)
    print(f"  Classified {len(all_gold_trues)} gold boxes across all {total_valid} charts in {gold_time}s.")
    gold_boot_ci = bootstrap_gold_by_chart(gold_chart_records, n_boot=1000)

    # 2. EVALUATION ON OCR BOXES (SAMPLE OF 250 CHARTS)
    ocr_sample_size = min(250, total_valid)
    print(f"\n[2/2] Evaluating OCR detected boxes on test split sample ({ocr_sample_size} charts)...")
    ocr_eval_charts = selected_charts[:ocr_sample_size]

    ocr_chart_records = []
    all_ocr_preds_std = []
    all_ocr_preds_adj = []
    all_ocr_trues = []

    overall_gold_counts = {c: 0 for c in TARGET_CLASSES}
    overall_det_counts = {c: 0 for c in TARGET_CLASSES}

    t_ocr0 = time.perf_counter()

    for idx, (stem, img_path, ann_path, blocks, roles) in enumerate(ocr_eval_charts):
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{ocr_sample_size} OCR charts...")

        role_by_id = {r["id"]: r["role"] for r in roles if isinstance(r, dict)}
        gold_boxes = []
        chart_gold_trues = []
        for b in blocks:
            b_id = b["id"]
            if b_id not in role_by_id or "polygon" not in b:
                continue
            bbox = polygon_to_bbox(b["polygon"])
            gold_boxes.append({
                "bbox": bbox,
                "text": b.get("text", "")
            })
            chart_gold_trues.append(mapping_doc.get(role_by_id[b_id], "OTHER"))

        # Process via real OCR
        raw_dets = processor.ocr.read(img_path)
        dropped_dets = [d for d in raw_dets if is_numeric_text(d["text"])]
        ocr_out = processor.process_figure(img_path)
        extracted = [e for e in ocr_out["elements"] if e.get("type") in TARGET_CLASSES and "bbox" in e]

        # Calculate OCR Detection Recall for this chart
        chart_gold_counts = {c: 0 for c in TARGET_CLASSES}
        chart_det_counts = {c: 0 for c in TARGET_CLASSES}

        for g_b, g_role in zip(gold_boxes, chart_gold_trues):
            chart_gold_counts[g_role] += 1
            overall_gold_counts[g_role] += 1
            # Check if any raw detection matched this gold box
            is_detected = any(compute_iou(g_b["bbox"], d["bbox"]) >= 0.30 for d in raw_dets)
            if is_detected:
                chart_det_counts[g_role] += 1
                overall_det_counts[g_role] += 1

        chart_preds_std = []
        chart_preds_adj = []
        chart_trues = []

        # Match extracted OCR elements to Gold
        matched_gold_ids = set()
        for elem in extracted:
            e_box = elem["bbox"]
            e_type = elem["type"]
            best_iou = 0.0
            best_gid = -1

            for g_idx, g_b in enumerate(gold_boxes):
                iou = compute_iou(e_box, g_b["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gid = g_idx

            if best_iou >= 0.30 and best_gid != -1:
                matched_gold_ids.add(best_gid)
                chart_preds_std.append(e_type)
                chart_preds_adj.append(e_type)
                chart_trues.append(chart_gold_trues[best_gid])
            else:
                chart_preds_std.append(e_type)
                chart_preds_adj.append(e_type)
                chart_trues.append("OTHER")

        # Handle unmatched gold boxes
        for g_idx, g_role in enumerate(chart_gold_trues):
            if g_idx not in matched_gold_ids:
                chart_trues.append(g_role)
                chart_preds_std.append("OTHER")

                # Step 1c: credit "correctly dropped" ONLY for gold numeric ticks
                # matched to an OCR box that the processor then dropped.
                g_text = gold_boxes[g_idx]["text"]
                is_num_tick = (g_role == "TICK" and is_numeric_text(g_text))
                matched_dropped = any(compute_iou(gold_boxes[g_idx]["bbox"], d["bbox"]) >= 0.30 for d in dropped_dets)

                if is_num_tick and matched_dropped:
                    chart_preds_adj.append("TICK")
                else:
                    chart_preds_adj.append("OTHER")

        all_ocr_preds_std.extend(chart_preds_std)
        all_ocr_preds_adj.extend(chart_preds_adj)
        all_ocr_trues.extend(chart_trues)

        ocr_chart_records.append({
            "trues": chart_trues,
            "preds_std": chart_preds_std,
            "preds_adj": chart_preds_adj,
            "gold_role_counts": chart_gold_counts,
            "det_role_counts": chart_det_counts
        })

    processor.ocr.save_cache()
    ocr_time = round(time.perf_counter() - t_ocr0, 2)
    print(f"  Processed {ocr_sample_size} OCR charts in {ocr_time}s.")

    ocr_boot_ci = bootstrap_ocr_by_chart(ocr_chart_records, n_boot=1000)

    # Compute aggregate metrics
    res_gold = calc_metrics(all_gold_preds, all_gold_trues)
    res_ocr_std = calc_metrics(all_ocr_preds_std, all_ocr_trues)
    res_ocr_adj = calc_metrics(all_ocr_preds_adj, all_ocr_trues)

    # OCR detection recall summary
    ocr_detection_recall = {
        c: {
            "gold_support": overall_gold_counts[c],
            "detected_by_ocr": overall_det_counts[c],
            "recall": round(float(overall_det_counts[c] / overall_gold_counts[c]), 4) if overall_gold_counts[c] > 0 else 0.0,
            "ci95": ocr_boot_ci["ocr_detection_recall_ci95"][c]
        }
        for c in TARGET_CLASSES
    }

    print("\n--- [1] FULL TEST SPLIT: GOLD TEXT BOXES (3,280 CHARTS) ---")
    print(f"Accuracy: {res_gold['accuracy']:.4f} {gold_boot_ci['accuracy_ci95']}")
    print(f"Macro-F1 (All 5 Roles): {res_gold['macro_f1']:.4f} {gold_boot_ci['macro_f1_ci95']}")
    print(f"Macro-F1 (Non-TICK: AXIS, LEGEND, LABEL, OTHER): {res_gold['macro_f1_non_tick']:.4f} {gold_boot_ci['macro_f1_non_tick_ci95']}")
    for c, m in res_gold["per_class"].items():
        print(f"  {c:<8}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f} (N={m['support']})")

    print("\n--- [2A] OCR DETECTED BOXES: STANDARD (EVERY UNMATCHED GOLD BOX = MISS) ---")
    print(f"Accuracy: {res_ocr_std['accuracy']:.4f} {ocr_boot_ci['standard_accuracy_ci95']}")
    print(f"Macro-F1 (All 5 Roles): {res_ocr_std['macro_f1']:.4f} {ocr_boot_ci['standard_macro_f1_ci95']}")
    print(f"Macro-F1 (Non-TICK): {res_ocr_std['macro_f1_non_tick']:.4f} {ocr_boot_ci['standard_macro_f1_non_tick_ci95']}")
    for c, m in res_ocr_std["per_class"].items():
        print(f"  {c:<8}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f} (N={m['support']})")

    print("\n--- [2B] OCR DETECTED BOXES: STRICT ADJUSTED (ONLY MATCHED DROPPED TICKS CREDITED) ---")
    print(f"Accuracy: {res_ocr_adj['accuracy']:.4f} {ocr_boot_ci['adjusted_accuracy_ci95']}")
    print(f"Macro-F1 (All 5 Roles): {res_ocr_adj['macro_f1']:.4f} {ocr_boot_ci['adjusted_macro_f1_ci95']}")
    print(f"Macro-F1 (Non-TICK): {res_ocr_adj['macro_f1_non_tick']:.4f} {ocr_boot_ci['adjusted_macro_f1_non_tick_ci95']}")
    for c, m in res_ocr_adj["per_class"].items():
        print(f"  {c:<8}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f} (N={m['support']})")

    print("\n--- [3] OCR DETECTION RECALL PER ROLE (SEPARATE) ---")
    for c, m in ocr_detection_recall.items():
        print(f"  {c:<8}: Recall = {m['recall']*100:.2f}% ({m['detected_by_ocr']}/{m['gold_support']}) CI95: {m['ci95']}")

    results = {
        "provenance": "human_external_dataset",
        "benchmark": "CHART-Info UB PMC Test Split (ICPR 2020 / ICDAR 2023 / ICPR 2024)",
        "source_url": "https://tc11.cvc.uab.es/datasets/ICPR2020-CHART-Info_1",
        "taxonomy_classes": TARGET_CLASSES,
        "non_tick_classes": NON_TICK_CLASSES,
        "mapping_table": mapping_doc,
        "full_test_split_charts": total_valid,
        "gold_boxes_evaluation": {
            "num_charts": total_valid,
            "runtime_s": gold_time,
            **res_gold,
            "bootstrap_ci95_by_chart": gold_boot_ci
        },
        "ocr_boxes_evaluation": {
            "num_charts": ocr_sample_size,
            "runtime_s": ocr_time,
            "standard_all_unmatched_misses": {
                **res_ocr_std,
                "bootstrap_ci95_by_chart": {
                    "macro_f1": ocr_boot_ci["standard_macro_f1_ci95"],
                    "macro_f1_non_tick": ocr_boot_ci["standard_macro_f1_non_tick_ci95"],
                    "accuracy": ocr_boot_ci["standard_accuracy_ci95"]
                }
            },
            "adjusted_numeric_ticks_matched_and_dropped": {
                **res_ocr_adj,
                "bootstrap_ci95_by_chart": {
                    "macro_f1": ocr_boot_ci["adjusted_macro_f1_ci95"],
                    "macro_f1_non_tick": ocr_boot_ci["adjusted_macro_f1_non_tick_ci95"],
                    "accuracy": ocr_boot_ci["adjusted_accuracy_ci95"]
                }
            },
            "ocr_detection_recall_per_role": ocr_detection_recall
        },
        **get_runtime_metadata("pipelines/evaluate_chart_info.py")
    }

    # Save raw predictions for independent verification
    raw_pred_path = os.path.join(ROOT, "results", "phase3", "raw", "chart_info_raw_predictions.json")
    os.makedirs(os.path.dirname(raw_pred_path), exist_ok=True)
    raw_payload = {
        "provenance": "human_external_dataset",
        "model_name": "FigureProcessor (heuristic text-role classifier) + EasyOCR (craft+crnn)",
        "model_id": "FigureProcessor_v1_EasyOCR",
        **get_runtime_metadata("pipelines/evaluate_chart_info.py"),
        "gold_trues": all_gold_trues,
        "gold_preds": all_gold_preds,
        "ocr_trues": all_ocr_trues,
        "ocr_preds_std": all_ocr_preds_std,
        "ocr_preds_adj": all_ocr_preds_adj,
        "overall_gold_counts": overall_gold_counts,
        "overall_det_counts": overall_det_counts
    }
    with open(raw_pred_path, "w") as f:
        json.dump(raw_payload, f)
    print(f"Saved raw predictions to: {raw_pred_path}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved CHART-Info evaluation results to: {OUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
