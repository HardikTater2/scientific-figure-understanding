"""
Task 3b & Task 4 (Updated / Step 1b): CharXiv NeurIPS 2024 Descriptive QA Evaluation.
Evaluates deterministic FigureProcessor rules vs. local 4-bit 7B VLM baseline on CharXiv validation split.

Evaluation Fixes & Scoring Rules Applied (Step 1b):
  1. Numeric QIDs (4, 5, 6, 7, 12): STRICTLY NO substring containment.
     - Parses the first number in predicted and gold answers.
     - Compares numerically with 5% relative tolerance (or |diff| <= 1e-4).
     - QID 12 (count): counts must match exactly (integer equality).
  2. Text QIDs (2, 3, 13):
     - QIDs 2, 3 (Axis titles): exact = normalized equality; fuzzy = SequenceMatcher ratio >= 0.80.
     - QID 13 (Legend names): compare comma-separated item sets, not words!
       Exact = set equality; Fuzzy = Jaccard index of item sets >= 0.80.
  3. Sampling: Sample 40 questions per qid with a fixed seed (random.Random(0)).
     Seed = 0, deterministic shuffle per QID pool. Evaluates 320 questions total.
  4. Reporting: Rules reported separately on text QIDs (2, 3, 13) since ticks are dropped by design.
  5. Bootstrap 95% CIs: Resampled by chart (chart_id), not by question/box.
  6. Non-Negotiable Rule 2: Logs raw prompts, responses, runtimes, timestamps to results/phase2/raw/vlm_charxiv_raw.jsonl.

Provenance: human_external_dataset
Outputs saved to results/phase3/charxiv_evaluation.json.
"""

import os
import sys
import json
import time
import re
import tempfile
import random
from typing import Dict, Any, List, Tuple, Optional, Set
from difflib import SequenceMatcher
import pandas as pd
from PIL import Image
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.vision.figure_processor import FigureProcessor
from src.common_metadata import get_runtime_metadata
from huggingface_hub import hf_hub_download

TARGET_QIDS = {
    2: "What is the label of the x-axis?",
    3: "What is the label of the y-axis?",
    4: "What is the leftmost labeled tick on the x-axis?",
    5: "What is the rightmost labeled tick on the x-axis?",
    6: "What is the spatially lowest labeled tick on the y-axis?",
    7: "What is the spatially highest labeled tick on the y-axis?",
    12: "How many discrete labels are there in the legend?",
    13: "What are the names of the labels in the legend? (from top to bottom, then left to right)"
}

TEXT_QIDS = [2, 3, 13]
NUMERIC_QIDS = [4, 5, 6, 7, 12]

OUT_PATH = os.path.join(ROOT, "results", "phase3", "charxiv_evaluation.json")
RAW_VLM_LOG = os.path.join(ROOT, "results", "phase2", "raw", "vlm_charxiv_raw.jsonl")


def normalize_text_answer(s: str) -> str:
    """Normalize text: lowercase, strip units in brackets/parentheses e.g. (dB), [nm], strip punctuation."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    # Strip units in parentheses or brackets e.g. (dB), [nm], (s), [%]
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
    # Strip punctuation except alphanumeric
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_first_float(s: str) -> Optional[float]:
    """Extracts first valid integer or floating point number from text string."""
    if s is None:
        return None
    cleaned = str(s).replace(",", "")
    m = re.search(r"[-+]?(?:\d*\.\d+|\d+)", cleaned)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def evaluate_charxiv_answer(pred: str, gold: str, qid: int) -> Tuple[bool, bool]:
    """
    Step 1b exact & fuzzy scoring function:
      - Numeric QIDs (4, 5, 6, 7, 12): NO substring containment.
        Compare parsed floats with 5% relative tolerance. QID 12 counts must match exactly.
      - Text QIDs (2, 3, 13):
        QIDs 2, 3: exact = normalized equality; fuzzy = SequenceMatcher ratio >= 0.80.
        QID 13: comma-separated item sets! exact = set equality; fuzzy = Jaccard >= 0.80.
    """
    if pred is None or gold is None:
        return False, False

    # Numeric QIDs
    if qid in NUMERIC_QIDS:
        p_val = parse_first_float(pred)
        g_val = parse_first_float(gold)
        if p_val is None or g_val is None:
            return False, False

        if qid == 12:
            exact = (int(round(p_val)) == int(round(g_val)))
            fuzzy = exact
            return exact, fuzzy
        else:
            diff = abs(p_val - g_val)
            if abs(g_val) > 1e-5:
                exact = (diff / abs(g_val) <= 0.05) or (diff <= 1e-4)
            else:
                exact = (diff <= 1e-4)
            fuzzy = exact
            return exact, fuzzy

    # Text QIDs (2, 3, 13)
    if qid == 13:
        # Compare comma-separated item sets, not words!
        p_items = {normalize_text_answer(x) for x in str(pred).split(",") if normalize_text_answer(x)}
        g_items = {normalize_text_answer(x) for x in str(gold).split(",") if normalize_text_answer(x)}
        if not g_items:
            return False, False
        exact = (p_items == g_items)
        union = len(p_items | g_items)
        inter = len(p_items & g_items)
        jaccard = (inter / union) if union > 0 else 0.0
        fuzzy = exact or (jaccard >= 0.80)
        return exact, fuzzy

    elif qid in (2, 3):
        p_norm = normalize_text_answer(pred)
        g_norm = normalize_text_answer(gold)
        if not p_norm or not g_norm:
            return False, False
        exact = (p_norm == g_norm)
        ratio = SequenceMatcher(None, p_norm, g_norm).ratio()
        fuzzy = exact or (ratio >= 0.80)
        return exact, fuzzy

    p_norm = normalize_text_answer(pred)
    g_norm = normalize_text_answer(gold)
    exact = (p_norm == g_norm and len(p_norm) > 0)
    fuzzy = exact or (SequenceMatcher(None, p_norm, g_norm).ratio() >= 0.80)
    return exact, fuzzy


def answer_with_deterministic_rules(elements: List[Dict[str, Any]], qid: int, W: int, H: int) -> str:
    """Answers descriptive question from extracted layout elements."""
    axes = [e for e in elements if e.get("type") == "AXIS" and "bbox" in e]
    legends = [e for e in elements if e.get("type") == "LEGEND" and "bbox" in e]
    labels = [e for e in elements if e.get("type") == "LABEL" and "bbox" in e]

    if qid == 2:
        x_axes = [a for a in axes if a["bbox"][1] > 0.6 * H]
        if x_axes:
            return max(x_axes, key=lambda a: a["bbox"][1]).get("text", "")
        return axes[0].get("text", "") if axes else ""

    elif qid == 3:
        y_axes = [a for a in axes if a["bbox"][0] < 0.4 * W]
        if y_axes:
            return min(y_axes, key=lambda a: a["bbox"][0]).get("text", "")
        return axes[0].get("text", "") if axes else ""

    elif qid == 12:
        return str(len(legends))

    elif qid == 13:
        sorted_leg = sorted(legends, key=lambda l: (l["bbox"][1], l["bbox"][0]))
        return ", ".join(l.get("text", "") for l in sorted_leg)

    elif qid == 4:
        bottom_labels = [l for l in labels if l["bbox"][1] > 0.7 * H]
        if bottom_labels:
            return min(bottom_labels, key=lambda l: l["bbox"][0]).get("text", "")
        return ""

    elif qid == 5:
        bottom_labels = [l for l in labels if l["bbox"][1] > 0.7 * H]
        if bottom_labels:
            return max(bottom_labels, key=lambda l: l["bbox"][2]).get("text", "")
        return ""

    elif qid == 6:
        left_labels = [l for l in labels if l["bbox"][0] < 0.35 * W]
        if left_labels:
            return max(left_labels, key=lambda l: l["bbox"][3]).get("text", "")
        return ""

    elif qid == 7:
        left_labels = [l for l in labels if l["bbox"][0] < 0.35 * W]
        if left_labels:
            return min(left_labels, key=lambda l: l["bbox"][1]).get("text", "")
        return ""

    return ""


def bootstrap_charxiv_by_chart(records: List[Dict[str, Any]], n_boot: int = 1000) -> Dict[str, Any]:
    """
    Computes 95% bootstrap confidence intervals resampling by CHART (not by question).
    """
    chart_ids = sorted(list({r["chart_id"] for r in records}))
    n_charts = len(chart_ids)
    by_chart = {c: [r for r in records if r["chart_id"] == c] for c in chart_ids}

    rng = np.random.default_rng(42)

    boot_rule_ex_all, boot_rule_fz_all = [], []
    boot_vlm_ex_all, boot_vlm_fz_all = [], []

    boot_rule_ex_text, boot_rule_fz_text = [], []
    boot_vlm_ex_text, boot_vlm_fz_text = [], []

    boot_rule_ex_num, boot_rule_fz_num = [], []
    boot_vlm_ex_num, boot_vlm_fz_num = [], []

    boot_per_qid = {q: {"r_ex": [], "r_fz": [], "v_ex": [], "v_fz": []} for q in TARGET_QIDS}

    for _ in range(n_boot):
        sample_charts = rng.choice(chart_ids, size=n_charts, replace=True)
        sampled_records = []
        for c in sample_charts:
            sampled_records.extend(by_chart[c])

        # Overall
        boot_rule_ex_all.append(np.mean([r["rule_exact"] for r in sampled_records]))
        boot_rule_fz_all.append(np.mean([r["rule_fuzzy"] for r in sampled_records]))
        boot_vlm_ex_all.append(np.mean([r["vlm_exact"] for r in sampled_records]))
        boot_vlm_fz_all.append(np.mean([r["vlm_fuzzy"] for r in sampled_records]))

        # Text QIDs
        text_recs = [r for r in sampled_records if r["qid"] in TEXT_QIDS]
        if text_recs:
            boot_rule_ex_text.append(np.mean([r["rule_exact"] for r in text_recs]))
            boot_rule_fz_text.append(np.mean([r["rule_fuzzy"] for r in text_recs]))
            boot_vlm_ex_text.append(np.mean([r["vlm_exact"] for r in text_recs]))
            boot_vlm_fz_text.append(np.mean([r["vlm_fuzzy"] for r in text_recs]))

        # Numeric QIDs
        num_recs = [r for r in sampled_records if r["qid"] in NUMERIC_QIDS]
        if num_recs:
            boot_rule_ex_num.append(np.mean([r["rule_exact"] for r in num_recs]))
            boot_rule_fz_num.append(np.mean([r["rule_fuzzy"] for r in num_recs]))
            boot_vlm_ex_num.append(np.mean([r["vlm_exact"] for r in num_recs]))
            boot_vlm_fz_num.append(np.mean([r["vlm_fuzzy"] for r in num_recs]))

        # Per QID
        for q in TARGET_QIDS:
            q_recs = [r for r in sampled_records if r["qid"] == q]
            if q_recs:
                boot_per_qid[q]["r_ex"].append(np.mean([r["rule_exact"] for r in q_recs]))
                boot_per_qid[q]["r_fz"].append(np.mean([r["rule_fuzzy"] for r in q_recs]))
                boot_per_qid[q]["v_ex"].append(np.mean([r["vlm_exact"] for r in q_recs]))
                boot_per_qid[q]["v_fz"].append(np.mean([r["vlm_fuzzy"] for r in q_recs]))

    def ci(arr):
        if not arr:
            return [0.0, 0.0]
        return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]

    return {
        "overall": {
            "rules": {"exact_ci95": ci(boot_rule_ex_all), "fuzzy_ci95": ci(boot_rule_fz_all)},
            "vlm": {"exact_ci95": ci(boot_vlm_ex_all), "fuzzy_ci95": ci(boot_vlm_fz_all)}
        },
        "text_qids": {
            "rules": {"exact_ci95": ci(boot_rule_ex_text), "fuzzy_ci95": ci(boot_rule_fz_text)},
            "vlm": {"exact_ci95": ci(boot_vlm_ex_text), "fuzzy_ci95": ci(boot_vlm_fz_text)}
        },
        "numeric_qids": {
            "rules": {"exact_ci95": ci(boot_rule_ex_num), "fuzzy_ci95": ci(boot_rule_fz_num)},
            "vlm": {"exact_ci95": ci(boot_vlm_ex_num), "fuzzy_ci95": ci(boot_vlm_fz_num)}
        },
        "per_qid": {
            str(q): {
                "rules": {"exact_ci95": ci(boot_per_qid[q]["r_ex"]), "fuzzy_ci95": ci(boot_per_qid[q]["r_fz"])},
                "vlm": {"exact_ci95": ci(boot_per_qid[q]["v_ex"]), "fuzzy_ci95": ci(boot_per_qid[q]["v_fz"])}
            }
            for q in TARGET_QIDS
        }
    }


def main():
    print("=" * 70)
    print("TASK 3b & 4 (STEP 1b): CHARXIV EVALUATION (320 QUESTIONS, STRICT SCORING)")
    print("=" * 70)

    # 1. Load CharXiv validation split
    val_parquet_path = hf_hub_download("princeton-nlp/CharXiv", filename="val.parquet", repo_type="dataset")
    df = pd.read_parquet(val_parquet_path)
    print(f"Loaded CharXiv validation split ({len(df)} charts).")

    # 2. Select candidate target question instances
    eval_instances = []
    for idx, row in df.iterrows():
        chart_id = row.get("original_id", f"charxiv_{idx}")
        img_dict = row["image"]
        img_bytes = img_dict.get("bytes") if isinstance(img_dict, dict) else None
        if not img_bytes:
            continue

        for q_col in range(1, 5):
            qid = row.get(f"descriptive_q{q_col}")
            ans = row.get(f"descriptive_a{q_col}")
            if qid in TARGET_QIDS and ans and ans != "Not Applicable":
                eval_instances.append({
                    "chart_idx": idx,
                    "chart_id": chart_id,
                    "img_bytes": img_bytes,
                    "qid": int(qid),
                    "question_text": TARGET_QIDS[qid],
                    "gold_answer": str(ans).strip()
                })

    total_valid = len(eval_instances)
    print(f"Found {total_valid} candidate descriptive questions in validation split.")

    # 3. Step 1b Sampling: Sample exactly 40 questions per qid with fixed seed (random.Random(0))
    rng = random.Random(0)
    by_qid = {q: [] for q in TARGET_QIDS}
    for inst in eval_instances:
        by_qid[inst["qid"]].append(inst)

    eval_sample = []
    per_qid_cap = 40
    print(f"\nSampling method: random.Random(0).shuffle per QID pool, selecting first {per_qid_cap} questions.")
    for q in sorted(TARGET_QIDS):
        pool = list(by_qid[q])
        rng.shuffle(pool)
        selected = pool[:per_qid_cap]
        eval_sample.extend(selected)
        print(f"  QID {q:<2} ({TARGET_QIDS[q][:30]}...): sampled {len(selected)} questions (from pool of {len(pool)})")

    print(f"\nTotal sampled questions: {len(eval_sample)} across {len(TARGET_QIDS)} target QIDs.")

    # 4. Setup FigureProcessor for deterministic rule baseline
    processor = FigureProcessor()

    # 5. Check if raw VLM predictions already exist
    have_raw_log = os.path.exists(RAW_VLM_LOG) and os.path.getsize(RAW_VLM_LOG) > 0
    raw_map = {}
    if have_raw_log:
        print(f"\n[Raw VLM outputs detected at {RAW_VLM_LOG}]")
        print("Loading raw VLM responses for scoring and recomputing metrics...")
        with open(RAW_VLM_LOG, "r") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    raw_map[(d["chart_id"], d["qid"])] = d
        if len(raw_map) < len(eval_sample):
            print(f"Warning: RAW_VLM_LOG has {len(raw_map)} items, expected {len(eval_sample)}. Re-running missing items...")
            have_raw_log = False

    raw_vlm_file = None
    if not have_raw_log:
        import mlx_vlm
        vlm_repo = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
        vlm_name = "Qwen2-VL-7B-Instruct-4bit"
        print(f"\nLoading local VLM model '{vlm_repo}' on Apple Silicon M4 Pro...")
        t_vlm0 = time.perf_counter()
        vlm_model, vlm_processor = mlx_vlm.load(vlm_repo)
        print(f"Successfully loaded {vlm_name} in {time.perf_counter() - t_vlm0:.2f}s!")
        os.makedirs(os.path.dirname(RAW_VLM_LOG), exist_ok=True)
        raw_vlm_file = open(RAW_VLM_LOG, "w")

    eval_records = []
    chart_elements_cache = {}

    print(f"\nRunning evaluation on {len(eval_sample)} questions...")
    t_eval0 = time.perf_counter()

    for i, inst in enumerate(eval_sample):
        c_idx = inst["chart_idx"]
        chart_id = inst["chart_id"]
        qid = inst["qid"]
        q_text = inst["question_text"]
        gold_ans = inst["gold_answer"]

        # Cache figure extraction per chart image
        if c_idx not in chart_elements_cache:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                tmp_img.write(inst["img_bytes"])
                tmp_path = tmp_img.name
            try:
                with Image.open(tmp_path) as im:
                    W, H = im.size
                out = processor.process_figure(tmp_path)
                chart_elements_cache[c_idx] = (out["elements"], W, H, tmp_path)
            except Exception as e:
                chart_elements_cache[c_idx] = ([], 800, 600, tmp_path)

        elements, W, H, img_file_path = chart_elements_cache[c_idx]

        # 1. Deterministic Rule Answer
        pred_rule = answer_with_deterministic_rules(elements, qid, W, H)
        rule_exact, rule_fuzzy = evaluate_charxiv_answer(pred_rule, gold_ans, qid)

        # 2. Local VLM Answer
        if have_raw_log and (chart_id, qid) in raw_map:
            cached_entry = raw_map[(chart_id, qid)]
            pred_vlm = cached_entry["raw_response"]
            vlm_ms = cached_entry.get("wall_clock_runtime_ms", 0.0)
        else:
            prompt_inner = f"Question: {q_text}\nAnswer with only the direct short answer without any explanation."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_inner}
                    ]
                }
            ]
            prompt_with_template = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            t0 = time.perf_counter()
            try:
                res = mlx_vlm.generate(vlm_model, vlm_processor, prompt=prompt_with_template, image=img_file_path, max_tokens=24, verbose=False)
                pred_vlm = res.text.strip()
            except Exception as e:
                pred_vlm = f"ERROR: {e}"
            vlm_ms = round((time.perf_counter() - t0) * 1000, 2)

            if raw_vlm_file is not None:
                raw_log_entry = {
                    "chart_id": chart_id,
                    "qid": qid,
                    "question": q_text,
                    "gold_answer": gold_ans,
                    "model_name": vlm_name,
                    "weights_path": vlm_repo,
                    "prompt": prompt_with_template,
                    "raw_response": pred_vlm,
                    "wall_clock_runtime_ms": vlm_ms,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                }
                raw_vlm_file.write(json.dumps(raw_log_entry) + "\n")
                raw_vlm_file.flush()

        vlm_exact, vlm_fuzzy = evaluate_charxiv_answer(pred_vlm, gold_ans, qid)

        rec = {
            "chart_id": chart_id,
            "chart_idx": c_idx,
            "qid": qid,
            "question": q_text,
            "gold_answer": gold_ans,
            "rule_pred": pred_rule,
            "rule_exact": float(rule_exact),
            "rule_fuzzy": float(rule_fuzzy),
            "vlm_pred": pred_vlm,
            "vlm_exact": float(vlm_exact),
            "vlm_fuzzy": float(vlm_fuzzy),
            "vlm_runtime_ms": vlm_ms
        }
        eval_records.append(rec)

        if (i + 1) % 40 == 0 or (i + 1) == len(eval_sample):
            print(f"  [{i+1}/{len(eval_sample)}] Q{qid}: Rule Exact={rule_exact} ('{pred_rule[:15]}') | VLM Exact={vlm_exact}, Fuzzy={vlm_fuzzy} ('{pred_vlm[:15]}') | Gold: '{gold_ans[:15]}'")

    if raw_vlm_file is not None:
        raw_vlm_file.close()
    total_eval_time = round(time.perf_counter() - t_eval0, 2)
    print(f"\nCompleted evaluation of {len(eval_sample)} questions in {total_eval_time}s.")


    # 6. Bootstrap Resampling by Chart
    print("\nComputing Bootstrap 95% Confidence Intervals (resampling by chart)...")
    ci_results = bootstrap_charxiv_by_chart(eval_records, n_boot=1000)

    # Calculate empirical means
    overall_rule_ex = round(float(np.mean([r["rule_exact"] for r in eval_records])), 4)
    overall_rule_fz = round(float(np.mean([r["rule_fuzzy"] for r in eval_records])), 4)
    overall_vlm_ex = round(float(np.mean([r["vlm_exact"] for r in eval_records])), 4)
    overall_vlm_fz = round(float(np.mean([r["vlm_fuzzy"] for r in eval_records])), 4)

    text_recs = [r for r in eval_records if r["qid"] in TEXT_QIDS]
    text_rule_ex = round(float(np.mean([r["rule_exact"] for r in text_recs])), 4)
    text_rule_fz = round(float(np.mean([r["rule_fuzzy"] for r in text_recs])), 4)
    text_vlm_ex = round(float(np.mean([r["vlm_exact"] for r in text_recs])), 4)
    text_vlm_fz = round(float(np.mean([r["vlm_fuzzy"] for r in text_recs])), 4)

    num_recs = [r for r in eval_records if r["qid"] in NUMERIC_QIDS]
    num_rule_ex = round(float(np.mean([r["rule_exact"] for r in num_recs])), 4)
    num_rule_fz = round(float(np.mean([r["rule_fuzzy"] for r in num_recs])), 4)
    num_vlm_ex = round(float(np.mean([r["vlm_exact"] for r in num_recs])), 4)
    num_vlm_fz = round(float(np.mean([r["vlm_fuzzy"] for r in num_recs])), 4)

    per_qid_summary = {}
    print("\nPer-Question Breakdown (Exact / Fuzzy with 95% CIs):")
    for q in sorted(TARGET_QIDS):
        q_recs = [r for r in eval_records if r["qid"] == q]
        q_str = str(q)
        r_ex_mean = round(float(np.mean([r["rule_exact"] for r in q_recs])), 4)
        r_fz_mean = round(float(np.mean([r["rule_fuzzy"] for r in q_recs])), 4)
        v_ex_mean = round(float(np.mean([r["vlm_exact"] for r in q_recs])), 4)
        v_fz_mean = round(float(np.mean([r["vlm_fuzzy"] for r in q_recs])), 4)

        per_qid_summary[q_str] = {
            "question": TARGET_QIDS[q],
            "category": "text" if q in TEXT_QIDS else "numeric",
            "N": len(q_recs),
            "rules": {
                "exact": r_ex_mean,
                "exact_ci95": ci_results["per_qid"][q_str]["rules"]["exact_ci95"],
                "fuzzy": r_fz_mean,
                "fuzzy_ci95": ci_results["per_qid"][q_str]["rules"]["fuzzy_ci95"]
            },
            "vlm": {
                "exact": v_ex_mean,
                "exact_ci95": ci_results["per_qid"][q_str]["vlm"]["exact_ci95"],
                "fuzzy": v_fz_mean,
                "fuzzy_ci95": ci_results["per_qid"][q_str]["vlm"]["fuzzy_ci95"]
            }
        }
        print(f"  qid {q:<2} ({per_qid_summary[q_str]['category']}, N={len(q_recs)}):")
        print(f"    Rules: Exact = {r_ex_mean*100:.1f}% {per_qid_summary[q_str]['rules']['exact_ci95']}, Fuzzy = {r_fz_mean*100:.1f}%")
        print(f"    VLM:   Exact = {v_ex_mean*100:.1f}% {per_qid_summary[q_str]['vlm']['exact_ci95']}, Fuzzy = {v_fz_mean*100:.1f}%")

    print("\n" + "=" * 70)
    print("FINAL CHARXIV SUMMARY EVALUATION RESULTS (N = 320, 40 PER QID):")
    print(f"  Overall (N=320):")
    print(f"    Rules: Exact = {overall_rule_ex*100:.2f}% {ci_results['overall']['rules']['exact_ci95']}, Fuzzy = {overall_rule_fz*100:.2f}%")
    print(f"    VLM:   Exact = {overall_vlm_ex*100:.2f}% {ci_results['overall']['vlm']['exact_ci95']}, Fuzzy = {overall_vlm_fz*100:.2f}%")
    print(f"  Text QIDs (2, 3, 13) Only (N=120):")
    print(f"    Rules: Exact = {text_rule_ex*100:.2f}% {ci_results['text_qids']['rules']['exact_ci95']}, Fuzzy = {text_rule_fz*100:.2f}%")
    print(f"    VLM:   Exact = {text_vlm_ex*100:.2f}% {ci_results['text_qids']['vlm']['exact_ci95']}, Fuzzy = {text_vlm_fz*100:.2f}%")
    print(f"  Numeric QIDs (4, 5, 6, 7, 12) Only (N=200):")
    print(f"    Rules: Exact = {num_rule_ex*100:.2f}% {ci_results['numeric_qids']['rules']['exact_ci95']}, Fuzzy = {num_rule_fz*100:.2f}%")
    print(f"    VLM:   Exact = {num_vlm_ex*100:.2f}% {ci_results['numeric_qids']['vlm']['exact_ci95']}, Fuzzy = {num_vlm_fz*100:.2f}%")

    results = {
        "provenance": "human_external_dataset",
        "benchmark": "CharXiv Validation Split (NeurIPS 2024)",
        "source_url": "https://charxiv.github.io/",
        "sampling_method": "Fixed seed random.Random(0).shuffle per QID pool, taking first 40 questions per QID",
        "seed": 0,
        "questions_per_qid": per_qid_cap,
        "num_questions_evaluated": len(eval_sample),
        "runtime_s": total_eval_time,
        "scoring_rules": {
            "numeric_qids_4_5_6_7_12": "No substring containment; parsed float comparison with 5% relative tolerance; Q12 counts match exactly",
            "text_qids_2_3": "Exact = normalized equality; Fuzzy = SequenceMatcher ratio >= 0.80",
            "text_qid_13": "Comma-separated item sets comparison (Exact = set equality, Fuzzy = Jaccard >= 0.80)"
        },
        "overall_all_8_qids": {
            "num_questions": len(eval_sample),
            "rules": {"exact": overall_rule_ex, "fuzzy": overall_rule_fz, **ci_results["overall"]["rules"]},
            "vlm": {"exact": overall_vlm_ex, "fuzzy": overall_vlm_fz, **ci_results["overall"]["vlm"]}
        },
        "text_qids_2_3_13_separate": {
            "num_questions": len(text_recs),
            "rules": {"exact": text_rule_ex, "fuzzy": text_rule_fz, **ci_results["text_qids"]["rules"]},
            "vlm": {"exact": text_vlm_ex, "fuzzy": text_vlm_fz, **ci_results["text_qids"]["vlm"]}
        },
        "numeric_qids_4_5_6_7_12_separate": {
            "num_questions": len(num_recs),
            "rules": {"exact": num_rule_ex, "fuzzy": num_rule_fz, **ci_results["numeric_qids"]["rules"]},
            "vlm": {"exact": num_vlm_ex, "fuzzy": num_vlm_fz, **ci_results["numeric_qids"]["vlm"]}
        },
        "per_qid_breakdown": per_qid_summary,
        **get_runtime_metadata("pipelines/evaluate_charxiv.py")
    }


    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved updated CharXiv evaluation to: {OUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
