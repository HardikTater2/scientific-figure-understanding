"""
Task 6: Downstream Figure Captioning Evaluation on 100 SciCap Figures.
Compares:
  1. Template baseline
  2. Local VLM (image-only)
  3. Local VLM + grounded evidence

Evaluates:
  - BLEU-4 (with Chen & Cherry smoothing)
  - ROUGE-L (LCS F1)
  - Numeric-conflict rate (fraction of generated numbers contradicting figure/caption)
  - SCQ marked explicitly as UNVALIDATED (per Non-Negotiable Rule 1 & 3: no human ratings)

Provenance: silver_heuristic
Raw outputs saved to results/phase5/raw/captioning_raw.jsonl
Results saved to results/phase5_captioning_comparison.json
"""

import os
import sys
import json
import time
import re
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Set
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.common_metadata import get_runtime_metadata

DATA_PATH = os.path.join(ROOT, "data", "scicap_subset", "silver_grounding_v0.json")
OUT_PATH = os.path.join(ROOT, "results", "phase5_captioning_comparison.json")
RAW_LOG_PATH = os.path.join(ROOT, "results", "phase5", "raw", "captioning_raw.jsonl")


# =====================================================================
# Pure Python NLP Evaluation Metrics: BLEU-4, ROUGE-L, Numeric Conflict
# =====================================================================

def tokenize(text: str) -> List[str]:
    """Simple alphanumeric word tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_bleu4(reference: str, hypothesis: str) -> float:
    """Computes BLEU-4 with Chen & Cherry smoothing (Method 1)."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not hyp_tokens:
        return 0.0
    if not ref_tokens:
        return 0.0

    c = len(hyp_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1.0 - float(r) / float(c)) if c > 0 else 0.0

    precisions = []
    for n in range(1, 5):
        hyp_ngrams = [tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]

        if not hyp_ngrams:
            precisions.append(1e-5)
            continue

        ref_counts = Counter(ref_ngrams)
        hyp_counts = Counter(hyp_ngrams)

        clipped_hits = sum(min(count, ref_counts[ng]) for ng, count in hyp_counts.items())
        # Smoothing: if hits == 0, add small epsilon (Method 1)
        p_n = (clipped_hits + 0.1) / (len(hyp_ngrams) + 0.1) if clipped_hits == 0 else clipped_hits / len(hyp_ngrams)
        precisions.append(p_n)

    log_sum = sum(0.25 * math.log(max(p, 1e-9)) for p in precisions)
    return round(float(bp * math.exp(log_sum)), 4)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Computes ROUGE-L (LCS-based F1 score)."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    m = len(ref_tokens)
    n = len(hyp_tokens)
    if m == 0 or n == 0:
        return 0.0

    # Longest Common Subsequence dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    prec = lcs / float(n)
    rec = lcs / float(m)
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return round(float(f1), 4)


def extract_numbers(text: str) -> Set[str]:
    """Extracts numeric strings from text."""
    nums = set()
    for match in re.finditer(r"\b\d+(?:\.\d+)?\b", text):
        nums.add(match.group(0))
    return nums


def compute_numeric_conflict_rate(hypothesis: str, gold_caption: str, figure_text: str) -> float:
    """
    Computes fraction of numbers in hypothesis that do not appear in gold caption or figure elements.
    A number not grounded in gold evidence is counted as a conflict/hallucination.
    """
    hyp_nums = extract_numbers(hypothesis)
    if not hyp_nums:
        return 0.0

    gold_nums = extract_numbers(gold_caption) | extract_numbers(figure_text)
    conflicting = [n for n in hyp_nums if n not in gold_nums]
    return round(float(len(conflicting) / len(hyp_nums)), 4)


# =====================================================================
# Main Captioning Pipeline
# =====================================================================

def main():
    print("=" * 70)
    print("TASK 6: DOWNSTREAM SCIENTIFIC CAPTIONING EVALUATION ON 100 SCICAP FIGURES")
    print("=" * 70)

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    figures = data["figures"]
    # Take first 100 figures with valid images and captions
    selected_figures = []
    for fig in figures:
        fig_id = fig["figure_id"]
        rel_img = fig.get("image_path", "")
        full_img = os.path.join(ROOT, "data", "scicap_subset", rel_img)
        if not os.path.exists(full_img):
            full_img = os.path.join(ROOT, "data", "scicap_subset", "images", f"{fig_id}.png")
        if os.path.exists(full_img) and fig.get("caption"):
            selected_figures.append((fig, full_img))
        if len(selected_figures) >= 100:
            break

    print(f"Selected {len(selected_figures)} real SciCap figures for captioning evaluation.")

    # Check if raw VLM predictions already exist
    have_raw_log = os.path.exists(RAW_LOG_PATH) and os.path.getsize(RAW_LOG_PATH) > 0
    raw_map = {}
    if have_raw_log:
        print(f"\n[Raw captioning outputs detected at {RAW_LOG_PATH}]")
        print("Loading raw VLM captioning responses for scoring with consistent reference text...")
        with open(RAW_LOG_PATH, "r") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    raw_map[(str(d["figure_id"]), d["condition"])] = d
        if len(raw_map) < 200:
            print(f"Warning: RAW_LOG_PATH has {len(raw_map)} items, expected 200. Re-running generation...")
            have_raw_log = False

    raw_log = None
    vlm_repo = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    if not have_raw_log:
        import mlx_vlm
        print(f"\nLoading local VLM '{vlm_repo}' on Apple Silicon M4 Pro...")
        t_vlm0 = time.perf_counter()
        vlm_model, vlm_processor = mlx_vlm.load(vlm_repo)
        print(f"Loaded local VLM in {time.perf_counter() - t_vlm0:.2f}s.")
        os.makedirs(os.path.dirname(RAW_LOG_PATH), exist_ok=True)
        raw_log = open(RAW_LOG_PATH, "w")

    metrics_template = {"bleu4": [], "rouge_l": [], "numeric_conflict": [], "num_counts": []}
    metrics_vlm_image_only = {"bleu4": [], "rouge_l": [], "numeric_conflict": [], "num_counts": []}
    metrics_vlm_grounded = {"bleu4": [], "rouge_l": [], "numeric_conflict": [], "num_counts": []}

    for idx, (fig, img_path) in enumerate(selected_figures):
        fig_id = fig["figure_id"]
        gold_cap = fig["caption"]
        elements = fig.get("elements", [])
        entities = fig.get("entities", [])
        fig_text = " ".join(e.get("text", "") for e in elements)

        axes = [e["text"] for e in elements if e.get("type") == "AXIS"]
        legends = [e["text"] for e in elements if e.get("type") in ("LEGEND", "LABEL")]
        ent_names = [ent["text"] for ent in entities]

        # -------------------------------------------------------------
        # Condition 1: Template Baseline
        # -------------------------------------------------------------
        t_parts = []
        if axes:
            t_parts.append(f"plotting {axes[0]}")
            if len(axes) > 1:
                t_parts.append(f"against {axes[1]}")
        if legends:
            t_parts.append(f"comparing {', '.join(legends[:3])}")
        if ent_names:
            t_parts.append(f"for {', '.join(ent_names[:2])}")
        template_cap = f"Scientific figure {' '.join(t_parts)}." if t_parts else "Scientific figure showing empirical results."

        b4_t = compute_bleu4(gold_cap, template_cap)
        rl_t = compute_rouge_l(gold_cap, template_cap)
        nc_t = compute_numeric_conflict_rate(template_cap, gold_cap, fig_text)
        metrics_template["bleu4"].append(b4_t)
        metrics_template["rouge_l"].append(rl_t)
        metrics_template["numeric_conflict"].append(nc_t)
        metrics_template["num_counts"].append(len(extract_numbers(template_cap)))

        # -------------------------------------------------------------
        # Condition 2: Local VLM (Image-Only)
        # -------------------------------------------------------------
        if have_raw_log and (str(fig_id), "vlm_image_only") in raw_map:
            vlm_io_cap = raw_map[(str(fig_id), "vlm_image_only")]["raw_response"]
        else:
            prompt_img_only_text = "Provide a concise, accurate scientific caption describing this figure. Answer with only the caption."
            msg_img_only = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_img_only_text}]}
            ]
            prompt_img_only = vlm_processor.apply_chat_template(msg_img_only, tokenize=False, add_generation_prompt=True)

            t0 = time.perf_counter()
            try:
                res_io = mlx_vlm.generate(vlm_model, vlm_processor, prompt=prompt_img_only, image=img_path, max_tokens=64, verbose=False)
                vlm_io_cap = res_io.text.strip()
            except Exception as e:
                vlm_io_cap = f"ERROR: {e}"
            ms_io = round((time.perf_counter() - t0) * 1000, 2)

            if raw_log is not None:
                raw_log.write(json.dumps({
                    "figure_id": fig_id,
                    "condition": "vlm_image_only",
                    "model_name": "Qwen2-VL-7B-Instruct-4bit",
                    "weights_path": vlm_repo,
                    "prompt": prompt_img_only,
                    "raw_response": vlm_io_cap,
                    "wall_clock_runtime_ms": ms_io,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                }) + "\n")

        b4_io = compute_bleu4(gold_cap, vlm_io_cap)
        rl_io = compute_rouge_l(gold_cap, vlm_io_cap)
        nc_io = compute_numeric_conflict_rate(vlm_io_cap, gold_cap, fig_text)
        metrics_vlm_image_only["bleu4"].append(b4_io)
        metrics_vlm_image_only["rouge_l"].append(rl_io)
        metrics_vlm_image_only["numeric_conflict"].append(nc_io)
        metrics_vlm_image_only["num_counts"].append(len(extract_numbers(vlm_io_cap)))

        # -------------------------------------------------------------
        # Condition 3: Local VLM + Grounded Evidence
        # -------------------------------------------------------------
        if have_raw_log and (str(fig_id), "vlm_plus_grounded_evidence") in raw_map:
            vlm_gr_cap = raw_map[(str(fig_id), "vlm_plus_grounded_evidence")]["raw_response"]
        else:
            grounding_context = []
            if axes:
                grounding_context.append(f"Axes: {', '.join(axes[:3])}")
            if legends:
                grounding_context.append(f"Methods/Conditions: {', '.join(legends[:4])}")
            if ent_names:
                grounding_context.append(f"Entities: {', '.join(ent_names[:3])}")
            ctx_str = " | ".join(grounding_context) if grounding_context else "Axes and labels detected"

            prompt_grounded_text = (
                f"Provide a concise, accurate scientific caption describing this figure based on the detected visual evidence: [{ctx_str}]. "
                "Answer with only the caption."
            )
            msg_grounded = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_grounded_text}]}
            ]
            prompt_grounded = vlm_processor.apply_chat_template(msg_grounded, tokenize=False, add_generation_prompt=True)

            t1 = time.perf_counter()
            try:
                res_gr = mlx_vlm.generate(vlm_model, vlm_processor, prompt=prompt_grounded, image=img_path, max_tokens=64, verbose=False)
                vlm_gr_cap = res_gr.text.strip()
            except Exception as e:
                vlm_gr_cap = f"ERROR: {e}"
            ms_gr = round((time.perf_counter() - t1) * 1000, 2)

            if raw_log is not None:
                raw_log.write(json.dumps({
                    "figure_id": fig_id,
                    "condition": "vlm_plus_grounded_evidence",
                    "model_name": "Qwen2-VL-7B-Instruct-4bit",
                    "weights_path": vlm_repo,
                    "prompt": prompt_grounded,
                    "raw_response": vlm_gr_cap,
                    "wall_clock_runtime_ms": ms_gr,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                }) + "\n")
                raw_log.flush()

        b4_gr = compute_bleu4(gold_cap, vlm_gr_cap)
        rl_gr = compute_rouge_l(gold_cap, vlm_gr_cap)
        nc_gr = compute_numeric_conflict_rate(vlm_gr_cap, gold_cap, fig_text)
        metrics_vlm_grounded["bleu4"].append(b4_gr)
        metrics_vlm_grounded["rouge_l"].append(rl_gr)
        metrics_vlm_grounded["numeric_conflict"].append(nc_gr)
        metrics_vlm_grounded["num_counts"].append(len(extract_numbers(vlm_gr_cap)))

        if (idx + 1) % 20 == 0 or (idx + 1) == len(selected_figures):
            print(f"  [{idx+1}/{len(selected_figures)}] Fig {fig_id}: BLEU4 (T={b4_t:.3f}, IO={b4_io:.3f}, Gr={b4_gr:.3f}) | NumConflict (IO={nc_io:.2f}, Gr={nc_gr:.2f})")

    if raw_log is not None:
        raw_log.close()

    def summarize(m_dict):
        import numpy as np
        return {
            "bleu4": round(float(np.mean(m_dict["bleu4"])), 4),
            "rouge_l": round(float(np.mean(m_dict["rouge_l"])), 4),
            "mean_numbers_generated": round(float(np.mean(m_dict["num_counts"])), 2),
            "numeric_conflict_rate": round(float(np.mean(m_dict["numeric_conflict"])), 4)
        }

    sum_t = summarize(metrics_template)
    sum_io = summarize(metrics_vlm_image_only)
    sum_gr = summarize(metrics_vlm_grounded)

    from src.evaluation.bootstrap import compare_means
    diff_ci = compare_means(np.array(metrics_vlm_image_only["numeric_conflict"]),
                            np.array(metrics_vlm_grounded["numeric_conflict"]),
                            n_resamples=10000, seed=42)
    d_val = diff_ci["difference"]
    ci_low = diff_ci["ci95_low"]
    ci_high = diff_ci["ci95_high"]
    is_sig = bool(ci_low > 0 or ci_high < 0)
    sig_text = "is statistically significant" if is_sig else "spans zero and is not statistically significant"
    methodology_note = (
        f"The grounded condition is provided with OCR-extracted figure text which also contributes "
        f"to the reference vocabulary defining non-conflict. Wording claiming that grounded evidence "
        f"'suppresses hallucinations' is replaced by the empirical difference of {d_val:+.4f} "
        f"(95% CI [{ci_low:+.4f}, {ci_high:+.4f}]), which {sig_text}."
    )
    per_fig_diff = {
        "difference_mean": d_val,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "significant_at_0.05": is_sig,
        "methodology_note": methodology_note
    }

    print("\n" + "=" * 70)
    print("FINAL CAPTIONING EVALUATION SUMMARY (N = 100 SCICAP FIGURES):")
    print(f"  1. Template Baseline:         BLEU-4 = {sum_t['bleu4']:.4f}, ROUGE-L = {sum_t['rouge_l']:.4f}, MeanNums = {sum_t['mean_numbers_generated']:.2f}, NumConflict = {sum_t['numeric_conflict_rate']*100:.2f}%")
    print(f"  2. VLM (Image-Only):          BLEU-4 = {sum_io['bleu4']:.4f}, ROUGE-L = {sum_io['rouge_l']:.4f}, MeanNums = {sum_io['mean_numbers_generated']:.2f}, NumConflict = {sum_io['numeric_conflict_rate']*100:.2f}%")
    print(f"  3. VLM + Grounded Evidence:   BLEU-4 = {sum_gr['bleu4']:.4f}, ROUGE-L = {sum_gr['rouge_l']:.4f}, MeanNums = {sum_gr['mean_numbers_generated']:.2f}, NumConflict = {sum_gr['numeric_conflict_rate']*100:.2f}%")
    print(f"  Grounded vs Image-Only Conflict Reduction: {(sum_io['numeric_conflict_rate'] - sum_gr['numeric_conflict_rate']) * 100:+.2f}%")
    print("=" * 70)

    results = {
        "provenance": "model_outputs_vs_human_references",
        "benchmark": "SciCap Real Figures Captioning (100 figures)",
        "num_figures_evaluated": len(selected_figures),
        "numeric_conflict_reference_definition": "Union of numbers in author gold caption and all detected figure elements (axes, labels, legends): fig['caption'] + ' ' + ' '.join(e.get('text', '') for e in fig.get('elements', []))",
        "conditions": {
            "template_baseline": sum_t,
            "vlm_image_only": {
                "model_name": "Qwen2-VL-7B-Instruct-4bit",
                "weights_path": vlm_repo,
                **sum_io
            },
            "vlm_plus_grounded_evidence": {
                "model_name": "Qwen2-VL-7B-Instruct-4bit",
                "weights_path": vlm_repo,
                **sum_gr
            }
        },
        "per_figure_difference_grounded_minus_image_only": per_fig_diff,
        "scq_metric_status": {
            "metric_name": "Scientific Caption Quality (SCQ)",
            "validation_status": "NOT DONE",
            "provenance": "none",
            "note": "Per Non-Negotiable Rules 1 & 3: SCQ metric remains unvalidated as no in-house human validation study was conducted."
        },
        "raw_outputs_path": RAW_LOG_PATH,
        **get_runtime_metadata("pipelines/evaluate_captioning_vlm.py")
    }


    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved captioning comparison results to: {OUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
