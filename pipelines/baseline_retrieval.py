"""
SciCap Retrieval Baselines (Phase 2):
Compares caption-to-figure retrieval on the 287 SciCap real arXiv figures across:
  1. Random Chance Baseline (pool size N=288)
  2. TF-IDF Text-Only vs. Text+OCR
  3. BM25 Okapi Text-Only vs. Text+OCR
  4. Dense Semantic Embedding: SentenceTransformer('all-MiniLM-L6-v2') Text-Only vs. Text+OCR

Adheres strictly to Phase 2 & Final Fixes directives:
  - Exact OCR tokens loaded via sha1 of image file bytes from cache/ocr.json.
  - OCR text is concatenated as '{paragraph} {ocr_text}' for every text+ocr channel.
  - Identical lowercasing and regex word tokenization used for TF-IDF and BM25.
  - Paired bootstrap significance computed for text+OCR vs. text-only for each retriever.
  - Holm-Bonferroni step-down correction applied across the three OCR-gain tests.
  - Full metadata: provenance, generated_by, git_commit, timestamp.
"""

import os
import sys
import json
import time
import re
import hashlib
import subprocess
from typing import Dict, Any, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from statsmodels.stats.multitest import multipletests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.evaluation.bootstrap import compare_means
from pipelines.evaluate_scicap import strip_caption, summarize
from src.common_metadata import get_runtime_metadata

SCICAP_DIR = os.path.join(ROOT, "data", "scicap_subset")
OCR_CACHE = os.path.join(SCICAP_DIR, "cache", "ocr.json")
OUT_FILE = os.path.join(ROOT, "results", "phase2", "retrieval_baselines.json")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() or "unknown"
    except Exception:
        return "unknown"


def tokenize(text: str) -> List[str]:
    """Shared identical lowercasing and regex word tokenization for TF-IDF and BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def load_scicap_records() -> List[Dict[str, Any]]:
    with open(os.path.join(SCICAP_DIR, "metadata.json"), "r") as f:
        meta = json.load(f)
    with open(OCR_CACHE, "r") as f:
        ocr_cache = json.load(f)

    records = []
    for r in meta["records"]:
        cap = r.get("caption", "").strip()
        if not cap:
            continue
        para_raw = r.get("paragraph_text", "")
        para = strip_caption(para_raw, cap)

        # Get cached OCR text via sha1 of image file contents
        full_path = os.path.join(SCICAP_DIR, r["image_path"])
        ocr_tokens = []
        if os.path.exists(full_path):
            with open(full_path, "rb") as im_f:
                h = hashlib.sha1(im_f.read()).hexdigest()
            items = ocr_cache.get(h, [])
            if isinstance(items, list):
                ocr_tokens = [
                    str(item["text"]).strip()
                    for item in items
                    if isinstance(item, dict) and "text" in item and str(item["text"]).strip()
                ]

        ocr_text = " ".join(ocr_tokens)
        combined = f"{para} {ocr_text}".strip()

        records.append({
            "figure_id": r["figure_id"],
            "caption": cap,
            "paragraph": para,
            "ocr_text": ocr_text,
            "combined": combined
        })
    return records


def run_tfidf_retrieval(records: List[Dict[str, Any]], field: str = "paragraph") -> Tuple[np.ndarray, Dict[str, float]]:
    corpus = [r[field] for r in records]
    queries = [r["caption"] for r in records]

    # Identical tokenization and lowercasing
    tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
    doc_mat = tfidf.fit_transform(corpus)
    q_mat = tfidf.transform(queries)

    sims = (q_mat @ doc_mat.T).toarray()  # (N, N)

    ranks = []
    for idx in range(len(records)):
        scores = sims[idx]
        gold_score = scores[idx]

        greater = np.sum(scores > gold_score)
        ties = np.sum(scores == gold_score)
        expected_rank = greater + (ties + 1) / 2.0
        ranks.append(expected_rank)

    ranks_arr = np.array(ranks, dtype=float)
    return ranks_arr, summarize(ranks_arr)


def run_bm25_retrieval(records: List[Dict[str, Any]], field: str = "paragraph") -> Tuple[np.ndarray, Dict[str, float]]:
    corpus = [r[field] for r in records]
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    ranks = []
    for idx, r in enumerate(records):
        query_tokens = tokenize(r["caption"])
        scores = bm25.get_scores(query_tokens)
        gold_score = scores[idx]

        greater = np.sum(scores > gold_score)
        ties = np.sum(scores == gold_score)
        expected_rank = greater + (ties + 1) / 2.0
        ranks.append(expected_rank)

    ranks_arr = np.array(ranks, dtype=float)
    return ranks_arr, summarize(ranks_arr)


def run_dense_retrieval(records: List[Dict[str, Any]], field: str = "paragraph", model=None) -> Tuple[np.ndarray, Dict[str, float]]:
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    queries = [r["caption"] for r in records]
    corpus = [r[field] if r[field].strip() else "empty" for r in records]

    q_emb = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    d_emb = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)

    sims = q_emb @ d_emb.T  # (N, N)

    ranks = []
    for idx in range(len(records)):
        scores = sims[idx]
        gold_score = scores[idx]

        greater = np.sum(scores > gold_score)
        ties = np.sum(scores == gold_score)
        expected_rank = greater + (ties + 1) / 2.0
        ranks.append(expected_rank)

    ranks_arr = np.array(ranks, dtype=float)
    return ranks_arr, summarize(ranks_arr)


def main():
    print("=" * 70)
    print("SCICAP RETRIEVAL EVALUATION (EXACT OCR CONCATENATION & IDENTICAL TOKENIZATION)")
    print("=" * 70)

    records = load_scicap_records()
    N = len(records)
    print(f"Total figures in gallery (candidate pool size N): {N}")
    ocr_counts = [len(r['ocr_text'].split()) for r in records if r['ocr_text']]
    print(f"Figures with non-empty OCR text: {len(ocr_counts)}/{N} (mean {np.mean(ocr_counts):.1f} tokens)")

    # 1. Random Chance Baseline
    chance_r1 = 1.0 / N
    chance_r5 = min(5.0 / N, 1.0)
    chance_mrr = sum(1.0 / (k + 1) for k in range(N)) / N
    print(f"\n[Baseline 0: Random Chance (Pool Size = {N})]")
    print(f"  Recall@1: {chance_r1:.4f}, Recall@5: {chance_r5:.4f}, MRR: {chance_mrr:.4f}")

    # 2. TF-IDF Channel
    print("\n[Channel 1: TF-IDF (Identical Regex Word Tokenization)]")
    tfidf_ranks_text, tfidf_text = run_tfidf_retrieval(records, field="paragraph")
    print(f"  TF-IDF (Text Only):   R@1={tfidf_text['recall@1']:.4f}, R@5={tfidf_text['recall@5']:.4f}, MRR={tfidf_text['mrr']:.4f}")

    tfidf_ranks_comb, tfidf_comb = run_tfidf_retrieval(records, field="combined")
    print(f"  TF-IDF (Text + OCR):  R@1={tfidf_comb['recall@1']:.4f}, R@5={tfidf_comb['recall@5']:.4f}, MRR={tfidf_comb['mrr']:.4f}")

    # 3. BM25 Channel
    print("\n[Channel 2: BM25Okapi (Identical Regex Word Tokenization)]")
    bm25_ranks_text, bm25_text = run_bm25_retrieval(records, field="paragraph")
    print(f"  BM25 (Text Only):     R@1={bm25_text['recall@1']:.4f}, R@5={bm25_text['recall@5']:.4f}, MRR={bm25_text['mrr']:.4f}")

    bm25_ranks_comb, bm25_comb = run_bm25_retrieval(records, field="combined")
    print(f"  BM25 (Text + OCR):    R@1={bm25_comb['recall@1']:.4f}, R@5={bm25_comb['recall@5']:.4f}, MRR={bm25_comb['mrr']:.4f}")

    # 4. Dense Semantic Embedding Channel
    print("\n[Channel 3: Dense Semantic Embedding (all-MiniLM-L6-v2)]")
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    dense_ranks_text, dense_text = run_dense_retrieval(records, field="paragraph", model=dense_model)
    print(f"  Dense (Text Only):    R@1={dense_text['recall@1']:.4f}, R@5={dense_text['recall@5']:.4f}, MRR={dense_text['mrr']:.4f}")

    dense_ranks_comb, dense_comb = run_dense_retrieval(records, field="combined", model=dense_model)
    print(f"  Dense (Text + OCR):   R@1={dense_comb['recall@1']:.4f}, R@5={dense_comb['recall@5']:.4f}, MRR={dense_comb['mrr']:.4f}")

    # 5. Paired Bootstrap Significance Tests for OCR Gain
    print("\n[Computing Paired Bootstrap Significance: Text+OCR vs. Text-Only per Channel]")
    ci_tfidf_ocr_gain = compare_means(1.0 / tfidf_ranks_text, 1.0 / tfidf_ranks_comb, n_resamples=10000, seed=42)
    ci_bm25_ocr_gain = compare_means(1.0 / bm25_ranks_text, 1.0 / bm25_ranks_comb, n_resamples=10000, seed=42)
    ci_dense_ocr_gain = compare_means(1.0 / dense_ranks_text, 1.0 / dense_ranks_comb, n_resamples=10000, seed=42)

    # Cross-retriever comparison (Dense Text+OCR vs BM25 Text-Only)
    ci_dense_vs_bm25 = compare_means(1.0 / bm25_ranks_text, 1.0 / dense_ranks_comb, n_resamples=10000, seed=42)

    # 6. Apply Holm-Bonferroni correction across the 3 OCR-gain comparisons
    raw_p_values = [
        ci_tfidf_ocr_gain["p_value_bootstrap"],
        ci_bm25_ocr_gain["p_value_bootstrap"],
        ci_dense_ocr_gain["p_value_bootstrap"]
    ]
    reject, p_adjusted, _, _ = multipletests(raw_p_values, method="holm")

    ci_tfidf_ocr_gain["holm_p_value"] = round(float(p_adjusted[0]), 4)
    ci_tfidf_ocr_gain["significant_after_holm"] = bool(reject[0])

    ci_bm25_ocr_gain["holm_p_value"] = round(float(p_adjusted[1]), 4)
    ci_bm25_ocr_gain["significant_after_holm"] = bool(reject[1])

    ci_dense_ocr_gain["holm_p_value"] = round(float(p_adjusted[2]), 4)
    ci_dense_ocr_gain["significant_after_holm"] = bool(reject[2])

    print(f"  TF-IDF OCR Gain: MRR diff = {ci_tfidf_ocr_gain['difference']:+.4f} [{ci_tfidf_ocr_gain['ci95_low']}, {ci_tfidf_ocr_gain['ci95_high']}], raw p = {ci_tfidf_ocr_gain['p_value_bootstrap']}, holm p = {ci_tfidf_ocr_gain['holm_p_value']}")
    print(f"  BM25   OCR Gain: MRR diff = {ci_bm25_ocr_gain['difference']:+.4f} [{ci_bm25_ocr_gain['ci95_low']}, {ci_bm25_ocr_gain['ci95_high']}], raw p = {ci_bm25_ocr_gain['p_value_bootstrap']}, holm p = {ci_bm25_ocr_gain['holm_p_value']}")
    print(f"  Dense  OCR Gain: MRR diff = {ci_dense_ocr_gain['difference']:+.4f} [{ci_dense_ocr_gain['ci95_low']}, {ci_dense_ocr_gain['ci95_high']}], raw p = {ci_dense_ocr_gain['p_value_bootstrap']}, holm p = {ci_dense_ocr_gain['holm_p_value']}")

    results = {
        "provenance": "human_external_dataset",
        "benchmark": "SciCap Caption-to-Figure Retrieval (288 Real Figures)",
        "candidate_pool_size": N,
        "figures_with_ocr": len(ocr_counts),
        "tokenization": "Shared regex word tokenization r'\\b\\w+\\b' and lowercasing across TF-IDF and BM25",
        **get_runtime_metadata("pipelines/baseline_retrieval.py"),
        "random_chance": {
            "recall@1": round(chance_r1, 4),
            "recall@5": round(chance_r5, 4),
            "mrr": round(chance_mrr, 4)
        },
        "baselines": {
            "random_chance": {"R@1": round(chance_r1, 4), "R@5": round(chance_r5, 4), "MRR": round(chance_mrr, 4)},
            "tfidf_text_only": tfidf_text,
            "tfidf_text_plus_ocr": tfidf_comb,
            "bm25_text_only": bm25_text,
            "bm25_text_plus_ocr": bm25_comb,
            "dense_all_minilm_text_only": dense_text,
            "dense_all_minilm_text_plus_ocr": dense_comb
        },
        "ocr_gain_significance_tests": {
            "tfidf_ocr_gain": ci_tfidf_ocr_gain,
            "bm25_ocr_gain": ci_bm25_ocr_gain,
            "dense_ocr_gain": ci_dense_ocr_gain,
            "dense_text_ocr_vs_bm25_text": ci_dense_vs_bm25
        }
    }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved updated retrieval baselines to: {OUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
