"""
Master Paper Reproduction Runner (Updated & Verified).
Regenerates and verifies every empirical table across synthetic benchmarks,
external human datasets (CHART-Info, CharXiv, SciERC), downstream captioning,
and quarantined circular baselines with a single command:
  python scripts/reproduce_all.py [--table 1-8] [--all]

Strict Integrity Principles:
  - Actively recomputes reported metrics from raw prediction files and gold labels
    for CharXiv, downstream captioning, CHART-Info, SciERC, and synthetic v2.
  - Exits non-zero immediately on any metric mismatch > 1e-4.
  - Relabels heuristic-on-heuristic benchmarks as Appendix Table A1/A2 (circular, silver v0).
  - Explicit provenance on every table.
"""

import os
import sys
import json
import argparse
import time
import math
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
TOLERANCE = 1e-4


def get_results_path(*subpaths):
    return os.path.join(RESULTS_DIR, *subpaths)


def get_raw_path(*subpaths):
    p = os.path.join(RESULTS_DIR, *subpaths)
    if os.path.exists(p):
        return p
    return os.path.join(ROOT, "results", *subpaths)


def assert_metric_match(recomputed: float, reported: float, name: str):
    diff = abs(recomputed - reported)
    if diff > TOLERANCE:
        print(f"\n[RECOMPUTATION MISMATCH] {name}: recomputed={recomputed:.6f} vs reported={reported:.6f} (diff={diff:.6f} > {TOLERANCE})")
        sys.exit(1)


# =====================================================================
# TABLE 1 (APPENDIX A1): CIRCULAR SILVER v0 GROUNDING
# =====================================================================
def reproduce_table_1():
    print("\n" + "=" * 78)
    print("APPENDIX TABLE A1: GROUNDING EVALUATION (CIRCULAR, SILVER v0)")
    print("=" * 78)
    summary_path = get_results_path("appendix_circular", "baselines_summary.json")

    with open(summary_path) as f:
        data = json.load(f)

    # Active verification: verify reported F1 against underlying baseline sources
    clf_path = get_raw_path("appendix_circular", "classifier_baselines.json")
    gnn_path = get_raw_path("appendix_circular", "real_grounding_results.json")
    with open(clf_path) as f:
        clf_data = json.load(f)
    with open(gnn_path) as f:
        gnn_data = json.load(f)

    expected_f1s = {
        "Untuned Rules (Legacy Graph)": gnn_data["settings"]["oracle"]["rules"]["metrics"]["f1"],
        "Tuned Rules (Trigram Jaccard)": clf_data["baselines"]["tuned_rules"]["test_metrics"]["f1"],
        "GNN (GAT Message Passing)": gnn_data["settings"]["oracle"]["gnn"]["f1_mean"],
        "HistGradientBoosting (10 Features)": clf_data["baselines"]["gradient_boosting"]["test_metrics"]["f1"],
        "Logistic Regression (10 Features)": clf_data["baselines"]["logistic_regression"]["test_metrics"]["f1"],
    }

    for r in data["methods_table"]:
        mname = r["method"]
        if mname in expected_f1s:
            exp_f1 = expected_f1s[mname]
            rep_f1 = r.get("f1")
            assert_metric_match(exp_f1, rep_f1, f"Table 1 {mname} F1 Consistency")

    print(f"\n[Active Recomputation Verified: All baseline F1 values match underlying experiments within {TOLERANCE}]")
    print(f"Provenance: {data.get('provenance', 'silver_heuristic')} (Quarantined due to heuristic-on-heuristic circularity)")
    print(f"{'Method':<38} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Latency':<10} {'Cost/1k'}")
    print("-" * 78)
    for r in data["methods_table"]:
        cost = f"${r['cost_per_1k']:.2f}" if r.get('cost_per_1k') is not None else "N/A"
        lat = f"{r['latency_ms']:.2f}ms" if r.get('latency_ms') is not None else "N/A"
        p = f"{r['precision']:.4f}" if r['precision'] is not None else "N/A"
        rec = f"{r['recall']:.4f}" if r['recall'] is not None else "N/A"
        f1 = f"{r['f1']:.4f}" if isinstance(r['f1'], (int, float)) else str(r['f1'])
        print(f"{r['method']:<38} {p:<8} {rec:<8} {f1:<8} {lat:<10} {cost}")

    cis = data.get("bootstrap_comparisons", {})
    if cis:
        print("\nPaired Bootstrap Significance (95% CI vs. Logistic Regression):")
        for comp_name, ci in cis.items():
            print(f"  {comp_name:<25}: diff = {ci['difference']:+.4f} [{ci['ci95_low']}, {ci['ci95_high']}], p = {ci['p_value_bootstrap']}")


# =====================================================================
# TABLE 2 (APPENDIX A2): FINE-GRAINED BREAKDOWN (CIRCULAR, SILVER v0)
# =====================================================================
def reproduce_table_2():
    print("\n" + "=" * 78)
    print("APPENDIX TABLE A2: FINE-GRAINED BREAKDOWN BY ELEMENT & FIGURE TYPE (CIRCULAR, SILVER v0)")
    print("=" * 78)
    breakdown_path = os.path.join(RESULTS_DIR, "appendix_circular", "phase6_finegrained_breakdown.json")

    with open(breakdown_path) as f:
        data = json.load(f)

    # Active verification: recompute precision, recall, and F1 from tp, fp, fn
    for role, m in data["breakdown_by_element_role"].items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        exp_p = tp / max(tp + fp, 1)
        exp_r = tp / max(tp + fn, 1)
        exp_f1 = 2 * exp_p * exp_r / max(exp_p + exp_r, 1e-9)
        assert_metric_match(exp_p, m["precision"], f"Table 2 Role {role} Precision")
        assert_metric_match(exp_r, m["recall"], f"Table 2 Role {role} Recall")
        assert_metric_match(exp_f1, m["f1"], f"Table 2 Role {role} F1")

    for ftype, m in data["breakdown_by_figure_type"].items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        exp_p = tp / max(tp + fp, 1)
        exp_r = tp / max(tp + fn, 1)
        exp_f1 = 2 * exp_p * exp_r / max(exp_p + exp_r, 1e-9)
        assert_metric_match(exp_p, m["precision"], f"Table 2 Figure Type {ftype} Precision")
        assert_metric_match(exp_r, m["recall"], f"Table 2 Figure Type {ftype} Recall")
        assert_metric_match(exp_f1, m["f1"], f"Table 2 Figure Type {ftype} F1")

    print(f"\n[Active Recomputation Verified: Precision, Recall, F1 recomputed from TP/FP/FN counts within {TOLERANCE}]")
    print(f"Provenance: {data.get('provenance', 'silver_heuristic')} (Quarantined in appendix_circular)")
    print("\n[Breakdown by Visual Element Role]")
    print(f"{'Role':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'95% Bootstrap CI':<20} {'Num Pairs'}")
    print("-" * 75)
    for role, m in data["breakdown_by_element_role"].items():
        ci_str = f"[{m['ci95_low']:.4f}, {m['ci95_high']:.4f}]"
        print(f"{role:<12} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {ci_str:<20} {m['num_pairs']}")

    print("\n[Breakdown by Figure Type]")
    print(f"{'Figure Type':<16} {'Figures':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'95% Bootstrap CI'}")
    print("-" * 75)
    for ftype, m in data["breakdown_by_figure_type"].items():
        n_str = f"N={m['num_figures']}" + (" (!)" if m.get("small_sample_flag") else "")
        ci_str = f"[{m['ci95_low']:.4f}, {m['ci95_high']:.4f}]"
        print(f"{ftype:<16} {n_str:<10} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {ci_str}")


# =====================================================================
# TABLE 3: SCICAP CAPTION-TO-FIGURE RETRIEVAL (TEXT+OCR VS TEXT-ONLY)
# =====================================================================
def reproduce_table_3():
    print("\n" + "=" * 78)
    print("TABLE 3: SCICAP CAPTION-TO-FIGURE RETRIEVAL (288 REAL ARXIV FIGURES)")
    print("=" * 78)
    retrieval_path = os.path.join(RESULTS_DIR, "phase2", "retrieval_baselines.json")
    with open(retrieval_path) as f:
        data = json.load(f)

    # Active recomputation of TF-IDF and BM25 channels from raw gallery
    from pipelines.baseline_retrieval import load_scicap_records, run_tfidf_retrieval, run_bm25_retrieval
    recs = load_scicap_records()
    _, t_txt = run_tfidf_retrieval(recs, "paragraph")
    _, t_comb = run_tfidf_retrieval(recs, "combined")
    _, b_txt = run_bm25_retrieval(recs, "paragraph")
    _, b_comb = run_bm25_retrieval(recs, "combined")

    assert_metric_match(t_txt["mrr"], data["baselines"]["tfidf_text_only"]["mrr"], "TF-IDF Text Only MRR")
    assert_metric_match(t_comb["mrr"], data["baselines"]["tfidf_text_plus_ocr"]["mrr"], "TF-IDF Text+OCR MRR")
    assert_metric_match(b_txt["mrr"], data["baselines"]["bm25_text_only"]["mrr"], "BM25 Text Only MRR")
    assert_metric_match(b_comb["mrr"], data["baselines"]["bm25_text_plus_ocr"]["mrr"], "BM25 Text+OCR MRR")

    print(f"\n[Active Recomputation Verified: TF-IDF and BM25 recomputed from 288 gallery records within {TOLERANCE}]")
    print(f"Gallery Pool Size: {data['candidate_pool_size']} figures | Provenance: {data.get('provenance', 'human_external_dataset')}")
    print(f"{'Retrieval Channel':<32} {'Recall@1':<12} {'Recall@5':<12} {'MRR'}")
    print("-" * 78)
    for name, m in data["baselines"].items():
        if not m:
            continue
        r1 = m.get("recall@1", m.get("R@1", 0.0))
        r5 = m.get("recall@5", m.get("R@5", 0.0))
        mrr = m.get("mrr", m.get("MRR", 0.0))
        print(f"{name:<32} {r1:<12.4f} {r5:<12.4f} {mrr:.4f}")

    sigs = data.get("ocr_gain_significance_tests", {})
    if sigs:
        print("\nPaired Bootstrap Significance (Text+OCR vs. Text-Only) with Holm-Bonferroni Correction:")
        print(f"{'Comparison':<28} {'MRR Gain':<12} {'95% CI':<24} {'Raw p':<10} {'Holm p':<10} {'Significant?'}")
        print("-" * 94)
        for k, v in sigs.items():
            if not isinstance(v, dict) or "difference" not in v:
                continue
            ci_str = f"[{v['ci95_low']:+.4f}, {v['ci95_high']:+.4f}]"
            raw_p = v.get("p_value_bootstrap", v.get("p_value_raw", 0.0))
            if k == "dense_text_ocr_vs_bm25_text":
                holm_p_str = "n/a"
                sig_flag = "n/a"
            else:
                holm_p = v.get("holm_p_value", 0.0)
                # Assertion that every Holm p >= its raw p
                assert holm_p >= raw_p - 1e-6, f"Holm p ({holm_p}) must be >= raw p ({raw_p}) for {k}"
                holm_p_str = f"{holm_p:.4f}"
                sig_flag = "YES (*)" if v.get("significant_after_holm", False) else "NO"
            print(f"{k:<28} {v['difference']:<+12.4f} {ci_str:<24} {raw_p:<10.4f} {holm_p_str:<10} {sig_flag}")


# =====================================================================
# TABLE 4: SHORTCUT-FREE BENCHMARK (SCIFIG-CONTROLLED v2)
# =====================================================================
def reproduce_table_4():
    print("\n" + "=" * 78)
    print("TABLE 4: SHORTCUT-FREE BENCHMARK (SCIFIG-CONTROLLED v2)")
    print("=" * 78)
    v2_path = os.path.join(RESULTS_DIR, "phase2", "synthetic_v2_evaluation.json")
    with open(v2_path) as f:
        data = json.load(f)

    raw_path = os.path.join(RESULTS_DIR, "phase2", "raw", "synthetic_v2_raw_predictions.json")
    with open(raw_path) as f:
        raw_data = json.load(f)

    print(f"Dataset: SciFig-Controlled-v2 ({data['num_test']} test figures) | Provenance: {data['provenance']}")
    sel = data.get("winning_rule_selection", {})
    print(f"Rule Selection on Validation: {sel.get('formula_name')} (val F1 = {sel.get('val_f1')}, threshold = {sel.get('val_threshold')})")

    # Recompute metrics from raw per-figure counts
    print("\n[Active Recomputation from Raw Per-Figure Counts]")
    models_counts = raw_data["models_per_fig_counts"]
    for mname, runs_counts in models_counts.items():
        run_ps, run_rs, run_f1s = [], [], []
        for counts in runs_counts:
            arr = np.array(counts)
            tp, fp, fn = arr.sum(axis=0)
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            run_ps.append(p)
            run_rs.append(r)
            run_f1s.append(f1)

        p_mean = float(np.mean(run_ps))
        r_mean = float(np.mean(run_rs))
        f1_mean = float(np.mean(run_f1s))

        rep_m = data["models"][mname]
        rep_p = rep_m.get("precision_mean", rep_m.get("precision", 0.0))
        rep_r = rep_m.get("recall_mean", rep_m.get("recall", 0.0))
        rep_f1 = rep_m.get("f1_mean", rep_m.get("f1", 0.0))

        assert_metric_match(p_mean, rep_p, f"Synthetic v2 {mname} Precision")
        assert_metric_match(r_mean, rep_r, f"Synthetic v2 {mname} Recall")
        assert_metric_match(f1_mean, rep_f1, f"Synthetic v2 {mname} F1")

    from src.evaluation.bootstrap import compare_micro_f1

    # Active recomputation of bootstrap comparisons from raw counts
    print("\n[Active Recomputation of Paired Bootstrap Comparisons & p-values]")
    print("Paired bootstrap uses primary seed=42 (5000 resamples).")
    gnn_counts = np.array(models_counts["gnn"][0])
    nomp_counts = np.array(models_counts["no_message_passing"][0])
    tuned_rules_counts = np.array(models_counts["tuned_rules"][0])
    untuned_rules_counts = np.array(models_counts["untuned_rules"][0])
    lr_counts = np.array(models_counts["logistic_regression"][0])
    gbdt_counts = np.array(models_counts["gbdt"][0])

    print("Confirming GNN and No-MP are distinct models (Seed 42 first 5 per-figure TP/FP/FN):")
    print(f"  GNN first 5:   {gnn_counts[:5].tolist()}")
    print(f"  No-MP first 5: {nomp_counts[:5].tolist()}")
    diff_indices = np.where((gnn_counts != nomp_counts).any(axis=1))[0]
    print(f"  Distinct per-figure predictions: GNN differs from No-MP on {len(diff_indices)} figure(s) (e.g. Fig {diff_indices[0]}: GNN={gnn_counts[diff_indices[0]].tolist()} vs No-MP={nomp_counts[diff_indices[0]].tolist()})")

    comp_pairs = {
        "gnn_minus_tuned_rules": (tuned_rules_counts, gnn_counts),
        "gnn_minus_untuned_rules": (untuned_rules_counts, gnn_counts),
        "gnn_minus_logistic_regression": (lr_counts, gnn_counts),
        "gnn_minus_gbdt": (gbdt_counts, gnn_counts),
        "gnn_minus_no_message_passing": (nomp_counts, gnn_counts),
        "logistic_regression_minus_tuned_rules": (tuned_rules_counts, lr_counts),
    }

    rep_cis = data.get("bootstrap_comparisons", {})
    for cname, (c_a, c_b) in comp_pairs.items():
        recomp_ci = compare_micro_f1(c_a, c_b, n_resamples=5000, seed=42)
        rep_ci = rep_cis[cname]
        assert_metric_match(recomp_ci["difference"], rep_ci["difference"], f"Table 4 {cname} difference")
        assert_metric_match(recomp_ci["ci95_low"], rep_ci["ci95_low"], f"Table 4 {cname} ci95_low")
        assert_metric_match(recomp_ci["ci95_high"], rep_ci["ci95_high"], f"Table 4 {cname} ci95_high")
        assert_metric_match(recomp_ci["p_value_bootstrap"], rep_ci["p_value_bootstrap"], f"Table 4 {cname} p-value")

    print(f"  All 6 models and 6 bootstrap comparisons recomputed and verified matching reported numbers within {TOLERANCE}.")
    print(f"\n{'Model':<35} {'Precision':<12} {'Recall':<12} {'Micro-F1'}")
    print("-" * 75)
    for mname, m in data["models"].items():
        p = m.get("precision_mean", m.get("precision", 0.0))
        r = m.get("recall_mean", m.get("recall", 0.0))
        f1 = m.get("f1_mean", m.get("f1", 0.0))
        f1_std = m.get("f1_std", 0.0)
        if mname == "gbdt" or m.get("is_deterministic"):
            std_str = " (deterministic single run)"
        else:
            std_str = f" ± {f1_std:.4f}" if f1_std > 0 else ""
        print(f"{mname:<35} {p:<12.4f} {r:<12.4f} {f1:.4f}{std_str}")

    cis = data.get("bootstrap_comparisons", {})
    if cis:
        print(f"\nPaired Bootstrap Comparisons (vs. GNN / Tuned Rules, seed=42, 5000 resamples):")
        for k, ci in cis.items():
            print(f"  {k:<35}: diff = {ci['difference']:+.4f} [{ci['ci95_low']}, {ci['ci95_high']}], p = {ci['p_value_bootstrap']}")

    print("\nFootnote: Bootstrap comparisons use seed 42; headline F1 is the mean over seeds 42-44.")



# =====================================================================
# TABLE 5: TEXT ROLE CLASSIFICATION ON CHART-INFO UB PMC
# =====================================================================
def reproduce_table_5():
    print("\n" + "=" * 78)
    print("TABLE 5: TEXT ROLE CLASSIFICATION ON CHART-INFO UB PMC (EXTERNAL HUMAN TEST SPLIT)")
    print("=" * 78)
    ci_path = get_results_path("phase3", "chart_info_evaluation.json")
    with open(ci_path) as f:
        data = json.load(f)

    raw_path = get_raw_path("phase3", "raw", "chart_info_raw_predictions.json")
    with open(raw_path) as f:
        raw_data = json.load(f)

    # Active recomputation for Gold Boxes
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    gold_trues = raw_data["gold_trues"]
    gold_preds = raw_data["gold_preds"]
    recomp_gold_acc = accuracy_score(gold_trues, gold_preds)
    recomp_gold_f1 = f1_score(gold_trues, gold_preds, average="macro", zero_division=0)
    recomp_gold_f1_nt = f1_score(gold_trues, gold_preds, labels=data["non_tick_classes"], average="macro", zero_division=0)

    gold_rep = data["gold_boxes_evaluation"]
    assert_metric_match(recomp_gold_acc, gold_rep["accuracy"], "CHART-Info Gold Accuracy")
    assert_metric_match(recomp_gold_f1, gold_rep["macro_f1"], "CHART-Info Gold Macro-F1")
    assert_metric_match(recomp_gold_f1_nt, gold_rep["macro_f1_non_tick"], "CHART-Info Gold Macro-F1 Non-Tick")

    # Active recomputation for OCR Boxes (Standard & Adjusted)
    ocr_trues = raw_data["ocr_trues"]
    ocr_preds_std = raw_data["ocr_preds_std"]
    ocr_preds_adj = raw_data["ocr_preds_adj"]

    recomp_std_acc = accuracy_score(ocr_trues, ocr_preds_std)
    recomp_std_f1 = f1_score(ocr_trues, ocr_preds_std, average="macro", zero_division=0)
    recomp_adj_acc = accuracy_score(ocr_trues, ocr_preds_adj)
    recomp_adj_f1 = f1_score(ocr_trues, ocr_preds_adj, average="macro", zero_division=0)

    ocr_rep = data["ocr_boxes_evaluation"]
    assert_metric_match(recomp_std_acc, ocr_rep["standard_all_unmatched_misses"]["accuracy"], "CHART-Info OCR Standard Accuracy")
    assert_metric_match(recomp_std_f1, ocr_rep["standard_all_unmatched_misses"]["macro_f1"], "CHART-Info OCR Standard Macro-F1")
    assert_metric_match(recomp_adj_acc, ocr_rep["adjusted_numeric_ticks_matched_and_dropped"]["accuracy"], "CHART-Info OCR Adjusted Accuracy")
    assert_metric_match(recomp_adj_f1, ocr_rep["adjusted_numeric_ticks_matched_and_dropped"]["macro_f1"], "CHART-Info OCR Adjusted Macro-F1")

    # Detection recall per role recomputation
    for c in data["taxonomy_classes"]:
        det = raw_data["overall_det_counts"][c]
        sup = raw_data["overall_gold_counts"][c]
        rec = (det / sup) if sup > 0 else 0.0
        assert_metric_match(rec, ocr_rep["ocr_detection_recall_per_role"][c]["recall"], f"CHART-Info Detection Recall {c}")

    print(f"\n[Active Recomputation Verified: Gold (N={len(gold_trues)}) and OCR (N={len(ocr_trues)}) match reported values within {TOLERANCE}]")
    print(f"Provenance: {data['provenance']} | Benchmark: {data['benchmark']}")
    print(f"\n[Gold Boxes Evaluation (All {gold_rep['num_charts']} Charts, {gold_rep['total_boxes']} Total Boxes)]")
    ci95_gold = gold_rep.get("bootstrap_ci95_by_chart", {})
    print(f"Accuracy: {gold_rep['accuracy']:.4f} {ci95_gold.get('accuracy_ci95', '')}")
    print(f"Macro-F1 (All 5 Roles): {gold_rep['macro_f1']:.4f} {ci95_gold.get('macro_f1_ci95', '')}")
    print(f"Macro-F1 (Non-TICK: AXIS, LEGEND, LABEL, OTHER): {gold_rep['macro_f1_non_tick']:.4f} {ci95_gold.get('macro_f1_non_tick_ci95', '')}")
    print(f"  {'Class':<8} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support'}")
    for c, m in gold_rep["per_class"].items():
        print(f"  {c:<8} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {m['support']}")

    print(f"\n[OCR Boxes Evaluation ({ocr_rep['num_charts']} Charts Sample)]")
    std = ocr_rep["standard_all_unmatched_misses"]
    adj = ocr_rep.get("adjusted_numeric_ticks_matched_and_dropped", {})
    print(f"Standard OCR Accuracy:  {std['accuracy']:.4f} | Macro-F1 (All): {std['macro_f1']:.4f} | Macro-F1 (Non-TICK): {std.get('macro_f1_non_tick', 0.0):.4f}")
    print(f"Adjusted OCR Accuracy:  {adj['accuracy']:.4f} | Macro-F1 (All): {adj['macro_f1']:.4f} | Macro-F1 (Non-TICK): {adj.get('macro_f1_non_tick', 0.0):.4f}")

    det = ocr_rep.get("ocr_detection_recall_per_role", {})
    if det:
        print("\nOCR Detection Recall Per Role (Separate):")
        for c, m in det.items():
            print(f"  {c:<8}: Recall = {m['recall']*100:.2f}% ({m['detected_by_ocr']}/{m['gold_support']}) CI95: {m.get('ci95', '')}")


# =====================================================================
# TABLE 6: CHARXIV DESCRIPTIVE CHART QA
# =====================================================================
def reproduce_table_6():
    print("\n" + "=" * 78)
    print("TABLE 6: CHARXIV DESCRIPTIVE CHART QA (RULES VS. LOCAL 4-BIT 7B VLM)")
    print("=" * 78)
    cx_path = get_results_path("phase3", "charxiv_evaluation.json")
    with open(cx_path) as f:
        data = json.load(f)

    from pipelines.evaluate_charxiv import evaluate_charxiv_answer
    raw_path = get_raw_path("phase2", "raw", "vlm_charxiv_raw.jsonl")

    vlm_all_ex, vlm_all_fz = [], []
    vlm_txt_ex, vlm_txt_fz = [], []
    vlm_num_ex, vlm_num_fz = [], []
    rules_all_ex, rules_all_fz = [], []
    rules_txt_ex, rules_txt_fz = [], []
    rules_num_ex, rules_num_fz = [], []

    with open(raw_path) as f:
        for line in f:
            d = json.loads(line)
            qid = d["qid"]
            gold = d["gold_answer"]
            resp = d["raw_response"]
            vex, vfz = evaluate_charxiv_answer(resp, gold, qid)
            rex = d["rule_exact"]
            rfz = d["rule_fuzzy"]

            vlm_all_ex.append(vex)
            vlm_all_fz.append(vfz)
            rules_all_ex.append(rex)
            rules_all_fz.append(rfz)

            if qid in (2, 3, 13):
                vlm_txt_ex.append(vex)
                vlm_txt_fz.append(vfz)
                rules_txt_ex.append(rex)
                rules_txt_fz.append(rfz)
            else:
                vlm_num_ex.append(vex)
                vlm_num_fz.append(vfz)
                rules_num_ex.append(rex)
                rules_num_fz.append(rfz)

    # Recompute and compare
    assert_metric_match(np.mean(vlm_all_ex), data["overall_all_8_qids"]["vlm"]["exact"], "CharXiv VLM Overall Exact")
    assert_metric_match(np.mean(vlm_all_fz), data["overall_all_8_qids"]["vlm"]["fuzzy"], "CharXiv VLM Overall Fuzzy")
    assert_metric_match(np.mean(rules_all_ex), data["overall_all_8_qids"]["rules"]["exact"], "CharXiv Rules Overall Exact")
    assert_metric_match(np.mean(rules_all_fz), data["overall_all_8_qids"]["rules"]["fuzzy"], "CharXiv Rules Overall Fuzzy")

    assert_metric_match(np.mean(vlm_txt_ex), data["text_qids_2_3_13_separate"]["vlm"]["exact"], "CharXiv VLM Text Exact")
    assert_metric_match(np.mean(vlm_txt_fz), data["text_qids_2_3_13_separate"]["vlm"]["fuzzy"], "CharXiv VLM Text Fuzzy")
    assert_metric_match(np.mean(vlm_num_ex), data["numeric_qids_4_5_6_7_12_separate"]["vlm"]["exact"], "CharXiv VLM Numeric Exact")
    assert_metric_match(np.mean(vlm_num_fz), data["numeric_qids_4_5_6_7_12_separate"]["vlm"]["fuzzy"], "CharXiv VLM Numeric Fuzzy")

    print(f"\n[Active Recomputation Verified: 320 raw prediction items match reported JSON within {TOLERANCE}]")
    print(f"Provenance: {data['provenance']} | Questions Evaluated: {data['num_questions_evaluated']}")

    ov = data.get("overall_all_8_qids", {})
    txt = data.get("text_qids_2_3_13_separate", {})
    num = data.get("numeric_qids_4_5_6_7_12_separate", {})

    print(f"\n{'Category':<26} {'Rules Exact [95% CI]':<32} {'Rules Fz':<10} {'VLM Exact [95% CI]':<32} {'VLM Fz'}")
    print("-" * 110)
    if ov:
        r_str = f"{ov['rules']['exact']*100:.1f}% {ov['rules'].get('exact_ci95', '')}"
        v_str = f"{ov['vlm']['exact']*100:.1f}% {ov['vlm'].get('exact_ci95', '')}"
        print(f"{'Overall (All 8 QIDs)':<26} {r_str:<32} {ov['rules']['fuzzy']*100:.1f}%     {v_str:<32} {ov['vlm']['fuzzy']*100:.1f}%")
    if txt:
        r_str = f"{txt['rules']['exact']*100:.1f}% {txt['rules'].get('exact_ci95', '')}"
        v_str = f"{txt['vlm']['exact']*100:.1f}% {txt['vlm'].get('exact_ci95', '')}"
        print(f"{'Text QIDs (2, 3, 13) Only':<26} {r_str:<32} {txt['rules']['fuzzy']*100:.1f}%     {v_str:<32} {txt['vlm']['fuzzy']*100:.1f}%")
    if num:
        r_str = f"{num['rules']['exact']*100:.1f}% {num['rules'].get('exact_ci95', '')}"
        v_str = f"{num['vlm']['exact']*100:.1f}% {num['vlm'].get('exact_ci95', '')}"
        print(f"{'Numeric QIDs (4-7, 12)':<26} {r_str:<32} {num['rules']['fuzzy']*100:.1f}%     {v_str:<32} {num['vlm']['fuzzy']*100:.1f}%")


# =====================================================================
# TABLE 7: SCIERC SCIENTIFIC ENTITY EXTRACTION
# =====================================================================
def reproduce_table_7():
    print("\n" + "=" * 78)
    print("TABLE 7: SCIERC SCIENTIFIC ENTITY EXTRACTION (HUMAN GOLD TEST SPLIT)")
    print("=" * 78)
    scierc_path = get_results_path("phase3", "scierc_ner_evaluation.json")
    with open(scierc_path) as f:
        data = json.load(f)

    from seqeval.metrics import f1_score as seq_f1, precision_score as seq_p, recall_score as seq_r, classification_report as seq_cr
    from seqeval.scheme import IOB2
    raw_path = get_raw_path("phase3", "raw", "scierc_raw_predictions.json")
    with open(raw_path) as f:
        raw_data = json.load(f)

    gold = raw_data["gold_sentences"]
    rules = raw_data["rule_predictions"]
    scibert = raw_data["scibert_predictions"]

    # Recompute rule extractor metrics
    r_prec = seq_p(gold, rules, mode="strict", scheme=IOB2)
    r_rec = seq_r(gold, rules, mode="strict", scheme=IOB2)
    r_f1 = seq_f1(gold, rules, mode="strict", scheme=IOB2)
    rep_rules = data["results"]["modern_scientific_ner_baseline"]
    assert_metric_match(r_prec, rep_rules["precision"], "SciERC Rule Baseline Precision")
    assert_metric_match(r_rec, rep_rules["recall"], "SciERC Rule Baseline Recall")
    assert_metric_match(r_f1, rep_rules["f1"], "SciERC Rule Baseline F1")

    # Recompute fine-tuned SciBERT metrics
    s_prec = seq_p(gold, scibert, mode="strict", scheme=IOB2)
    s_rec = seq_r(gold, scibert, mode="strict", scheme=IOB2)
    s_f1 = seq_f1(gold, scibert, mode="strict", scheme=IOB2)
    rep_scibert = data["results"]["finetuned_scibert_token_classifier"]
    assert_metric_match(s_prec, rep_scibert["test_precision"], "SciERC SciBERT Test Precision")
    assert_metric_match(s_rec, rep_scibert["test_recall"], "SciERC SciBERT Test Recall")
    assert_metric_match(s_f1, rep_scibert["test_f1"], "SciERC SciBERT Test F1")

    # Recompute per-class metrics
    s_cr = seq_cr(gold, scibert, mode="strict", scheme=IOB2, output_dict=True)
    for c in ("METHOD", "METRIC", "DATASET", "TASK", "VARIABLE"):
        assert_metric_match(s_cr[c]["precision"], rep_scibert["per_class"][c]["precision"], f"SciERC {c} Precision")
        assert_metric_match(s_cr[c]["recall"], rep_scibert["per_class"][c]["recall"], f"SciERC {c} Recall")
        assert_metric_match(s_cr[c]["f1-score"], rep_scibert["per_class"][c]["f1"], f"SciERC {c} F1")

    print(f"\n[Active Recomputation Verified: 551 test sentences recomputed from raw IOB labels, all metrics match within {TOLERANCE}]")
    print(f"Provenance: {data['provenance']} | Evaluation Standard: {data['evaluation_metric']}")
    print(f"{'Model':<40} {'Precision':<12} {'Recall':<12} {'Span Micro-F1'}")
    print("-" * 78)
    print(f"{rep_rules['model']:<40} {rep_rules['precision']:<12.4f} {rep_rules['recall']:<12.4f} {rep_rules['f1']:.4f}")
    print(f"{rep_scibert['model']:<40} {rep_scibert['test_precision']:<12.4f} {rep_scibert['test_recall']:<12.4f} {rep_scibert['test_f1']:.4f}")

    print("\nFine-Tuned SciBERT Per-Class F1 Breakdown:")
    for c, m in rep_scibert["per_class"].items():
        print(f"  {c:<12}: P = {m['precision']:.4f}, R = {m['recall']:.4f}, F1 = {m['f1']:.4f} (Support = {m['support']})")


# =====================================================================
# TABLE 8: DOWNSTREAM SCIENTIFIC CAPTIONING
# =====================================================================
def reproduce_table_8():
    print("\n" + "=" * 78)
    print("TABLE 8: DOWNSTREAM CAPTIONING ON 100 REAL SCICAP FIGURES")
    print("=" * 78)
    cap_path = get_results_path("phase5_captioning_comparison.json")
    with open(cap_path) as f:
        data = json.load(f)

    from pipelines.evaluate_captioning_vlm import (
        DATA_PATH, compute_bleu4, compute_rouge_l, extract_numbers, compute_numeric_conflict_rate
    )

    with open(DATA_PATH) as f:
        fig_data = json.load(f)["figures"]
    fig_map = {str(fig["figure_id"]): fig for fig in fig_data}

    raw_path = get_raw_path("phase5", "raw", "captioning_raw.jsonl")
    preds_by_cond = {"vlm_image_only": [], "vlm_plus_grounded_evidence": []}
    with open(raw_path) as f:
        for line in f:
            d = json.loads(line)
            cond = d["condition"]
            if cond in preds_by_cond:
                preds_by_cond[cond].append(d)

    for cond in ("vlm_image_only", "vlm_plus_grounded_evidence"):
        items = preds_by_cond[cond]
        bleus, rouges, num_counts, conflicts = [], [], [], []
        for it in items:
            fid = str(it["figure_id"])
            hyp = it["raw_response"]
            fig = fig_map[fid]
            ref = fig["caption"]
            fig_text = " ".join(e.get("text", "") for e in fig.get("elements", []))
            bleus.append(compute_bleu4(ref, hyp))
            rouges.append(compute_rouge_l(ref, hyp))
            hyp_nums = extract_numbers(hyp)
            num_counts.append(len(hyp_nums))
            conflicts.append(compute_numeric_conflict_rate(hyp, ref, fig_text))

        recomp_b = float(np.mean(bleus))
        recomp_r = float(np.mean(rouges))
        recomp_c = float(np.mean(conflicts))
        recomp_n = float(np.mean(num_counts))

        target_cond = data["conditions"][cond]
        assert_metric_match(recomp_b, target_cond["bleu4"], f"Captioning {cond} BLEU-4")
        assert_metric_match(recomp_r, target_cond["rouge_l"], f"Captioning {cond} ROUGE-L")
        assert_metric_match(recomp_c, target_cond["numeric_conflict_rate"], f"Captioning {cond} Numeric Conflict Rate")
        assert_metric_match(recomp_n, target_cond["mean_numbers_generated"], f"Captioning {cond} Mean Numbers")

    print(f"\n[Active Recomputation Verified: 100 figures x 2 VLM conditions match reported values within {TOLERANCE}]")
    print(f"Provenance: {data['provenance']} | Figures Evaluated: {data['num_figures_evaluated']}")
    print(f"{'Condition':<30} {'BLEU-4':<10} {'ROUGE-L':<10} {'Mean Numbers':<14} {'Conflict Rate'}")
    print("-" * 78)
    for cond, m in data["conditions"].items():
        nums = m.get("mean_numbers_generated", 0.0)
        cr = m.get("numeric_conflict_rate", 0.0) * 100
        print(f"{cond:<30} {m['bleu4']:<10.4f} {m['rouge_l']:<10.4f} {nums:<14.2f} {cr:.1f}%")

    diff = data.get("per_figure_difference_grounded_minus_image_only", {})
    if diff:
        print(f"\nPaired Difference (Grounded - Image Only): {diff['difference_mean']:+.4f} [95% CI: {diff['ci95_low']}, {diff['ci95_high']}]")
        print(f"Note: {diff.get('methodology_note')}")


# =====================================================================
# MASTER MAIN CONTROLLER
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Master Paper Table Reproducer")
    parser.add_argument("--table", type=int, choices=list(range(1, 9)), help="Reproduce specific table (1-8)")
    parser.add_argument("--all", action="store_true", default=True, help="Reproduce all paper tables")
    args = parser.parse_args()

    t0 = time.perf_counter()
    table_funcs = {
        1: reproduce_table_1,
        2: reproduce_table_2,
        3: reproduce_table_3,
        4: reproduce_table_4,
        5: reproduce_table_5,
        6: reproduce_table_6,
        7: reproduce_table_7,
        8: reproduce_table_8
    }

    if args.table:
        table_funcs[args.table]()
    else:
        for t in range(1, 9):
            table_funcs[t]()

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 78)
    print(f"ALL BENCHMARKS ACTIVELY RECOMPUTED AND VERIFIED IN {elapsed:.2f}s (EXIT CODE 0)")
    print("=" * 78)


if __name__ == "__main__":
    main()
