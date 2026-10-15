"""
Mutation Test: Verify that scripts/reproduce_all.py detects tampering.

For each table (1 through 8):
1. Copies the table's results JSON to a temporary directory.
2. Perturbs one headline metric by +0.05.
3. Runs the reproduction verifier against the mutated directory.
4. Asserts that the verifier exits with a non-zero status code.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def perturb_table_1(data):
    # Mutate F1 of the 5th method (e.g. Logistic Regression or GNN) by +0.05
    data["methods_table"][4]["f1"] += 0.05


def perturb_table_2(data):
    # Mutate F1 of AXIS role by +0.05
    data["breakdown_by_element_role"]["AXIS"]["f1"] += 0.05


def perturb_table_3(data):
    # Mutate MRR of TF-IDF text only by +0.05
    data["baselines"]["tfidf_text_only"]["mrr"] += 0.05


def perturb_table_4(data):
    # Mutate GNN F1 mean by +0.05
    data["models"]["gnn"]["f1_mean"] += 0.05


def perturb_table_5(data):
    # Mutate CHART-Info gold boxes accuracy by +0.05
    data["gold_boxes_evaluation"]["accuracy"] += 0.05


def perturb_table_6(data):
    # Mutate CharXiv VLM overall exact accuracy by +0.05
    data["overall_all_8_qids"]["vlm"]["exact"] += 0.05


def perturb_table_7(data):
    # Mutate SciERC SciBERT test F1 by +0.05
    data["results"]["finetuned_scibert_token_classifier"]["test_f1"] += 0.05


def perturb_table_8(data):
    # Mutate captioning VLM image-only BLEU-4 by +0.05
    data["conditions"]["vlm_image_only"]["bleu4"] += 0.05


TABLE_MUTATIONS = {
    1: ("appendix_circular/baselines_summary.json", perturb_table_1),
    2: ("appendix_circular/phase6_finegrained_breakdown.json", perturb_table_2),
    3: ("phase2/retrieval_baselines.json", perturb_table_3),
    4: ("phase2/synthetic_v2_evaluation.json", perturb_table_4),
    5: ("phase3/chart_info_evaluation.json", perturb_table_5),
    6: ("phase3/charxiv_evaluation.json", perturb_table_6),
    7: ("phase3/scierc_ner_evaluation.json", perturb_table_7),
    8: ("phase5_captioning_comparison.json", perturb_table_8),
}


@pytest.mark.parametrize("table_num", sorted(TABLE_MUTATIONS.keys()))
def test_reproduce_detects_tampering(table_num):
    rel_path, mutator = TABLE_MUTATIONS[table_num]
    src_file = os.path.join(ROOT, "results", rel_path)
    assert os.path.exists(src_file), f"Source results file not found: {src_file}"

    with tempfile.TemporaryDirectory() as tmpdir:
        dst_file = os.path.join(tmpdir, rel_path)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)

        # Mutate the metric by +0.05
        with open(dst_file, "r") as f:
            data = json.load(f)
        mutator(data)
        with open(dst_file, "w") as f:
            json.dump(data, f, indent=2)

        # Run verifier pointing to tmpdir
        env = os.environ.copy()
        env["RESULTS_DIR"] = tmpdir
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "reproduce_all.py"), "--table", str(table_num)]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)

        assert res.returncode != 0, (
            f"Verifier failed to detect +0.05 tampering for Table {table_num}!\n"
            f"Stdout:\n{res.stdout}\nStderr:\n{res.stderr}"
        )
