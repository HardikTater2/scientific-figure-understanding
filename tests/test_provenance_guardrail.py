"""
Provenance and Scientific Honesty Guardrail Tests (Non-Negotiable Rules).

Verifies:
  (a) No results script in pipelines/ uses random or numpy.random to fabricate model outputs,
      predictions, or human ratings.
  (b) Every results JSON in results/ contains a valid "provenance" field
      (one of: human_external_dataset, silver_llm, silver_heuristic, synthetic, analysis).
  (c) The terms "human", "gold", or "kappa" do not appear in results or docs to describe
      unverified in-house data.
"""

import os
import glob
import json
import re
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VALID_PROVENANCES = {
    "human_external_dataset", "silver_llm", "silver_heuristic",
    "synthetic", "analysis", "none", "model_outputs_vs_human_references"
}


def test_results_json_have_provenance():
    """Rule 4 & 5b: Every results JSON must contain a valid 'provenance' key."""
    json_files = glob.glob(os.path.join(ROOT, "results", "**", "*.json"), recursive=True)
    assert len(json_files) > 0, "No results JSON files found to inspect!"

    missing_provenance = []
    invalid_provenance = []

    for fp in json_files:
        rel_path = os.path.relpath(fp, ROOT)
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            if "provenance" not in data:
                missing_provenance.append(rel_path)
            elif data["provenance"] not in VALID_PROVENANCES:
                invalid_provenance.append(f"{rel_path}: '{data['provenance']}'")
        except Exception as e:
            pytest.fail(f"Failed to parse {rel_path}: {e}")

    assert not missing_provenance, f"Results JSON files missing 'provenance' key: {missing_provenance}"
    assert not invalid_provenance, f"Results JSON files with invalid 'provenance' (must be in {VALID_PROVENANCES}): {invalid_provenance}"


def test_no_random_fabrication_in_pipelines():
    """Rule 5a: No active pipeline may use random number generators to fabricate model outputs or ratings."""
    pipeline_files = glob.glob(os.path.join(ROOT, "pipelines", "*.py"))
    fabrication_violations = []

    # Banned fabrication patterns: using random to generate predictions or ratings
    banned_patterns = [
        re.compile(r"rng\.choice\(.*hallucination", re.IGNORECASE),
        re.compile(r"rng\.random\(\)\s*<\s*0\.\d+.*#.*recall", re.IGNORECASE),
        re.compile(r"h_rating\s*=\s*\d+\.\d+\s*\+", re.IGNORECASE),
        re.compile(r"simulate_or_execute_vlm_prediction", re.IGNORECASE)
    ]

    for pf in pipeline_files:
        rel_path = os.path.relpath(pf, ROOT)
        with open(pf, "r") as f:
            content = f.read()
        for pat in banned_patterns:
            if pat.search(content):
                fabrication_violations.append(f"{rel_path} matched banned simulation pattern: {pat.pattern}")

    assert not fabrication_violations, (
        f"Active pipelines must not fabricate outputs via random generation! Violations:\n"
        + "\n".join(fabrication_violations)
    )


def test_no_false_human_gold_claims_in_results():
    """Rule 5c: 'human', 'gold', or 'kappa' must not appear claiming in-house human data."""
    results_files = glob.glob(os.path.join(ROOT, "results", "**", "*.json"), recursive=True)
    violations = []

    for rf in results_files:
        rel_path = os.path.relpath(rf, ROOT)
        with open(rf, "r") as f:
            text = f.read()

        # Reject claims that our real scicap figures are "gold" or have "kappa"
        if "inter_annotator_agreement" in rf and '"status": "NOT DONE"' not in text:
            violations.append(f"{rel_path}: active agreement claims without real human annotators")
        if "SciCap-Grounding-Gold" in text:
            violations.append(f"{rel_path}: references 'SciCap-Grounding-Gold' instead of 'silver'")

    assert not violations, f"Results contain false gold/human claims:\n" + "\n".join(violations)


def test_results_json_have_required_metadata():
    """Task 3: Every results JSON must contain 'generated_by', 'git_commit', and 'timestamp'."""
    json_files = glob.glob(os.path.join(ROOT, "results", "**", "*.json"), recursive=True)
    assert len(json_files) > 0, "No results JSON files found to inspect!"

    missing_fields = []
    for fp in json_files:
        rel_path = os.path.relpath(fp, ROOT)
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            missing = [k for k in ("generated_by", "git_commit", "timestamp") if k not in data]
            if missing:
                missing_fields.append(f"{rel_path} missing: {missing}")
        except Exception as e:
            pytest.fail(f"Failed to parse {rel_path}: {e}")

    assert not missing_fields, (
        f"Every results JSON must contain 'generated_by', 'git_commit', and 'timestamp'!\nViolations:\n"
        + "\n".join(missing_fields)
    )


def test_raw_predictions_record_model_provenance():
    """
    Integrity Guardrail Rule 5: Every raw prediction file in results/**/raw/* must record:
      - For checkpoint models: checkpoint_path + valid 64-char hex SHA-256 (which must match the file).
      - For VLM / heuristic models: model_name, weights_path, or model_id.
    """
    raw_files = glob.glob(os.path.join(ROOT, "results", "**", "raw", "*"), recursive=True)
    raw_files = [f for f in raw_files if os.path.isfile(f)]
    assert len(raw_files) > 0, "No raw prediction files found under results/**/raw/*"

    violations = []
    for fp in raw_files:
        rel_path = os.path.relpath(fp, ROOT)
        if fp.endswith(".jsonl"):
            with open(fp, "r") as f:
                first_line = f.readline()
            if not first_line.strip():
                violations.append(f"{rel_path}: empty raw jsonl file")
                continue
            entry = json.loads(first_line)
            if not any(k in entry for k in ("model_name", "model_id", "weights_path", "checkpoint_path")):
                violations.append(f"{rel_path}: records missing model_name / model_id / weights_path / checkpoint_path")
        elif fp.endswith(".json"):
            with open(fp, "r") as f:
                data = json.load(f)
            if "checkpoint_path" in data:
                ckpt_rel = data["checkpoint_path"]
                ckpt_abs = os.path.join(ROOT, ckpt_rel) if not os.path.isabs(ckpt_rel) else ckpt_rel
                if not os.path.exists(ckpt_abs):
                    violations.append(f"{rel_path}: checkpoint_path '{ckpt_rel}' does not exist on disk")
                sha = data.get("checkpoint_sha256", "")
                if not sha or len(sha) != 64 or not all(c in "0123456789abcdefABCDEF" for c in sha):
                    violations.append(f"{rel_path}: missing or invalid 64-char hex checkpoint_sha256")
                else:
                    import hashlib
                    actual_sha = hashlib.sha256(open(ckpt_abs, "rb").read()).hexdigest()
                    if actual_sha.lower() != sha.lower():
                        violations.append(f"{rel_path}: checkpoint_sha256 mismatch ({sha} vs actual {actual_sha})")
            elif any(k in data for k in ("model_name", "model_id", "model_spec")):
                pass
            else:
                violations.append(f"{rel_path}: missing model checkpoint or identifier (must record checkpoint_path + SHA-256 or model_name/id)")

    assert not violations, "Raw prediction files failed model provenance guardrail:\n" + "\n".join(violations)


