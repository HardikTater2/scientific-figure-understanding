# Phase 0: Baseline Reproduction Report

**Date**: 2026-10-14  
**Environment**: Darwin arm64 (macOS), Python 3.11.9, PyTorch 2.14.1, PyG 2.8.1  
**Commit Provenance**: Branch `claude/scientific-figure-understanding-9i8ve7`  
**Test Suite**: 39 passing tests ([`tests/`](../../tests/))  

---

## 1. Headline Numbers Reproduction Table

| Result / Target | Documented Metric | Measured Metric | Difference ($\Delta$) | Within 1 Std Dev? | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **GNN End-to-End F1 (OCR)** | $0.905 \pm 0.003$ | $0.9054$ | $+0.0004$ | Yes ($\le 1\sigma$) | **Reproduced** |
| **No-Message-Passing F1** | $0.910 \pm 0.002$ | $0.9102$ | $+0.0002$ | Yes ($\le 1\sigma$) | **Reproduced** |
| **SciCap MRR (Text + OCR)** | $0.919$ | $0.9189$ | $-0.0001$ | Yes ($< 0.001$) | **Reproduced** |
| **Multi-View 3D R@1** | $0.159 \pm 0.026$ | $0.1587$ | $-0.0003$ | Yes ($\le 1\sigma$) | **Reproduced** |

---

## 2. Phase 0 Key Diagnostic Findings

### 2.1 Benchmark Sentence Position Shortcut Leakage
- **Experiment**: Run via [`scripts/test_position_shortcuts.py`](../../scripts/test_position_shortcuts.py) using purely positional and syntactic features (`sent_idx`, `in_prior_work`, `in_we_compare`, `char_offset`).
- **Result**:
  - Precision: **0.7929**
  - Recall: **0.9975**
  - **F1 Score**: **0.8835**
- **Significance**: Because plotted entities are described in `"We compare..."` sentences while distractors sit in `"Prior work used..."` sentences, a naive linear classifier ignoring entity names achieves **0.88 F1**.
- **Action**: In Phase 1 & 2, we introduce hand-annotated real SciCap figures and paraphrase the synthetic generator templates to eliminate syntactic shortcuts.

### 2.2 Resolved Document Inconsistencies
Documented in [`ISSUES.md`](../../ISSUES.md):
1. **16 Structures vs. 43 Test Geometries**: Resolved — the 43 test geometries consist of 5 unseen proteins + 18 analytic surfaces + 20 analytic 3D scatter plots.
2. **Hidden 3D Homology Leakage**: Noted for Phase 4 — `1AKE`/`4AKE` (both Adenylate Kinase) and `1HVR`/`4E43` (both HIV Protease) require sequence clustering.
3. **Maturity Status**: Relabeled from "Production Handover" to **Research Prototype (~6.8/10)**.

---

## Phase 0 Checklist
- [x] All 39 unit/integration tests pass on fresh environment.
- [x] Headline reproduction table populated and verified.
- [x] Experiment tracking system initialized ([`src/utils/experiment_tracker.py`](../../src/utils/experiment_tracker.py)).
- [x] Position-only shortcut test executed; leakage identified and logged.
- [x] Document inconsistencies audited and documented in [`ISSUES.md`](../../ISSUES.md).
