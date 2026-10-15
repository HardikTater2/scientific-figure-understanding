# SCI-CAP Research Paper Core Story & Architectural Scope (Updated / Honest Protocol)

---

## 1. Chosen Paper Strategy: Empirical Audit & External Human Benchmark Reality

> *"We conduct an empirical audit of scientific figure element grounding, demonstrating that synthetic benchmarks hide failures: on human-labelled real data (CharXiv, CHART-Info, SciERC), the heuristic pipeline falls well behind learned models, including a 4-bit 7B local VLM."*

---

## 2. The One-Sentence Claim

> **"Synthetic benchmarks hide failures: on human-labelled real data (CharXiv, CHART-Info, SciERC), the heuristic pipeline falls well behind learned models, including a 4-bit 7B local VLM."**

---

## 3. Structural Mapping: Main Paper vs. Appendix vs. Quarantined Components

| Component | Destination | Rationale |
| :--- | :--- | :--- |
| **Synthetic Position Shortcut Audit** | **Main Paper** (Sec. 1 & 4) | Critical empirical finding: exposes position shortcuts ($F_1 = 0.8835$) in synthetic benchmark templates. |
| **SciCap-Grounding-Silver-v0** | **Main Paper** (Sec. 3) | 250 real arXiv figures partitioned strictly by paper arXiv ID with transparent `silver_heuristic` provenance. |
| **Baseline Evaluation (Tuned Rules, Classifiers, GNN)** | **Main Paper** (Sec. 5) | Tuned rules, Logistic Regression, HistGradientBoosting, and GNN message passing on the silver test set. |
| **External Human Benchmark Evaluation** | **Main Paper** (Sec. 5) | Text-role classification on CHART-Info UB PMC, chart QA on CharXiv, and entity extraction on SciERC. |
| **Feature Importance Analysis** | **Main Paper** (Sec. 5) | Identifies lexical drivers: substring match (+5.206), char-trigram Jaccard (+1.924), token overlap (+1.643). |
| **Multiple Comparisons Correction** | **Main Paper** (Sec. 5) | Holm-Bonferroni step-down correction applied across all testing families. |
| **In-House Human Annotation & Cohen's Kappa** | **QUARANTINED** | **[REMOVED: no real data]** No dual human annotation was performed. |
| **Simulated Zero-Shot VLM Baseline** | **QUARANTINED** | **[REMOVED: no real data]** Simulation removed; replaced with actual local `mlx-vlm` execution. |
| **Simulated SCQ Human Validation** | **QUARANTINED** | **[REMOVED: no real data]** No human ratings collected. |
| **Simulated 50-Error Qualitative Breakdown** | **QUARANTINED** | **[REMOVED: no real data]** Real error analysis derived from objective programmatic errors on external benchmarks. |
| **3D Protein Cross-Modal Matching** | **Future Work / Dropped** | CATH fold leakage and lack of verified real 3D paper figures; cut to preserve scientific integrity. |
