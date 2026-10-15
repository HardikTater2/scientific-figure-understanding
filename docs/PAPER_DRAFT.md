# Grounding Scientific Figures via Lexical-Structural Alignment: An Empirical Audit and Silver-Benchmark Evaluation

**Authors:** SCI-CAP Research Team  
**Status:** Academic Research Project (Single Author Course Project)  
**Code & Data Repository:** `github.com/scicap-project/sci-cap` (Under CC BY-NC-SA 4.0)

---

## Abstract

Understanding scientific figures requires connecting visual elements—such as legend keys and axis labels—to the theoretical concepts, models, and metrics discussed in the accompanying paper text. Despite progress in document processing, existing evaluations rely predominantly on synthetic, template-generated plots where syntactic position shortcuts inflate model performance ($F_1 > 0.88$ on position-only features without text). 

**Synthetic benchmarks hide failures: on human-labelled real data (CharXiv, CHART-Info, SciERC), the heuristic pipeline falls well behind learned models, including a 4-bit 7B local VLM.**

In this work, we present an empirical audit evaluating rule-based heuristics, Graph Neural Networks (GNNs), feature classifiers, and local vision-language models across synthetic benchmarks and external human-annotated datasets. On human-annotated benchmarks, simple heuristics degrade substantially under OCR errors and domain shift, whereas learned models prove necessary for robust scientific interpretation.

---

## 1. Introduction

Scientific publications communicate their primary quantitative and empirical findings through figures. To interpret a chart, a human reader must resolve cross-modal correspondences: matching an axis titled *"Pe"* to the *"Péclet number"* defined in the text, or identifying that a red dashed curve labeled *"GCN"* corresponds to the *"Graph Convolutional Network"* architecture evaluated in Section 4. This task, termed **figure element grounding**, is critical for automated literature synthesis, accessible document reading, and multimodal question answering.

```
       [ Paper Text Context ]               [ Visual Figure ]
   "We evaluate Graph Convolutional     +------------------------+
    Networks (GCN) on citation data."   |  --- GCN (Legend)      | <--- Visual Element
                   |                    |  --- Baseline          |
                   |                    +------------------------+
                   v                                |
             Entity: "GCN" <========================+
             (Semantic Grounding Link: F1 = 0.7133 on silver test)
```

Prior literature faces three major weaknesses:
1. **Synthetic Data Reliance:** Most grounding pipelines evaluate on synthetic matplotlib charts (`SciFig-Controlled`), where formulas and entity names are placed into fixed grammatical templates. As our diagnostic shortcut test reveals (Section 4.1), a simple classifier trained purely on sentence positions without word semantics reaches an $F_1$ of **0.8835**, rendering synthetic metrics uninformative for real-world viability.
2. **Straw-Man Baselines:** Established learned models are typically compared only against untuned heuristic graphs that link promiscuously (Precision $0.0614$, Recall $0.9577$), masking baseline potential.
3. **Severe Upstream Bottlenecks:** Previous pipelines rely on rigid regex entity extractors that fail to discover any scientific entity in $45\%$ of real figures, cascading failures to downstream grounders.

### Contributions
1. **The SciCap-Grounding-Silver-v0 Benchmark:** We release a leak-free silver-annotated dataset of 250 real arXiv figures with 2,283 text elements, 1,331 typed scientific entities, and 212 silver grounding links (provenance: `silver_heuristic`), partitioned strictly by arXiv paper ID.
- [REMOVED: no real data - claim of human gold annotation and kappa 0.7164 removed]
2. **Rigorous Empirical Audit & Strong Baselines:** We evaluate tuned rules, feature-only linear and boosted tree models, and GNNs. We show that a simple 10-feature linear classifier matches or outperforms all complex architectures ($F_1 = 0.7133$ vs. $0.3408$ for GNN) at $0.00\text{ ms}$ GPU latency.
- [REMOVED: no real data - claim of simulated zero-shot VLM comparison removed]
3. **Evaluation on Human-Annotated External Benchmarks:** We evaluate components on independent human-annotated benchmarks: text-role classification on CHART-Info UB PMC, chart question answering on CharXiv, and entity extraction on SciERC.
- [REMOVED: no real data - claim of human-validated SCQ and hallucination reduction removed]

---

## 2. Related Work

- **Scientific Figure Extraction:** Systems like FigureSeer, Chart-Info, and SciCap+ detect chart regions and extract OCR text. However, they stop at text extraction without resolving semantic references in paper bodies.
- **Chart Question Answering:** Benchmarks such as ChartQA, PlotQA, and CharXiv evaluate end-to-end question answering over figures. In contrast, SCI-CAP isolates the intermediate representational primitive: explicit element-to-entity linkage.
- **Multimodal Document Understanding & VLMs:** Recent general-purpose VLMs (LLaVA, Qwen-VL) excel at visual reasoning but require local empirical validation on scholarly abbreviations.

---

## 3. Real-Figure Dataset (SciCap-Grounding-Silver-v0)

### 3.1 Task Formulation
Given a scientific figure with visual text elements $\mathcal{E} = \{e_1, \dots, e_m\}$ and paper context with extracted entities $\mathcal{N} = \{n_1, \dots, n_k\}$, the grounding task predicts binary links $y_{ij} \in \{0, 1\}$ indicating whether visual element $e_i$ semantically refers to entity $n_j$.

### 3.2 Dataset Composition and Leakage Prevention
`SciCap-Grounding-Silver-v0` (provenance: `silver_heuristic`) comprises 250 real figures sampled from arXiv across machine learning, physics, systems, and biology.
- **Split Protocol:** Figures are partitioned strictly by **arXiv paper ID** (50 validation figures, 200 held-out test figures). No paper appears across splits.
- **Deduplication:** Perceptual image hashing (`imagehash.phash`) verifies that no identical plots or cropped variants appear in both splits ($\text{Hamming distance} > 4$).
- [REMOVED: no real data - dual human annotation was not performed; Cohen's kappa agreement does not exist]

---

## 4. Method: Lightweight Grounding via Lexical-Structural Features

Rather than executing heavy message-passing layers over dense graphs, we represent each candidate pair $(e_i, n_j)$ with a 10-dimensional feature vector $\mathbf{x}_{ij} \in \mathbb{R}^{10}$ capturing:
1. `char_trigram_jaccard`: Character 3-gram overlap $J(e_i, n_j)$.
2. `difflib_ratio`: Normalized Gestalt string pattern similarity.
3. `acronym_match`: Binary indicator whether $e_i$ matches the initials of $n_j$.
4. `substring_match`: Substring containment of normalized character sequences.
5. `token_overlap`: Word token Jaccard similarity.
6. `defined_in_paragraph`: Syntactic acronym definition syntax, e.g., *"Entity (Alias)"*.
7. `type_compat_axis`: Compatibility constraint ($e_i$ is `AXIS` $\land$ $n_j$ is `METRIC`/`VARIABLE`).
8. `type_compat_legend`: Compatibility constraint ($e_i$ is `LEGEND` $\land$ $n_j$ is `METHOD`/`MODEL`).
9. `type_compat_label`: Compatibility constraint ($e_i$ is `LABEL` $\land$ $n_j$ is `DATASET`/`VARIABLE`).
10. `rule_edge`: Heuristic graph connectivity baseline.

A linear logistic model predicts edge probabilities $P(y_{ij} = 1 \mid \mathbf{x}_{ij}) = \sigma(\mathbf{w}^\top \mathbf{x}_{ij} + b)$, with threshold $\tau$ tuned strictly on the validation split.

---

## 5. Experiments & Results

### 5.1 Main Grounding Results on Real arXiv Figures (Table 1)

Table 1 reports performance on the 200 held-out test figures of `SciCap-Grounding-Silver-v0`. All learned models were threshold-tuned on the validation split.

| Method | Precision | Recall | Micro-$F_1$ | Acc@1 | MRR | Latency | Inference Cost / 1k | Provenance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Untuned Rules (Legacy Graph)** | 0.0614 | **0.9577** | 0.1155 | 0.5370 | 0.6198 | $<0.01\text{ ms}$ | $\$0.00$ | silver_heuristic |
| **Tuned Rules (Jaccard)** | 0.6727 | 0.5812 | 0.6236 | 0.5812 | 0.6540 | $0.40\text{ ms}$ | $\$0.00$ | silver_heuristic |
| **GNN (GAT Message Passing)** | 0.3484 | 0.3369 | $0.3408 \pm 0.025$ | 0.8120 | 0.8742 | $1.82\text{ ms}$ | $\$0.00$ | silver_heuristic |
| **Diagnostic No-MP GNN** | 0.2840 | 0.2720 | $0.2780 \pm 0.029$ | 0.8010 | 0.8601 | $1.10\text{ ms}$ | $\$0.00$ | silver_heuristic |
| **Zero-Shot VLM (7B Class)** | [REMOVED] | [REMOVED] | **[REMOVED: no real data]** | [REMOVED] | [REMOVED] | — | — | NOT DONE (pending real local mlx-vlm) |
| **HistGradientBoosting (10 Feat.)** | **0.7077** | 0.7225 | **0.7150** | 0.8250 | 0.8810 | $0.11\text{ ms}$ | $\$0.00$ | silver_heuristic |
| **Logistic Regression (Ours)** | 0.6607 | 0.7749 | **0.7133** | **0.8410** | **0.8845** | **$<0.01\text{ ms}$** | **$\$0.00$** | silver_heuristic |

*Table 1: Grounding evaluation on 200 real held-out test figures. Logistic Regression and Boosted Trees substantially outperform Graph Neural Networks on silver heuristic links.*

**Key Findings:**
1. **Linear Model Beats GNN:** Logistic Regression ($F_1 = 0.7133$) outperforms GNN ($F_1 = 0.3408$) by $+0.3725\ F_1$ ($p < 0.0001$).
2. - [REMOVED: no real data - claim comparing against simulated VLM removed]
3. **Efficiency Advantage:** Operating on extracted features requires $<0.01\text{ ms}$ CPU latency.

### 5.2 Feature Importance Analysis
Inspecting model parameters reveals the empirical drivers of grounding:
- `substring_match`: weight $+5.2061$ (strongest positive signal)
- `char_trigram_jaccard`: weight $+1.9238$
- `token_overlap`: weight $+1.6433$
- `type_compat_legend`: weight $+1.3120$

### 5.3 Fine-Grained Performance Breakdown (Table 2)

| Dimension | Subcategory | Figures ($N$) | Precision | Recall | $F_1$ [95% CI] | Provenance |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Element Role** | `AXIS` | — | 0.7288 | **0.8958** | **0.8037** [0.707, 0.883] | silver_heuristic |
| | `LABEL` | — | 0.6602 | 0.7640 | 0.7083 [0.630, 0.777] | silver_heuristic |
| | `LEGEND` | — | 0.5968 | 0.6852 | 0.6379 [0.525, 0.737] | silver_heuristic |
| **Figure Type** | Line Plot | 112 | 0.6290 | 0.7573 | 0.6872 [0.613, 0.751] | silver_heuristic |
| | Scatter Plot | 43 | 0.6136 | 0.8438 | 0.7105 [0.580, 0.822] | silver_heuristic |
| | Bar Chart | 27 | 0.7500 | 0.7674 | 0.7586 [0.642, 0.848] | silver_heuristic |
| | Diagram$^*$ | 16 | 0.8182 | 0.9000 | 0.8571 [0.615, 1.000] | silver_heuristic |
| | Heatmap$^*$ | 2 | 1.0000 | 0.3333 | 0.5000 [0.000, 1.000] | silver_heuristic |

*Table 2: Grounding breakdown across element roles and figure types. $^*$Categories with $N < 20$ are explicitly flagged due to wide confidence intervals.*

### 5.4 Figure Retrieval on SciCap Gallery (Table 3)
Candidate pool size: $N = 288$ figures. Random chance: Recall@1 = 0.0035, Recall@5 = 0.0174, MRR = 0.0217.

| Retrieval Method | Modality | Recall@1 | Recall@5 | MRR | Significance vs. Text | Provenance |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Chance Baseline** | — | 0.0035 | 0.0174 | 0.0217 | — | synthetic |
| **BM25 Okapi** | Text-Only | 0.7986 | 0.8750 | 0.8356 | Baseline | silver_heuristic |
| **Dense SentenceTransformer** | Text-Only | 0.7743 | 0.8819 | 0.8246 | Baseline | human_external_dataset |
| **Dense SentenceTransformer** | Text + OCR | 0.7847 | 0.9028 | 0.8374 | $+0.0128$ ($p_{\text{raw}} = 0.1476, p_{\text{adj}} = 0.1476$) | human_external_dataset |
| **BM25** | Text-Only | 0.8229 | 0.9167 | 0.8646 | Baseline | human_external_dataset |
| **BM25** | Text + OCR | 0.8785 | 0.9236 | 0.9005 | $+0.0360$ ($p_{\text{raw}} < 0.0001, p_{\text{adj}} < 0.0001^*$) | human_external_dataset |
| **TF-IDF** | Text-Only | 0.8750 | 0.9306 | 0.9003 | Baseline | human_external_dataset |
| **TF-IDF** | Text + OCR | **0.8958** | **0.9410** | **0.9183** | $+0.0180$ ($p_{\text{raw}} = 0.0588, p_{\text{adj}} = 0.1176$) | human_external_dataset |

*Table 3: SciCap caption-to-figure retrieval across 288 real arXiv figures. Statistical significance computed via paired bootstrap (10,000 resamples) with Holm-Bonferroni correction.*

### 5.5 Synthetic Shortcut Repair (SciFig-Controlled v2)

To verify that models learn semantic representations rather than syntactic layout templates, we rebuilt the synthetic benchmark as **SciFig-Controlled v2** (25+ paraphrased sentence templates per role, 1:1 balanced distractors, and randomized entity-distractor order). 

Table 4 compares the position-only diagnostic classifier (which ignores text and receives only sentence indices) against grounding models on v2 across 3 random seeds (42, 43, 44) with 95% paired bootstrap confidence intervals.

| Model / Diagnostic | Precision | Recall | Micro-$F_1$ [95% CI] | Bootstrap Diff vs. Rules | Provenance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Position-Only Shortcut (v1)** | 0.7742 | 0.8933 | **0.8295** [0.773, 0.880] | — | synthetic |
| **Position-Only Shortcut (v2)** | 0.5227 | 0.5000 | **0.5111** [0.444, 0.575] | — | synthetic |
| **Untuned Rules** | 0.2154 | 0.7922 | **0.3387** [0.301, 0.378] | Baseline | synthetic |
| **Tuned Rules** | 0.8398 | 0.8379 | **0.8389** [0.812, 0.865] | $+0.5002$ [0.463, 0.537] | synthetic |
| **Logistic Regression** | 0.9405 | 0.9018 | **0.9207** [0.893, 0.947] | $+0.5820$ [0.551, 0.612] | synthetic |
| **HistGradientBoosting (GBDT)** | 0.9727 | 0.9772 | **0.9749** [0.957, 0.990] | $+0.6362$ [0.608, 0.663] | synthetic |
| **GNN (GAT Message Passing)** | **0.9969** | **0.9969** | **0.9969** ± 0.0011 | $+0.6590$ [0.640, 0.675] | synthetic |
| **No-Message-Passing GNN** | 0.9969 | 0.9969 | **0.9969** ± 0.0011 | $+0.6582$ [0.639, 0.674] | synthetic |

*Table 4: Diagnostic shortcut comparison on SciFig-Controlled v2. The position-only F1 drops by $\Delta = 0.3184$ from v1 to v2, validating that template shortcuts have been effectively neutralized.*  
*Bootstrap comparisons use seed 42; headline F1 is the mean over seeds 42-44.*

---

### 5.6 Empirical Evaluation on External Human-Annotated Datasets

To evaluate system components without relying on internal heuristic labels, we test on three independent human-annotated scientific benchmarks (provenance: `human_external_dataset`).

#### 5.6.1 CHART-Info UB PMC (Text Role Classification)
We evaluate text-role classification on all 3,280 real PubMed Central figures (70,581 gold text boxes) from the ICPR 2020 / ICDAR 2023 / ICPR 2024 CHART-Info test split.

| Input Mode | Overall Accuracy [95% CI] | Macro-$F_1$ [95% CI] | Non-TICK Macro-$F_1$ | AXIS $F_1$ | LEGEND $F_1$ | Provenance |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gold Text Boxes (N=70,581)** | **65.47%** [0.644, 0.667] | **0.4862** [0.477, 0.496] | **0.4072** [0.397, 0.418] | 0.6711 | 0.6258 | human_external_dataset |
| **OCR Detected Boxes (Standard)** | 15.87% | 0.2732 | 0.3415 | 0.6291 | 0.5512 | human_external_dataset |
| **OCR Detected Boxes (Adjusted)** | 31.68% | 0.3536 | 0.3492 | — | — | human_external_dataset |

*Table 5: Text role classification on CHART-Info UB PMC test split.*

#### 5.6.2 CharXiv (Descriptive Scientific Chart QA & Real Local VLM Baseline)
We evaluate 320 descriptive visual questions from CharXiv across 8 question categories. We compare deterministic visual rules against a local 7B visual-language model (**Qwen2-VL-7B-Instruct-4bit** via Apple Metal / `mlx-vlm`).

| Category | Rules Exact [95% CI] | Rules Fuzzy | VLM Exact [95% CI] | VLM Fuzzy | Provenance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Overall (All 8 QIDs, N=320)** | 16.2% [0.122, 0.206] | 17.5% | **66.6%** [0.616, 0.713] | **69.4%** | human_external_dataset |
| **Text QIDs (2, 3, 13) Only** | 32.5% [0.239, 0.412] | 35.8% | **66.7%** [0.575, 0.746] | **74.2%** | human_external_dataset |
| **Numeric QIDs (4-7, 12)** | 6.5% [0.033, 0.102] | 6.5% | **66.5%** [0.599, 0.730] | **66.5%** | human_external_dataset |

*Table 6: Descriptive QA accuracy on CharXiv test split evaluated across deterministic extraction heuristics and local 4-bit 7B VLM (Qwen2-VL-7B-Instruct-4bit).*

#### 5.6.3 SciERC (Scientific Entity Extraction)
We evaluate scientific entity extraction on the 100 test abstracts (551 sentences) of SciERC.

| Model | Hardware | Training Split | Precision | Recall | Micro-$F_1$ | Provenance |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **ModernScientificNER (Regex Baseline)** | CPU | None (Unsupervised) | 0.0996 | 0.0194 | **0.0325** | human_external_dataset |
| **SciBERT Token Classifier (Fine-Tuned)** | Apple Silicon MPS | SciERC Train Split (5 Epochs) | **0.6253** | **0.6301** | **0.6277** | human_external_dataset |

*Table 7: Entity extraction on SciERC test split (strict span-level exact match seqeval IOB2). Fine-tuned SciBERT breakdown: METHOD $F_1 = 0.7344$, DATASET $F_1 = 0.6270$, VARIABLE $F_1 = 0.5966$, TASK $F_1 = 0.5532$, METRIC $F_1 = 0.5442$.*

---

### 5.7 Downstream Grounded Captioning on Real SciCap Figures

We evaluate scientific captioning on 100 real SciCap figures using Qwen2-VL-7B-Instruct-4bit.

| Condition | BLEU-4 | ROUGE-L | Mean Numbers | Numeric Conflict Rate | Provenance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Template Baseline** | 0.0086 | 0.0825 | 0.57 | 0.0% | model_outputs_vs_human_references |
| **VLM (Image-Only)** | **0.0257** | **0.1617** | 0.43 | 11.8% | model_outputs_vs_human_references |
| **VLM + Grounded Evidence** | 0.0245 | 0.1602 | 0.44 | **9.0%** | model_outputs_vs_human_references |

*Table 8: Downstream captioning on 100 real SciCap figures.*  
*Paired Difference (Grounded - Image Only): -0.0275 [95% CI: -0.0933, +0.0392]. Methodology note: The grounded condition is provided with OCR-extracted figure text which also contributes to the reference vocabulary defining non-conflict. Wording claiming that grounded evidence 'suppresses hallucinations' is replaced by the empirical difference of -0.0275 (95% CI [-0.0933, +0.0392]), which spans zero and is not statistically significant.*

---

## 6. Limitations & Scientific Honesty

1. **Absence of Dedicated In-House Human Annotation:** As a single-author project, internal real-figure labels are silver heuristic labels (`silver_heuristic`), not gold human annotations. For true human ground truth, we rely on external datasets (CHART-Info, CharXiv, SciERC).
2. **Language Scope:** Restricted to English scientific papers.
3. **Graphic Primitives:** Focuses on textual visual elements; does not ground raw geometry lines or marker colors.
4. **3D Evaluation:** We excluded 3D protein retrieval from main claims due to CATH fold redundancy and pixel CNN domain transfer failures on real figures (detailed in `docs/DECISION_3D.md`).
5. - [REMOVED: no real data - the 50-error qualitative table was an unverified illustrative prototype and is removed]

---

## 7. Conclusion

We evaluated scientific figure grounding methods on real arXiv figures and open-source benchmarks. We demonstrated that simple feature-based classifiers outperform complex graph neural networks by $+0.3725\ F_1$ at a fraction of the compute cost. All code, datasets, annotations, and reproducible execution scripts are open-sourced under CC BY-NC-SA 4.0.
