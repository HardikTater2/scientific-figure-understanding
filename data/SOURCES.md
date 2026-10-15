# Open-Source Datasets & Data Governance Plan

This document establishes the dataset inventory, licensing terms, leakage prevention policies, and acquisition roadmap for the SCI-CAP upgrade from prototype to research paper.

---

## 1. Current Repository Datasets

| Dataset | Path | Source & Citation | License | Role in Project | Leakage Safeguard |
|---|---|---|---|---|---|
| **SciFig-Controlled** | `data/benchmark/` | Procedural Matplotlib Generator (`src/data/benchmark_generator.py`) | MIT (Generated) | Controlled ablations, component validation | Aliases, distractors, unlinked axes |
| **SciCap Subset** | `data/scicap_subset/` | CrowdAILab / Hsu et al. (EMNLP Findings 2021) | CC BY-NC-SA 4.0 | Real arXiv pilot: OCR coverage, search, detector FP | `strip_caption()` leakage filter |
| **PDB Structures** | `data/raw/pdb/` | wwPDB via MDAnalysis & Biopython test suites | CC0 1.0 (Public Domain) | 3D molecular coordinates ($C_\alpha$ traces) | Structure-disjoint split by PDB ID |
| **Demo Samples** | `demo/dataset/` | Synthetic placeholders | Internal Mock | Backend API and frontend UI display only | **Excluded from all evaluations** |

---

## 2. Open-Source Datasets to Add

| Dataset | Primary Contents | Targeted Role in SCI-CAP | License / Access Terms | Integration Mechanism |
|---|---|---|---|---|
| **[CHART-Info UB PMC](https://tc11.cvc.uab.es/datasets/ICPR2020-CHART-Info_1)** | Real PubMed Central charts annotated for text bounding boxes, roles (`AXIS_TITLE`, `LEGEND_LABEL`, `TICK_LABEL`), and axes | Real-world benchmark for element role classifier (`AXIS` / `LEGEND` / `LABEL`), replacing synthetic-only type accuracy | Academic Use (ICPR 2020 / Adobe Research) | Mapped via `src/evaluation/chart_info_eval.py` |
| **[SciCap+](https://github.com/ZhishenYang/scientific_figure_captioning_dataset)** | ~414k SciCap figures with mention paragraphs and pre-extracted OCR tokens + bboxes (Yang et al., 2023) | Pre-filling candidate element boxes for real annotators; scaling figure retrieval from 287 to 5,000 figures | CC BY-NC-SA 4.0 (Inherits SciCap terms) | Candidate box caching & retrieval index |
| **[SPIQA](https://huggingface.co/datasets/google/spiqa)** | 270k QA pairs on computer science figures and tables across train, val, and 3 test splits (Google Research, 2024) | Downstream evaluation: does grounded context improve figure-grounded question answering? | Apache 2.0 / Dataset Card Terms | Evaluated in downstream QA pilot |
| **[CharXiv](https://charxiv.github.io/)** | 2,323 real arXiv charts across 8 subjects with descriptive and reasoning questions (Wang et al., 2024) | Descriptive question stress test: tests whether extracted figure elements and metrics match human questions | Research / Evaluation Only (Never train on it) | Held-out evaluation benchmark |
| **[SciERC](https://nlp.cs.washington.edu/sciIE)** | 500 AI paper abstracts annotated with 6 entity types (`Task`, `Method`, `Metric`, `Material`, `Other`, `Generic`) & relations | Training and evaluating modernized scientific NER to replace regex | Open Research Release (Univ. of Washington) | Entity type mapping & fine-tuning |
| **[SciER](https://arxiv.org/pdf/2410.21155)** | ~24.5k full-text entity annotations of datasets, methods, and tasks | Full-paper paragraph NER model fine-tuning | Public Research Release | Contextual token classifier |
| **[CATH Non-Redundant Sets](https://cathdb.info/download)** | Protein domains filtered by sequence identity ($S20$ and $S40$ non-redundant sets, $\le 20-40\%$ sequence identity) | Truly leak-free 3D cross-modal alignment splits across 300+ non-redundant domains | Academic / Non-commercial (CATH Database) | GroupShuffleSplit on CATH Superfamily |

---

## 3. Strict Leakage Prevention & Data Hygiene Rules

1. **Split by Paper, Never by Figure**:
   - SciCap splits at the figure level, which allows figures from the same paper to appear in both train and test.
   - For all multi-figure datasets, splits must be grouped strictly by **arXiv paper ID** or **PMC ID**.
2. **Cross-Dataset Perceptual De-duplication**:
   - Overlaps between SciCap, SciCap+, SPIQA, and CharXiv are checked via perceptual image hashing (`imagehash.phash(img)` with Hamming distance threshold $\le 4$).
   - Any figure overlapping with test splits is removed from training sets.
3. **Protein Sequence-Cluster Disjointness**:
   - Protein structures must be split by sequence cluster or CATH superfamily, ensuring homologous folds (e.g. `1HVR` and `4E43`, or `1AKE` and `4AKE`) never span both train and test.
4. **Untouched Test Policy**:
   - CharXiv and the 250-figure hand-annotated SciCap test set are strictly test-only and never accessed during model tuning or feature design.
