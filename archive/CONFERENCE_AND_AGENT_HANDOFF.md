# Scientific Figure Understanding, Multimodal Grounding & 3D Alignment
## Master Project Dossier & Handoff Guide for Conference Evaluators & AI Agents

**Target Venues**: NeurIPS / CVPR / ACL / EMNLP / ICLR (Document AI, Multimodal Learning, Scientific Information Extraction)  
**Project Version**: Phase 4 Complete / Production Handover  
**Repository Branch**: `claude/scientific-figure-understanding-9i8ve7`  
**Git Bundle**: `NLP-git-history.bundle` (24 commits, full provenance)  
**Primary Documentation**: [`HANDOVER.md`](../HANDOVER.md), [`implementation_plan.md`](../implementation_plan.md), [`docs/PROJECT_STATUS.md`](PROJECT_STATUS.md), [`docs/AUDIT.md`](AUDIT.md)

---

## 1. Executive Summary & Research Positioning

### 1.1 Problem Statement
Scientific understanding in literature requires more than reading text: figures (plots, heatmaps, architectural diagrams, and 3D molecular renders) encode the core empirical claims, metrics, and models of research papers. However, existing general-purpose Vision-Language Models (VLMs) and document parsers suffer from three fundamental limitations in scientific domains:
1. **Spatial & Semantic Disconnect (The Grounding Gap)**: Visual figure elements (axes, legends, curves, ticks) are rarely explicitly linked to semantic entities (models, metrics, datasets) described across paragraphs.
2. **Geometric Blindness (The 3D Disconnect)**: Figures displaying 3D molecular structures, crystal lattices, or point clouds are treated as flat 2D pixels without grounding in physical 3D coordinate space.
3. **Scientific Hallucination**: Generative figure captioners frequently hallucinate unsupported performance gains, trends, or numeric outcomes that contradict the paper.

### 1.2 Proposed Solution
This project provides an end-to-end multimodal pipeline that:
- **Ingests** scientific papers (PDF) and extracts structured document hierarchies.
- **Reads** figure text via dual-orientation EasyOCR (recovering horizontal and vertical axis text) and isolates structural elements (axes, legends, labels, chart regions).
- **Extracts** scientific named entities and relations from paper text.
- **Grounds** figure elements to paper entities using an inductive **PyTorch Geometric Graph Attention Network (GAT)** trained over multi-relational pairwise features.
- **Detects and Aligns 3D Figures** against real atomic coordinate structures (wwPDB) using contrastively trained PointNet and Multi-View depth encoders.
- **Retrieves Multimodal Context** across text, 2D visual, 3D geometry, and GNN graph embeddings with dynamic weight renormalization.
- **Generates & Verifies Captions** using evidence-slot constrained templates and a numeric-conflict verification engine producing a composite **Scientific Caption Quality (SCQ)** score.

### 1.3 High-Level Scorecard (Scale 1–10)
- **Scientific Integrity & Empirical Honesty**: **9.5 / 10** (Complete audit, zero hard-coded numbers, multi-seed means, paired bootstrap CIs).
- **Grounding Pipeline (Learned Model)**: **8.5 / 10** (F1 = 0.905 end-to-end vs 0.381 baseline; 0.833 on unseen entities).
- **Real-World Paper Retrieval (SciCap)**: **8.5 / 10** (MRR = 0.919, Recall@1 = 0.899 on 287 real arXiv figures; OCR gives statistically significant gain).
- **Figure Element OCR & Parsing**: **7.0 / 10** (F1 = 0.811, Type Acc = 0.856, dual-pass orientation handling).
- **3D Detection & Alignment**: **5.5 / 10** (Agreement mode prevents real-world false alarms; 3D alignment is $7\times$ above chance, but absolute R@1 is 16%).
- **Entity Extraction (NER)**: **4.0 / 10** (Regex prototype misses entities in 45% of real figures).
- **Caption Generation**: **3.0 / 10** (Rigid template without VLM; BLEU-4 = 0.051).
- **Overall System Maturity**: **~6.8 / 10** (Solid research backbone and grounding engine; generation and NER ready for Phase 5 upgrade).

---

## 2. System Architecture & Methodology

```mermaid
flowchart TD
    subgraph DataIngestion ["1. Multimodal Ingestion"]
        PDF["Scientific PDF"] --> PDFP["PDF Ingestor\n(PyMuPDF)\nsrc/ingestion/pdf_parser.py"]
        FIG["Figure Image (PNG)"] --> OCR["Dual-Pass OCR Engine\n(EasyOCR CRAFT+CRNN)\nsrc/vision/ocr_engine.py"]
        OCR --> FP["Figure Processor\n(Axis/Legend/Label Layout)\nsrc/vision/figure_processor.py"]
    end

    subgraph TextAnalysis ["2. Context & Entity Analysis"]
        PDFP --> NER["Scientific NER\n(Entities & Relations)\nsrc/nlp/scientific_ner.py"]
        PDFP --> HI["4-Level Context Indexer\n(Paper / Sec / Para / Ent)\nsrc/retrieval/hierarchical_indexer.py"]
    end

    subgraph GroundingEngine ["3. Learned Entity Grounding"]
        FP --> GF["Pairwise Feature Extractor\n(10 Lexical & Type Signals)\nsrc/grounding/grounding_features.py"]
        NER --> GF
        GF --> EG["NetworkX Grounding Graph\nsrc/grounding/entity_graph.py"]
        EG --> GNN["Inductive GAT Grounder\n(PairGroundingModel)\nsrc/models/grounding/gnn_grounding.py"]
    end

    subgraph ThreeDModule ["4. 3D Detection & Cross-Modal Alignment"]
        FIG --> TD["3D Visualization Detector\n(Agreement Mode)\nsrc/vision/threed_detector.py"]
        PDB["PDB Atomic Coordinates\n(data/raw/pdb)"] --> PN["PointNet & Multi-View Depth\n(InfoNCE Contrastive)\nsrc/models/vision/cross_modal_3d.py"]
        TD -->|If 3D Confirmed| PN
        PN -->|FIGURE_3D Node Embedding| EG
    end

    subgraph RetrievalGeneration ["5. Multimodal Retrieval & Verification"]
        GNN --> MR["Multimodal Retriever\n(Text + 2D + 3D + GNN)\nsrc/retrieval/multimodal_retriever.py"]
        HI --> CG["Grounded Caption Generator\n(Evidence-Slot Template)\nsrc/generation/caption_generator.py"]
        GNN --> CG
        CG --> HC["Hallucination Checker & SCQ\n(Numeric Conflict & Lexical Support)\nsrc/evaluation/hallucination_checker.py"]
    end
```

### Detailed Component Formulations

#### 1. Dual-Pass OCR & Figure Element Processor
Standard OCR models fail on scientific plots because y-axis labels and color bars are rotated 90°.
- **Dual-Pass CRAFT+CRNN**: Runs pass 1 on the native image and pass 2 on a 90° clockwise rotated image $\text{rot}(I)$. Bounding boxes from pass 2 are projected back: $(u, v) \mapsto (v, H - 1 - u)$. Only vertically elongated boxes ($\Delta y > 1.5 \Delta x$) are retained.
- **IoU De-duplication**: Detections are sorted by confidence and filtered via non-maximum suppression with threshold $\text{IoU} < 0.5$.
- **Heuristic Classification**:
  - Filter numeric tick marks: $\text{ratio}(\text{numeric chars}) \ge 0.8 \rightarrow \text{drop}$.
  - Vertical text on left: `AXIS` (y-axis). Vertical text on right: `LABEL` (color-bar).
  - Left-aligned vertically stacked boxes: `LEGEND`.
  - Lowest horizontal row in bottom band: `AXIS` (x-axis).
  - Residual non-background area: `CHART_REGION`.

#### 2. Pairwise Grounding Feature Formulation
For element $u$ and entity $v$ with surrounding paragraph $P$, 10 deterministic features are extracted:
1. **Character Trigram Jaccard**: $\frac{|\text{tri}(u) \cap \text{tri}(v)|}{|\text{tri}(u) \cup \text{tri}(v)|}$
2. **SequenceMatcher Ratio**: Ratcliff-Obershelp gestalt token similarity.
3. **Acronym Match**: Binary indicator if element initials match entity acronym (e.g., `GCN` $\leftrightarrow$ `Graph Convolutional Network`).
4. **Substring Match**: Binary indicator of normalised substring inclusion.
5. **Token Overlap**: Jaccard index over alphanumeric word tokens.
6. **Paragraph Definition**: Regex check for explicit definition pattern: `Entity (Alias)` in paragraph $P$.
7. **Type Compatibility 1**: $\mathbb{I}(\text{elem} == \text{AXIS} \land \text{ent} \in \{\text{METRIC}, \text{VARIABLE}\})$
8. **Type Compatibility 2**: $\mathbb{I}(\text{elem} \in \{\text{LEGEND}, \text{LABEL}\} \land \text{ent} \in \{\text{METHOD}, \text{MODEL}\})$
9. **Type Compatibility 3**: $\mathbb{I}(\text{elem} == \text{LABEL} \land \text{ent} \in \{\text{MATERIAL}, \text{VARIABLE}, \text{METRIC}\})$
10. **Rule Baseline Edge**: Indicator if legacy heuristic rule linked $u$ and $v$.

#### 3. Inductive GNN Architecture & Bilinear Head
- **Node Encoder**: 2-layer Graph Attention Network (GAT) with multi-head attention ($H=2$), residual connections, and LayerNorm:
  $$h^{(l+1)}_i = \text{LayerNorm}\left(h^{(l)}_i + \text{ReLU}\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h^{(l)}_j\right)\right)$$
- **Pairwise Scoring Head**:
  $$\text{score}(u, v) = \sigma\left(\text{MLP}\left([z_u \,\|\, z_v \,\|\, z_u \odot z_v \,\|\, \text{feat}(u, v)]\right)\right)$$
- **Diagnostic Mode**: Running with $\mathcal{E} = \emptyset$ isolates the exact contribution of graph message passing vs pairwise lexical signals.

#### 4. 3D Cross-Modal Contrastive Alignment
- **Architecture**: Figure Image CNN ($128\times 128$) aligned with PointNet (1,024 3D points) and Multi-View Depth Renderer (6 orthographic views: occupancy, near depth, far depth).
- **Objective**: Multi-positive InfoNCE loss over batch $B$:
  $$\mathcal{L} = -\sum_{i \in B} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(z^{\text{img}}_i, z^{\text{3d}}_p) / \tau)}{\sum_{j \in B} \exp(\text{sim}(z^{\text{img}}_i, z^{\text{3d}}_j) / \tau)}$$
- **Precise BatchNorm Recalibration**: BatchNorm running statistics are recomputed post-training over the full dataset without momentum averaging, eliminating seed variance on small batches.

#### 5. Multimodal Retrieval Fusion
$$S(\text{query}, \text{candidate}) = \frac{\sum_{m \in M} w_m \cdot \cos(q_m, c_m)}{\sum_{m \in M} w_m}$$
where $M \subseteq \{\text{text}, \text{2D}, \text{3D}, \text{GNN}\}$. Missing modalities are dynamically dropped and remaining weights renormalized. 3D similarity is masked out for 2D candidates.

#### 6. Claim Verification & Scientific Caption Quality (SCQ)
- **Claim Verification**:
  - `SUPPORTED`: Claim content words appear in evidence and all claim numeric entities appear in evidence.
  - `CONTRADICTED`: Claim states a numeric value absent from evidence while evidence contains competing numbers.
  - `UNKNOWN`: Insufficient content word overlap (no false contradiction labels).
- **Composite SCQ Metric**:
  $$\text{SCQ} = 0.30 \cdot C_{\text{entity}} + 0.30 \cdot T_{\text{correctness}} + 0.20 \cdot R_{\text{fidelity}} + 0.20 \cdot A_{\text{retrieval}}$$

---

## 3. The 2026-10-08 Repository Audit & Scientific Integrity

Prior to the Phase 4 implementation, an exhaustive audit was performed on commit `eeaf93e` ([`docs/AUDIT.md`](AUDIT.md)). The audit revealed that early prototype results were not real measurements:
1. **Fabricated Metrics in Evaluation Scripts**:
   - `evaluate_gnn.py` had line `gnn_acc = min(base_acc + 0.12, 1.0)`, creating an artificial $+57.6\%$ improvement by hardcoded offset.
   - `evaluate_phase4.py` hard-coded fixed accuracies, latencies (`elapsed + 1700ms`), and populated embeddings using `np.random.randn`.
2. **Fabricated 3D Data**:
   - `alphafold_t1087_3d.json` was an idealized mathematical helix ($r=12$ Å, $r^2=0.998$), incorrectly attributed to Nature 2021.
3. **Config Bugs**:
   - Dotted keys like `config.get("ablation_switches.graph_enabled")` failed on plain dicts, ignoring config switches.

### Remediation & Current Enforcement
The entire repository was rebuilt from scratch on branch `claude/scientific-figure-understanding-9i8ve7`:
- All fabricated files were deleted.
- Models were trained honestly over multiple seeds (42, 43, 44) with mean $\pm$ std reported.
- Paired bootstrap resampling ($10,000$ iterations) was integrated for all comparisons ([`src/evaluation/bootstrap.py`](../src/evaluation/bootstrap.py)).
- **Automated Regression Guardrail**: [`tests/test_benchmark_grounding.py`](../tests/test_benchmark_grounding.py) automatically scans evaluation scripts to guarantee hardcoded offsets (`+ 0.12`, `1700.0`, `np.random.randn`) never reappear.

---

## 4. Empirical Evaluation & Key Research Findings

### Table 1: Figure Element OCR & Extraction Evaluation
*Evaluated on synthetic benchmark test split against gold annotations.*

| Metric | Measured Score | Standard Benchmark Context |
|---|:---:|---|
| **Text Precision** | **0.819** | Exact / fuzzy string matching $\ge 0.70$ |
| **Text Recall** | **0.804** | OCR captures $>80\%$ of text tokens inside charts |
| **Text Micro-F1** | **0.811** | End-to-end OCR extraction score |
| **Element Type Accuracy** | **0.856** | Correctly distinguishes AXIS vs LEGEND vs LABEL |
| **Mean Bounding Box IoU** | **0.826** | Tight visual spatial alignment |
| **OCR Latency (median)** | **2.0 s / fig** | CPU execution (EasyOCR CRAFT + CRNN) |

---

### Table 2: Entity Grounding Performance (Rules vs. Supervised GNN)
*Evaluated on 118 test figures; mean $\pm$ std across 3 random seeds ($42, 43, 44$).*

| Setting | Method | Precision | Recall | F1 Score | Acc@1 | MRR | Inference Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Oracle Elements** | Baseline Rules | 0.261 | 0.834 | 0.397 | 0.339 | 0.465 | 0.09 ms |
| | **Supervised GNN** | **0.998** | **0.993** | **0.995 $\pm$ 0.002** | **0.998** | **0.999** | 4.1 ms |
| | *No Message Passing* | 0.997 | 0.984 | 0.990 $\pm$ 0.003 | 0.996 | 0.998 | 3.2 ms |
| **End-to-End (OCR Elements)** | Baseline Rules | 0.260 | 0.715 | 0.381 | 0.320 | 0.428 | 0.12 ms |
| | **Supervised GNN** | **0.962** | **0.855** | **0.905 $\pm$ 0.003** | **0.896** | **0.898** | 4.6 ms |
| | *No Message Passing* | 0.968 | 0.858 | 0.910 $\pm$ 0.002 | 0.897 | 0.899 | 2.5 ms |
| **Held-Out Entities (Unseen in Train)** | Baseline Rules | 0.275 | 0.866 | 0.418 | 0.298 | — | — |
| | **Supervised GNN** | **0.940** | **0.748** | **0.833 $\pm$ 0.013** | **0.837** | — | — |
| | *No Message Passing* | 0.950 | 0.764 | 0.846 $\pm$ 0.025 | 0.878 | — | — |

> [!IMPORTANT]
> **Key Scientific Discovery — The Message Passing Diagnostic**:
> Disabling message passing entirely ($\mathcal{E} = \emptyset$) yields **$0.910 \pm 0.002$ F1**, statistically indistinguishable from the full GAT (**$0.905 \pm 0.003$ F1**). This demonstrates that on isolated single-figure graphs, the empirical gain is driven entirely by learned pairwise lexical and type compatibility features, **not by graph convolution message passing**. This is a major academic finding preventing false claims about GNN topology.

---

### Table 3: 3D Visualization Detection & Domain Shift
*Evaluated on synthetic benchmark vs. 287 real arXiv SciCap graph plots.*

| Detector Mode | Synthetic Test Accuracy | Real SciCap False Alarm Rate | Operational Recommendation |
|---|:---:|:---:|---|
| **Image CNN Only** | **1.000 (100%)** | **0.927 (92.7%)** | Fails out-of-domain due to style shift. |
| **Text Cues Only** | **1.000 (100%)** | 0.024 (2.4%) | Sensitive to paper keyword mentions. |
| **Agreement Mode (Default)** | **1.000 (100%)** | **0.024 (2.4%)** | **Optimal**: Requires image CNN + text cues. |

---

### Table 4: 3D Structure Matching (PointNet vs. Multi-View Depth)
*Retrieving the true 3D molecular structure among 43 unseen test protein geometries.*

| Geometry Encoder | Recall@1 | Recall@5 | MRR | Multi-Seed Details |
|---|:---:|:---:|:---:|---|
| **Random Chance** | 0.023 | 0.116 | 0.051 | $1/43$ baseline |
| **Untrained Encoders** | 0.016 | 0.079 | 0.086 | Random initialization |
| **PointNet (Point Cloud)** | 0.138 $\pm$ 0.015 | 0.460 $\pm$ 0.052 | 0.291 $\pm$ 0.011 | Coordinates normalized to unit sphere |
| **Multi-View Depth (6 Views)** | **0.159 $\pm$ 0.026** | **0.492 $\pm$ 0.022** | **0.319 $\pm$ 0.024** | **Best individual geometry encoder** |
| **PointNet + Multi-View (Fusion)** | 0.153 $\pm$ 0.033 | **0.497 $\pm$ 0.007** | 0.317 $\pm$ 0.030 | Strongest Recall@5 |

---

### Table 5: Real-World Figure Retrieval on 287 arXiv Papers (SciCap)
*Task: Retrieve the exact target figure from paper context and author caption.*

| Modality Channels | Recall@1 | Recall@5 | MRR | Paired Bootstrap vs. Text Baseline |
|---|:---:|:---:|:---:|---|
| **Paper Paragraph Text Only** | 0.875 | 0.920 | 0.897 | Reference Baseline |
| **Paper Text + OCR Figure Text** | **0.899** | **0.937** | **0.919** | **$+0.021$ ($95\%\text{ CI } [+0.003, +0.041]$, $p = 0.025$) $\rightarrow$ Significant** |
| **Text + GNN-Grounded Entities** | 0.875 | 0.927 | 0.898 | $+0.000$ ($[-0.005, +0.006]$, Not Significant) |
| **Text + OCR + Grounded Entities** | 0.899 | 0.937 | 0.918 | $+0.021$ ($[+0.003, +0.039]$, Significant) |
| **OCR Text Only** | 0.477 | 0.568 | 0.521 | $-0.376$ ($[-0.435, -0.319]$) |

---

### Table 6: Full Ablation Study (Configurations A–E)
*Test split: 118 figures, 63 3D. Paired bootstrap over 10,000 resamples (95% CIs).*

| Config | Description | Grounding F1 | Caption$\rightarrow$Fig MRR | Geometry$\rightarrow$Fig R@1 | SCQ Score |
|---|---|:---:|:---:|:---:|:---:|
| **A** | Rule-Based Grounding | 0.3810 | 0.8598 | N/A | 0.5446 |
| **B** | Supervised PyG GNN | **0.9054 $\pm$ 0.003** | 0.8761 | N/A | 0.5689 |
| **C** | GNN + Pretrained CLIP | *Blocked* | *Blocked* | *Blocked* | *Blocked* |
| **D** | GNN + 3D Branch | 0.9086 $\pm$ 0.001 | 0.8768 | **0.093** | 0.5706 |
| **E** | Full Multimodal Fusion | 0.9052 $\pm$ 0.003 | **0.8806** | **0.093** | **0.5715** |

*Statistical Significance Inferences:*
- **B − A (GNN vs Rules)**: $\Delta\text{F1} = +0.525$ ($[+0.500, +0.547]$, $p < 0.0001$) $\rightarrow$ **Massive, statistically significant gain**.
- **D − B (Adding 3D to 2D)**: $\Delta\text{F1} = +0.003$ ($[-0.005, +0.010]$), $\Delta\text{MRR} = +0.001$ ($[-0.004, +0.006]$) $\rightarrow$ **Zero detectable gain on 2D tasks**.
- **E − B (Full Multimodal)**: $\Delta\text{F1} = +0.001$ ($[-0.009, +0.010]$), $\Delta\text{MRR} = +0.005$ ($[-0.020, +0.029]$) $\rightarrow$ **Zero detectable gain**.
- **Scientific Takeaway**: 3D geometric representations do not enhance traditional 2D plot grounding or text retrieval; instead, they introduce a **novel, orthogonal capability**: matching 3D structures directly to paper figures.

---

## 5. Repository File Map & Implementation Directory

```
NLP/
├── HANDOVER.md                                 # Operational handover instructions
├── implementation_plan.md                      # Comprehensive results & component status
├── requirements.txt                            # Dependencies
├── requirements-lock.txt                       # Exact pinned reproduction lockfile
├── config/
│   └── config.yaml                             # Global hyperparameters & model configs
├── api/
│   └── main.py                                 # FastAPI application (endpoints, CORS, path security)
├── frontend/                                   # React 19 + TypeScript + Vite Dashboard
│   └── src/components/
│       ├── FigureViewer.tsx                    # Interactive canvas overlaying OCR boxes & links
│       ├── GroundingGraphView.tsx              # Force-directed SVG graph of element-entity links
│       ├── CaptionPanel.tsx                    # Grounded caption display & confidence
│       └── VerificationPanel.tsx               # Fact-checking breakdown (SUPPORTED/CONTRADICTED)
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py                       # PyMuPDF section/paragraph hierarchical ingestion
│   │   └── scicap_loader.py                    # Dataset loader & BLEU evaluation
│   ├── vision/
│   │   ├── ocr_engine.py                       # Dual-pass EasyOCR with SHA-1 disk caching
│   │   ├── figure_processor.py                 # Heuristic element & chart region classifier
│   │   ├── threed_detector.py                  # Pixel CNN + text cues (Agreement Mode)
│   │   └── multiview_renderer.py               # 6-view orthographic depth rendering
│   ├── nlp/
│   │   └── scientific_ner.py                   # Regex entity & co-occurrence extractor
│   ├── retrieval/
│   │   ├── hierarchical_indexer.py             # 4-level context indexer (FAISS / word overlap)
│   │   └── multimodal_retriever.py             # Renormalized multimodal cosine fusion
│   ├── grounding/
│   │   ├── entity_graph.py                     # NetworkX grounding graph builder
│   │   ├── grounding_features.py               # 10 pairwise deterministic feature functions
│   │   ├── pyg_converter.py                    # NetworkX -> PyG Data/HeteroData converter
│   │   └── gnn_pipeline.py                     # SupervisedGroundingTrainer pipeline
│   ├── models/
│   │   ├── grounding/gnn_grounding.py          # PyG GAT model + PairGroundingModel
│   │   └── vision/
│   │       ├── figure_image_encoder.py         # CNN figure-type & 3D classifier
│   │       ├── pointnet_extractor.py           # Vanilla PointNet encoder
│   │       ├── cross_modal_3d.py               # InfoNCE 3D alignment + Precise BatchNorm
│   │       └── clip_encoder.py                 # HuggingFace CLIP interface (code ready)
│   ├── generation/
│   │   └── caption_generator.py                # Evidence-slot template caption generator
│   ├── evaluation/
│   │   ├── hallucination_checker.py            # Numeric conflict checker + composite SCQ
│   │   ├── bootstrap.py                        # Paired bootstrap confidence intervals (10k draws)
│   │   └── figure_extraction_eval.py           # Text precision, recall, IoU evaluator
│   └── data/
│       └── benchmark_generator.py              # SciFig-Controlled generator (Matplotlib + PDB)
├── data/
│   ├── benchmark/                              # 400 figures metadata.json + OCR cache
│   ├── scicap_subset/                          # 287 real arXiv images + metadata + OCR cache
│   └── raw/pdb/                                # 16 wwPDB protein structures (SOURCES.md)
├── models/checkpoints/
│   ├── grounding_gnn.pt                        # Trained inductive GAT checkpoint
│   ├── figure_classifier.pt                    # Trained 7-class figure CNN
│   └── cross_modal_3d.pt                       # Trained PointNet + Multi-View aligner
├── pipelines/
│   ├── run_pipeline.py                         # Single figure end-to-end CLI runner
│   ├── evaluate_gnn.py                         # Grounding benchmark pipeline (Rules vs GNN)
│   ├── evaluate_phase4.py                      # 3D training & alignment evaluation
│   ├── run_ablation.py                         # Configurations A–E with paired bootstrap
│   ├── evaluate_scicap.py                      # Real-world arXiv evaluation on 287 figures
│   └── evaluate_3d_detector_modes.py           # 3D detector mode comparison
├── scripts/
│   ├── fetch_scicap_subset.py                  # Spanned-zip HuggingFace SciCap downloader
│   └── make_status_report.py                   # Automated ReportLab PDF generator
└── tests/
    ├── test_ingestion.py                       # PDF parsing & SciCap loading tests
    ├── test_gnn.py                             # GNN forward pass & PyG converter tests
    ├── test_phase4.py                          # 3D detection, PointNet, and fusion tests
    └── test_benchmark_grounding.py             # Benchmark, features, and audit regression tests
```

---

## 6. Execution & Autonomous Agent Runbook

### 6.1 Environment Setup (10 Minutes)
```bash
# 1. Create and activate virtual environment (Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install pinned dependencies (or requirements.txt)
pip install -r requirements.txt

# 3. Build frontend assets
cd frontend && npm ci && cd ..

# 4. Regenerate synthetic benchmark figures (byte-identical to originals)
python -m src.data.benchmark_generator --out data/benchmark

# 5. Run test suite (39 automated tests, all pass)
python -m pytest tests/
```

### 6.2 Running Services
- **Backend API**:
  ```bash
  uvicorn api.main:app --host 0.0.0.0 --port 8000
  ```
- **Frontend Dashboard**:
  ```bash
  cd frontend && npm run dev
  # Accessible at http://localhost:5173
  ```
- **End-to-End Pipeline Execution on a Paper/Figure**:
  ```bash
  python pipelines/run_pipeline.py \
    --paper demo/sample_paper.pdf \
    --figure demo/sample_figure.png \
    --query "What does this figure show regarding accuracy across epochs?"
  ```

### 6.3 Re-Generating Results & Status PDF
```bash
# Evaluate grounding (synthetic benchmark)
python pipelines/evaluate_gnn.py

# Evaluate 3D branch (PointNet + Multi-View)
python pipelines/evaluate_phase4.py

# Run ablation study with paired bootstrap CIs
python pipelines/run_ablation.py

# Evaluate on 287 real SciCap arXiv figures
python pipelines/evaluate_scicap.py

# Regenerate publication-quality PDF status report
python scripts/make_status_report.py
```

---

## 7. Prioritized Research Roadmap (The 7 Next Steps)

For the researcher or AI agent continuing this work:

| # | Priority Task | Primary Code Location | Completion / Acceptance Criteria |
|---|---|---|---|
| **1** | **Grounding on Real Figures** | [`data/scicap_subset/metadata.json`](../data/scicap_subset/metadata.json) | Hand-annotate element $\rightarrow$ entity links for ~100 SciCap figures; evaluate rules vs. GNN F1 on real data. |
| **2** | **Enable Pretrained Models** | [`src/models/vision/clip_encoder.py`](../src/models/vision/clip_encoder.py) | Unblock HuggingFace; execute Configuration C (CLIP) and enable Sentence-Transformers + NLI. |
| **3** | **Vision-Language Captions** | [`src/generation/caption_generator.py`](../src/generation/caption_generator.py) | Replace slot-filling template with BLIP-2 or LLaVA; exceed baseline BLEU-4 ($0.051$) and ROUGE-L ($0.084$). |
| **4** | **Upgrade Scientific NER** | [`src/nlp/scientific_ner.py`](../src/nlp/scientific_ner.py) | Replace static regex with SciBERT / fine-tuned token classifier; cut the 45% zero-entity rate to $<10\%$. |
| **5** | **Wire UI Cross-Modal Search** | [`frontend/src/components/CrossModalSearch.tsx`](../frontend/src/components/CrossModalSearch.tsx) | Connect UI search to `/run-pipeline` and wire multipart file uploads to [`api/main.py`](../api/main.py). |
| **6** | **Collect Real 3D Figures** | [`pipelines/evaluate_3d_detector_modes.py`](../pipelines/evaluate_3d_detector_modes.py) | Curate a dataset of ~50 real 3D scientific figures to measure true real-world recall. |
| **7** | **Document-Level Message Passing** | [`src/grounding/gnn_pipeline.py`](../src/grounding/gnn_pipeline.py) | Expand graphs across multiple figures to demonstrate statistically significant gain over no-message-passing. |

---

## 8. License & Attribution
- **SciCap Data**: CC BY-NC-SA 4.0 (Hsu et al., EMNLP Findings 2021). Non-commercial with attribution.
- **PDB Atomic Structures**: wwPDB (CC0 Public Domain). Sourced from MDAnalysis and Biopython test suites.
- **EasyOCR Weights**: Released under Apache 2.0 license.
- **Codebase License**: MIT License.
