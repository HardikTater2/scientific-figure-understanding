# Scientific Figure Understanding & Entity Grounding

A reproducible NLP & computer vision research repository investigating cross-modal scientific document understanding: grounding textual entity mentions in academic papers to visual chart elements (axis titles, legend labels, tick numbers), caption-to-figure retrieval, downstream grounded captioning, and diagnostic audits of syntactic template shortcuts.

---

## 1. What the Project Does

Modern scientific documents communicate key findings through figures (e.g., plots, charts, and diagrams) accompanied by explanatory paragraphs and captions. This project investigates:
1. **Visual Text-Role Classification:** Detecting and classifying text boxes within chart images into functional roles (`AXIS`, `LEGEND`, `LABEL`, `OTHER`, `TICK`).
2. **Text-to-Figure Entity Grounding:** Linking extracted scientific entities from paper paragraphs to corresponding visual chart elements using rule-based heuristics, linear classifiers, and Graph Neural Networks (GAT).
3. **Diagnostic Shortcut Analysis:** Identifying positional layout shortcuts in synthetic benchmark templates and introducing controlled benchmarks (`SciFig-Controlled v2`) where positional shortcuts are neutralized.
4. **Caption-to-Figure Retrieval:** Evaluating TF-IDF, BM25, and dense embeddings on real arXiv figures, quantifying the retrieval gain from OCR-extracted figure text.
5. **Human External Benchmarking:** Validating components on independent human-annotated academic datasets: **CHART-Info UB PMC** (chart text-role classification), **CharXiv** (descriptive visual QA with a local 4-bit 7B VLM), and **SciERC** (fine-tuned SciBERT scientific entity extraction).
6. **Downstream Captioning Evaluation:** Assessing hallucination and numeric conflict rates when providing grounded evidence to vision-language models on real scientific figures.

---

## 2. Installation & Setup

### Environment Requirements
- Python 3.11+
- macOS (Apple Silicon MPS supported) or Linux / CPU

```bash
# Clone the repository
git clone https://github.com/gayatriyadavv/NLP.git
cd NLP

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Reproduction Command

Every table, comparison, confidence interval, and metric across the paper is dynamically computed and verified against raw model outputs and gold labels with a single command:

```bash
# Run full master verification across all 8 paper tables
python scripts/reproduce_all.py --all

# Or verify any individual table (1 through 8):
python scripts/reproduce_all.py --table 4
```

The reproducer actively recomputes metrics from raw per-figure prediction files and exits with a non-zero status code if any recomputed value differs from the reported results by more than $10^{-4}$.

---

## 4. Final Headline Results

All numbers below are copied directly from the output of `python scripts/reproduce_all.py --all`:

| Table | Benchmark / Task | Metric / Model | Value [95% CI] | Provenance |
| :--- | :--- | :--- | :--- | :--- |
| **App. A1** | Circular Silver-v0 Grounding ($N=200$) | Untuned Rules $F_1$<br>Tuned Rules $F_1$<br>Logistic Regression (10 Feat.) $F_1$<br>GNN (GAT) $F_1$<br>HistGradientBoosting $F_1$ | 0.1155<br>0.6236<br>**0.7133**<br>0.3408<br>0.7150 | `silver_heuristic` *(Quarantined: circular)* |
| **Table 3** | SciCap Figure Retrieval ($N=288$) | TF-IDF Text-Only MRR<br>TF-IDF Text+OCR MRR<br>BM25 Text-Only MRR<br>BM25 Text+OCR MRR<br>Dense MiniLM Text-Only MRR<br>Dense MiniLM Text+OCR MRR | 0.9003<br>**0.9183** (+0.0180 MRR, $p=0.059$)<br>0.8646<br>0.9005 (+0.0360 MRR, $p<0.0001^*$)<br>0.8246<br>0.8374 (+0.0128 MRR, $p=0.148$) | `human_external_dataset` |
| **Table 4** | Shortcut-Free Benchmark (`SciFig-Controlled v2`, $N=95$) | Position-Only Shortcut (v1)<br>Position-Only Shortcut (v2)<br>Untuned Rules Micro-$F_1$<br>Tuned Rules Micro-$F_1$<br>Logistic Regression Micro-$F_1$<br>HistGradientBoosting Micro-$F_1$<br>GNN (GAT Message Passing) Micro-$F_1$<br>No-Message-Passing GNN Micro-$F_1$ | 0.8295 (template shortcut)<br>0.5111 (shortcut neutralized)<br>0.3387<br>0.8389<br>0.9207<br>0.9749<br>**0.9969 ± 0.0011**<br>0.9969 ± 0.0011 | `synthetic` *(Footnote: Bootstrap comparisons use seed 42; headline F1 is the mean over seeds 42-44)* |
| **Table 5** | CHART-Info UB PMC Text Roles ($N=3,280$ charts, 70,581 boxes) | Gold Boxes Overall Accuracy<br>Gold Boxes Macro-$F_1$<br>Gold Boxes Non-TICK Macro-$F_1$<br>OCR Standard Accuracy / Macro-$F_1$<br>OCR Adjusted Accuracy / Macro-$F_1$ | **0.6547** [0.6437, 0.6668]<br>**0.4862** [0.4768, 0.4963]<br>0.4072 [0.3969, 0.4183]<br>0.1587 / 0.2732<br>0.3168 / 0.3536 | `human_external_dataset` |
| **Table 6** | CharXiv Descriptive QA ($N=320$ questions) | Overall Visual Rules Exact / Fuzzy<br>Overall Local 7B VLM Exact / Fuzzy<br>Text QIDs (2, 3, 13) Rules / VLM<br>Numeric QIDs (4-7, 12) Rules / VLM | 16.2% / 17.5%<br>**66.6%** [0.616, 0.713] / **69.4%**<br>32.5% / 66.7%<br>6.5% / 66.5% | `human_external_dataset` |
| **Table 7** | SciERC Entity Extraction ($N=551$ test sentences) | ModernScientificNER (Regex Baseline)<br>Fine-Tuned SciBERT (allenai/scibert) | 0.0325 (P=0.0996, R=0.0194)<br>**0.6277** (P=0.6253, R=0.6301) | `human_external_dataset` |
| **Table 8** | Downstream Captioning ($N=100$ real SciCap figures) | Template Baseline BLEU-4 / Conflict<br>Local VLM Image-Only BLEU-4 / Conflict<br>Local VLM + Grounded Evidence | 0.0086 / 0.0%<br>**0.0257** / 11.8%<br>0.0245 / **9.0%** | `model_outputs_vs_human_references` *(Diff: -0.0275 [-0.0933, +0.0392], not sig)* |

---

## 5. Limitations & Scientific Honesty

1. **No In-House Human Annotation:**  
   As a single-author project, no internal human annotation campaign was conducted. All heuristic-labeled datasets (`SciCap-Grounding-Silver-v0`) are explicitly quarantined in `results/appendix_circular/` due to heuristic-on-heuristic evaluation circularity. Primary empirical validation relies entirely on external human-annotated test splits (CHART-Info, CharXiv, SciERC).
2. **Silver Labels Unreliable ($\kappa = 0.0168$):**  
   Dual-model annotation attempting to create a silver test set (`SciCap-Grounding-Silver-v1`) between Qwen2.5-1.5B (text LLM) and Qwen2-VL-7B (VLM) yielded an inter-model agreement Cohen's kappa of only $\kappa = 0.0168$. Because 0 consensus positive links emerged on the test set, silver v1 was formally flagged as `INSUFFICIENT LABELS`, and model ranking claims were discarded.
3. **SCQ Metric Unvalidated:**  
   The proposed Scientific Caption Quality (SCQ) composite metric remains unvalidated as no dedicated human correlation study was conducted. It is cataloged as `validation_status: "NOT DONE"` and excluded from empirical claims.
4. **3D Reconstruction / Retrieval is Future Work:**  
   3D geometric cross-modal retrieval is excluded from main empirical claims. Evaluation on real paper figures revealed a 92.7% false-positive rate for image-only 3D detection on 2D plots. Grounded 3D understanding remains future work.
5. **Single Author Scope:**  
   The study was executed by a single author. All results are audited to prevent subjective bias, with raw prediction files, seeds, and strict regression tests committed.
