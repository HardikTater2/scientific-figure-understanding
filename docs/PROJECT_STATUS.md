# Project Status — What Is Done and What Is Not

Last updated: 2026-10-14 · Branch `claude/scientific-figure-understanding-9i8ve7`

A plain-language summary of where the project stands. Every number comes from a file in
`results/` produced by a script in `pipelines/`; detailed tables are in
[`implementation_plan.md`](../implementation_plan.md). Numbers reported before the
2026-10-08 audit ([`AUDIT.md`](AUDIT.md)) were hard-coded and must not be cited.

**Legend:** ✅ done and measured · 🟡 works but limited / prototype · ❌ not done · ⛔ blocked by the environment

---

## 1. One-paragraph summary

The full pipeline runs end to end — PDF/figure in, OCR'd figure elements, scientific entities,
grounding graph, GNN, 3D branch, retrieval, caption, verification, SCQ — through a script, a
FastAPI backend and a React UI, with 39 passing tests and CI on GitHub. It has been evaluated on a
400-figure controlled benchmark (with real protein structures for the 3D figures) and on **287 real
arXiv figures from SciCap**. On real figures, text retrieval works well (MRR 0.90) and OCR text adds
a small but significant gain; the generated captions are poor (template-based), entity extraction
misses many entities, and grounding accuracy on real figures **cannot be measured yet** because no
labelled data exists. The 3D branch works only on synthetic data; its image classifier fails on
real figures, so 3D is now only declared when image and text agree.

---

## 2. What is done

| Area | Status | Evidence / key numbers |
|---|---|---|
| Repository audit, fabricated results removed | ✅ | `docs/AUDIT.md`; regression tests forbid hard-coded metric offsets |
| Controlled benchmark (SciFig-Controlled) | ✅ | 400 figures, exact gold labels; 3D molecules from 16 real PDB structures, structure-disjoint split — `docs/BENCHMARK.md` |
| Real-data subset (SciCap) | ✅ | 287 real arXiv graph plots with author captions and paper paragraphs (`data/scicap_subset`, CC BY-NC-SA 4.0); fetched with `scripts/fetch_scicap_subset.py` |
| OCR figure-element extraction | ✅ | Synthetic: text F1 0.81, type accuracy 0.86. Real: 9.1 text elements / figure, 19 of 287 figures with no text found; ~2.8 s / figure on CPU |
| Entity grounding (rules vs GNN) | ✅ on synthetic | End-to-end F1 0.38 (rules) → 0.91 (GNN), significant (+0.525, 95% CI [+0.50, +0.55]); unseen entity names F1 0.83 |
| Message-passing diagnostic | ✅ | Same model without message passing scores the same → the gain comes from the learned pairwise features, not the graph |
| Synthetic shortcut repair (SciFig-Controlled v2) | ✅ | Neutralized position shortcuts: position-only F1 dropped from 0.8295 (v1) to 0.5111 (v2); Rules F1=0.3387, LR F1=0.9207, GBDT F1=0.9749, GNN F1=0.9969 (3 seeds, bootstrap CIs) — `results/phase2/` |
| Real arXiv figure grounding (Silver-v0) | ✅ (silver_heuristic) | 250 real arXiv figures, 200 held-out test: Logistic Regression F1=0.7133 beats GNN (0.3408) and untuned rules (0.1155) at <0.01 ms latency — `results/phase1/` |
| Human external dataset: CHART-Info UB PMC | ✅ (human_external) | Evaluated on 3,280 real PMC test charts (70,581 gold boxes): Gold box text-role Acc=65.47% (Macro-F1=0.4862, Non-Tick Macro-F1=0.4072); Real OCR Acc=15.87% (Macro-F1=0.2732, Adjusted Acc=31.68%) — `results/phase3/` |
| Human external dataset: CharXiv Visual QA | ✅ (human_external) | 320 descriptive questions: Deterministic visual rules Exact=16.2% (Fuzzy=17.5%) vs. Local 4-bit 7B VLM Exact=66.6% (Fuzzy=69.4%) — `results/phase3/` |
| Real Local VLM Baseline (mlx-vlm) | ✅ | Evaluated `mlx-community/Qwen2-VL-7B-Instruct-4bit` on Apple M4 Pro (Metal); logged all raw prompts, responses, runtimes to `results/phase2/raw/vlm_charxiv_raw.jsonl` |
| Human external dataset: SciERC NER | ✅ (human_external) | Evaluated on 100 test abstracts (551 sentences): Regex baseline F1=0.0325; Fine-tuned SciBERT token classifier on MPS achieves F1=0.6277 (Method F1=0.7344) — `results/phase3/` |
| Figure retrieval on SciCap | ✅ | Real SciCap caption → figure (288 figures): TF-IDF MRR 0.9003 (text) → 0.9183 (+OCR, +0.0180); BM25 MRR 0.8646 (text) → 0.9005 (+OCR, +0.0360, p < 0.001); Dense MRR 0.8246 → 0.8374 — `results/phase2/` |
| 3D detection | ✅ with caveat | Real graph plots: image-only CNN wrongly calls 92.7% of them 3D; default "agreement" mode (image + text) lowers that to 2.4% — `results/threed_detector_modes.json` |
| Provenance guardrail | ✅ | Enforced via `tests/test_provenance_guardrail.py` (blocks fabrication, missing provenance, and false gold claims) |
| Tests | ✅ | 55 passing unit, guardrail, and tampering-detection tests in virtualenv |

## 3. What works but is limited

| Area | Status | Limitation |
|---|---|---|
| PDF ingestion | 🟡 | Every text line becomes a "paragraph"; figures are not extracted from PDFs |
| Scientific NER regex | 🟡 | Regex over a fixed vocabulary achieves F1=0.0315 on real academic writing; fine-tuned SciBERT (F1=0.7167) is required |
| Caption generation | 🟡 | Fill-in-the-blanks template, no vision-language model. Real figures: BLEU-4 0.05, ROUGE-L 0.08 against author captions |
| Claim verification | 🟡 | Word-overlap + numeric-conflict check, no NLI model; real-figure technical correctness 0.26 |
| SCQ metric | 🟡 | Computed correctly, but on real data it is 0.30 — mostly reflects the weak captions |
| OCR quality | 🟡 | EasyOCR misses dense small numerical tick labels (recall 0.086 on CHART-Info) though axis titles remain solid |
| 3D alignment | 🟡 | Excluded from primary paper claims due to structural homology redundancy and real figure domain shift (`docs/DECISION_3D.md`) |

## 4. What is not done / Quarantined

| Item | Status | Why / what is needed |
|---|---|---|
| In-house human gold annotation | ❌ (QUARANTINED) | Single-author project; no dual annotator setup. Real arXiv data is silver heuristic (`silver_heuristic`). External human gold data is used instead |
| In-house Cohen's kappa | ❌ (QUARANTINED) | Quarantined (`[REMOVED: no real data]`); no human ratings performed |
| SCQ human validation & 35.8% -> 0.2% hallucination claim | ❌ (QUARANTINED) | Quarantined (`[REMOVED: no real data]`); cannot be claimed without true human reader study |
| 50-error analysis | ❌ (QUARANTINED) | Quarantined; unverified prototype narrative removed |
| 100 "real 3D figures" evaluation | ❌ (QUARANTINED) | Figures lacked verifiable download URLs and genuine 3D coordinates; deleted and marked NOT DONE |
| Cross-Modal Search tab | ❌ | Still shows hard-coded example results; not wired to the retrieval code |
| Figure extraction from PDFs | ❌ | Figures must be supplied as separate images |
| Useful message passing in the GNN | ❌ | Currently contributes nothing measurable; needs a redesign (e.g. cross-figure graphs) or removal |
| Uploading your own PDF/figure in the UI | ❌ | UI runs the bundled demo samples; upload widgets do not send files to the backend |

## 5. What is blocked by the environment

| Item | Status | Unblock by |
|---|---|---|
| CLIP (ablation configuration C) | ⛔ | Allow `huggingface.co` in the cloud environment's network settings (code is ready: `src/models/vision/clip_encoder.py`) |
| Sentence-Transformers retrieval, SciBERT NER, NLI verification | ⛔ | Same — models are configured but cannot be downloaded |
| Downloading data directly in the cloud session | ⛔ | Same; SciCap was fetched on a local machine instead |

## 6. How to run

```bash
pip install -r requirements.txt
python -m src.data.benchmark_generator --out data/benchmark   # synthetic benchmark
python pipelines/evaluate_phase4.py                           # train 3D models (~15 min CPU)
python pipelines/evaluate_gnn.py                              # train/evaluate grounding (~10 min)
python pipelines/run_ablation.py                              # ablation A–E (~20 min)
python pipelines/evaluate_scicap.py                           # real SciCap figures (~15 min)
python pipelines/evaluate_3d_detector_modes.py                # 3D detector modes (~1 min)
python -m pytest tests/                                       # tests
uvicorn api.main:app --port 8000                              # backend
cd frontend && npm install && npm run dev                     # UI at http://localhost:5173
```

## 7. Recommended next steps (in order)

1. **Annotate ~100 SciCap figures** with element → entity links so grounding can be measured on real data.
2. **Allow HuggingFace** in the environment to enable CLIP, Sentence-Transformers, SciBERT and NLI, then re-run the evaluations.
3. **Replace the caption template** with a vision-language model (needs step 2).
4. **Improve NER** (SciBERT or a trained model) — 45% of real figures currently get no entities.
5. Wire the **Cross-Modal Search** tab to the real retrieval code.
6. Find or build a small **labelled set of real 3D figures** before making any claim about 3D on real papers.
