# Datasheet for SciCap-Grounding-Silver-v0

Following the framework proposed by Gebru et al. (*Datasheets for Datasets*, CACM 2021).

---

## 1. Motivation

- **For what purpose was the dataset created?**  
  To evaluate scientific figure element-to-entity grounding on real published arXiv figures using transparent, reproducible heuristic silver labels.
- **Who created the dataset?**  
  The SCI-CAP Research Team (algorithmic extraction pipeline; provenance: `silver_heuristic`).
- **Notice on Human Annotation:**  
  This dataset is **silver** (heuristically extracted). It was **not** hand-annotated by humans. Human-annotated evaluations in this project are conducted exclusively on established external open benchmarks (CHART-Info, CharXiv, SciERC).

---

## 2. Composition

- **What do the instances that comprise the dataset represent?**  
  Each instance corresponds to a real arXiv figure from SciCap with its caption, mentioning paragraph, OCR text elements, extracted scientific entities, and heuristic silver grounding links.
- **How many instances are there in total?**  
  - Total figures: **250**
  - Validation split: **50 figures**
  - Test split: **200 figures**
  - Provenance: `silver_heuristic`
  - Silver grounding links: **212** (Val: 23, Test: 189)
- **Data structure:**  
  Structured JSON record with `figure_id`, `split`, `arxiv_id`, `image_path`, `caption`, `paragraph_text`, `elements`, `entities`, `silver_links`, and `provenance`.

---

## 3. Collection Process

- **How was the data generated?**  
  Elements were extracted via EasyOCR with character-level bounding boxes. Entities were extracted using `ModernScientificNER` matching SciERC/SciER taxonomies. Grounding links were generated via normalized string alignment, acronym matching, and layout type constraints.
- **Annotators:**  
  Heuristic algorithm only. No human annotators were employed.

---

## 4. Preprocessing and Leakage Prevention

- **Paper-level splitting:**  
  Partitioned strictly by arXiv ID (`GroupShuffleSplit`) to guarantee zero paper overlap between validation and test splits.
- **Deduplication:**  
  Perceptual image hashing (`imagehash.phash`) was performed across splits.

---

## 5. Uses, Limitations, and Licensing

- **Limitations:**  
  Because links are generated heuristically (silver), they serve as an automated sanity check rather than a gold human standard.
- **License:**  
  CC BY-NC-SA 4.0 (inheriting SciCap / arXiv terms).
