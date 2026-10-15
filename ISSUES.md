# Repository Issues & Document Inconsistencies Log

This document tracks inconsistencies and known limitations identified during the Phase 0 audit and reproduction process, as mandated by the SCI-CAP Upgrade Guide.

---

## Issue 1: Discrepancy Between 16 PDB Structures and 43 "Unseen" Test Geometries

### Description
In earlier documentation and results tables ([`implementation_plan.md`](implementation_plan.md), §3.4), the 3D geometry alignment evaluation is described as querying a gallery of 43 unique test geometries. However, the repository contains only 16 experimental PDB structures in [`data/raw/pdb/SOURCES.md`](data/raw/pdb/SOURCES.md).

### Root Cause Analysis
1. The 400-figure benchmark (`SciFig-Controlled`) contains:
   - 80 molecular protein views rendered from the 16 real PDB structures ($16 \text{ structures} \times 5 \text{ camera views}$).
   - 60 3D surface plots ($z = f(x, y)$ analytic families).
   - 60 3D scatter / point cloud plots (analytic geometric shapes: sphere, torus, helix, cube, cylinder, cone).
2. The test split (118 figures total, 63 of which are 3D) contains:
   - 25 protein views rendered from 5 held-out PDB structures (`1HVR`, `1AKE`, `4AKE`, `2BEG`, `5A7U`).
   - 18 surface plots.
   - 20 3D scatter plots.
3. Therefore, the 43 unique test geometries correspond to:
   - $5$ held-out protein structures $+$ $18$ unique analytic surface geometries $+$ $20$ unique 3D scatter geometries $= 43$ test geometries.

### Critical Hidden Leakage Note (Identified for Phase 4)
While the 5 test proteins are split by PDB ID from the 11 training proteins, there is structural homology:
- `1AKE` (closed conformation) and `4AKE` (open conformation) are both **Adenylate Kinase**.
- `1HVR` (test) and `4E43` (train) are both **HIV-1 Protease**.
- The analytic surface and point cloud shapes belong to shared mathematical families across train and test.

### Required Remediation (Phase 4)
- Group splits strictly by CATH superfamily or fold sequence identity ($\le 30-40\%$), ensuring both conformations of Adenylate Kinase and all HIV Protease structures fall into the same split.
- Expand structures using CATH non-redundant S40 domains or hold out entire analytic shape families.

---

## Issue 2: Mischaracterization of Prototype as "Production Handover"

### Description
Previous handover dossiers and commit descriptions referenced "Production Handover" or "Complete System" despite critical components operating in preliminary prototype or heuristic states.

### Concrete Deficiencies
1. **Caption Generation**: Uses rigid fill-in-the-blanks string templates ([`src/generation/caption_generator.py`](src/generation/caption_generator.py)), achieving BLEU-4 of $0.051$ and ROUGE-L of $0.084$ on real arXiv captions. No Vision-Language Model (VLM) is integrated.
2. **Scientific NER**: Uses a static regex of ~40 hand-coded patterns ([`src/nlp/scientific_ner.py`](src/nlp/scientific_ner.py)), resulting in a **45% zero-entity rate** on real SciCap papers.
3. **Grounding Validation on Real Data**: Link prediction had zero real-world gold annotations in `data/scicap_subset/`. Metrics were only evaluated on synthetic Matplotlib renders.
4. **Pretrained Weights Offline**: CLIP, Sentence-Transformers, and NLI models were blocked by environment network isolation.

### Required Remediation
- Relabel repository documentation to reflect true maturity: **Research Prototype (~6.8/10)**.
- Implement the 7-phase upgrade plan to elevate the project to top-tier conference quality ($\ge 8/10$).
