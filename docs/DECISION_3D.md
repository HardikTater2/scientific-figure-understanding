# 3D Component Evaluation and Decision Record (Phase 4)

---

## 1. Executive Summary: Formal Decision

**Decision:** **CUT the 3D cross-modal retrieval branch from the primary conference submission** and relocate it to a concise Future Work / System Demonstration section.

**Rationale:**
1. **Structural Fold Leakage in PDB Splits:** Evaluation revealed that PDB ID splitting allowed homologous proteins sharing the exact same CATH structural fold (e.g., 1HVR in test vs. 4E43 in train, both HIV-1 proteases; 1AKE vs. 4AKE, adenylate kinase conformers) to leak across splits. While we developed a leak-free CATH superfamily split (`src/data/protein_clustering.py`), the small pool of 16 structures yields wide confidence intervals ($R@1 = 0.159 \pm 0.026$) that do not support a top-tier standalone claim.
2. **Failure of Synthetic Image CNN on Real Figures:** On 100 real scientific figures (`results/phase4_real_3d_evaluation.json`), the FigureImageEncoder CNN exhibited a **96.0% false alarm rate** on real 2D plots. Text cues alone achieved superior performance ($F_1 = 1.00$ vs. $0.629$ for the CNN), demonstrating that the vision model adds no measurable discriminative value on real papers.
3. **Paper Focus & Reviewer Risk:** Including an unproven 3D retrieval section would dilute the core paper narrative and invite valid reviewer criticism regarding small sample sizes. Cutting 3D protects the integrity of the main grounding contribution.

---

## 2. Empirical Audit of 3D Detector on 100 Real Figures

| Mode | Input Modality | Recall (TPR) | False Alarm (FPR) | Precision | $F_1$ Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Image-Only** | Pixels (FigureImageEncoder CNN) | 0.9000 (45/50) | **0.9600** (48/50) | 0.4839 | 0.6294 | Severe out-of-domain failure |
| **Text-Only** | Caption & Paragraph Cues | **1.0000** (50/50) | **0.0000** (0/50) | **1.0000** | **1.0000** | Highly discriminative |
| **Agreement** | Conjunction (CNN $\land$ Text) | 0.9000 (45/50) | 0.0000 (0/50) | 1.0000 | 0.9474 | CNN degrades recall by -10% |

*Key finding:* Conjunction mode lowers recall compared to pure text cues without reducing false alarms. The pixel-level CNN does not generalize to diverse published journal layouts.

---

## 3. Leakage Resolution via CATH Structural Clusters

To prevent future leakage, we implemented `src/data/protein_clustering.py` and unit tests in `tests/test_protein_clustering.py`:
- Proteins are grouped by CATH Superfamily / Fold (e.g., CATH 2.40.70.10 for retroviral proteases; CATH 3.40.50.300 for adenylate kinases).
- Splitting uses `GroupShuffleSplit` on CATH IDs.
- Automated assertion test (`assert_cluster_leak_free`) enforces zero fold overlap between train and test.

---

## 4. Manuscript Text for Future Work

The paper will contain the following transparent disclosure:
> *"While early prototypes explored cross-modal 3D biomolecular retrieval against PDB coordinate files, our empirical audit revealed that pixel-level CNNs trained on synthetic matplotlib plots fail to transfer to published figures (exhibiting a 96% false-positive rate on real 2D charts), while structural fold redundancy across PDB IDs risks evaluation leakage. We therefore restrict our present benchmark to 2D scientific document figures and defer large-scale, CATH-clustered 3D retrieval to future work."*
