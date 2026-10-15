| Config | Grounding F1 | Grounding Acc@1 | Grounding MRR | Caption->Fig R@1 | R@5 | MRR | Geometry->Fig R@1 | SCQ | Train time (s) | Inference ms/fig |
|---|---|---|---|---|---|---|---|---|---|---|
| A NetworkX rules | 0.3644 | 0.3118 | 0.4226 | 0.8136 | 0.9322 | 0.8651 | N/A | 0.5484 | 0.0 | 0.05 |
| B NetworkX + PyG GNN | 0.9120 ± 0.0031 | 0.9015 ± 0.0012 | 0.9036 ± 0.0006 | 0.8390 | 0.9407 | 0.8810 | N/A | 0.5753 | 1.786 | 1.847 |
| C GNN + CLIP | AVAILABLE (not wired into this ablation run) | | | | | | | | | |
| D GNN + 3D branch | 0.9068 ± 0.0044 | 0.9007 ± 0.0000 | 0.9029 ± 0.0002 | 0.8390 | 0.9407 | 0.8810 | 0.093 | 0.5737 | 1.816 | 1.91 |
| E Full multimodal (text + 2D + 3D + GNN) | 0.9072 ± 0.0051 | 0.9007 ± 0.0000 | 0.9027 ± 0.0001 | 0.8305 | 0.9237 | 0.8773 | 0.093 | 0.5741 | 1.848 | 1.846 |
