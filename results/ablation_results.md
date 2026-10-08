| Config | Grounding F1 | Grounding Acc@1 | Grounding MRR | Caption->Fig R@1 | R@5 | MRR | Geometry->Fig R@1 | SCQ | Train time (s) | Inference ms/fig |
|---|---|---|---|---|---|---|---|---|---|---|
| A NetworkX rules | 0.3810 | 0.3200 | 0.4284 | 0.8220 | 0.9322 | 0.8683 | N/A | 0.5432 | 0.0 | 0.078 |
| B NetworkX + PyG GNN | 0.9054 ± 0.0034 | 0.8958 ± 0.0035 | 0.8981 ± 0.0019 | 0.8390 | 0.9407 | 0.8803 | N/A | 0.5697 | 4.086 | 4.007 |
| C GNN + CLIP | BLOCKED: CLIP weights 'openai/clip-vit-base-patch32' could n | | | | | | | | | |
| D GNN + 3D branch | 0.9087 ± 0.0017 | 0.8975 ± 0.0012 | 0.8991 ± 0.0006 | 0.8390 | 0.9492 | 0.8843 | 0.1395 | 0.5723 | 4.002 | 4.004 |
| E Full multimodal (text + 2D + 3D + GNN) | 0.9103 ± 0.0012 | 0.8966 ± 0.0012 | 0.8987 ± 0.0006 | 0.8390 | 0.9407 | 0.8856 | 0.1395 | 0.5726 | 4.151 | 4.075 |
