# Scientific Figure Grounding: Annotation Guidelines

**Version**: 1.1 (Revised post-pilot on 20 figures)  
**Task**: Scientific Figure Element Extraction and Element-to-Entity Grounding  
**Target Corpus**: Real arXiv figures from the SciCap dataset (`data/scicap_subset/`)  

---

## 1. Objective & Scope

The goal of this annotation task is to establish the first ground-truth evaluation set for **figure-element grounding** in published scientific papers:
1. **Identify** all text-bearing structural elements inside the scientific figure image.
2. **Identify** all relevant scientific named entities in the accompanying paper context (referring paragraphs and caption).
3. **Link** each figure element to the exact semantic entity it represents or corresponds to in the paper text.

---

## 2. Core Annotation Rules

### 2.1 What Counts as a Figure Element?
Annotate a tight bounding box `[x_min, y_min, x_max, y_max]` in original pixel coordinates and assign one of the following types:
- **`AXIS`**: Text indicating the quantity or variable plotted on a spatial axis (e.g., x-axis title, y-axis title).
  - *Example*: `"Epoch"`, `"Top-1 Accuracy (%)"`, `"Loss"`, `"Learning Rate"`.
- **`LEGEND`**: Text entry in a chart legend distinguishing different data series, experimental runs, or model variants.
  - *Example*: `"ResNet-50"`, `"ViT-Base"`, `"Ours (GNN)"`, `"Baseline"`.
- **`LABEL`**: Semantic text appearing inside the plotting canvas or alongside a color scale (e.g., color-bar titles, data point annotations, column headers in multi-panel figures).
  - *Example*: `"Accuracy"`, `"Normalized Density"`, `"State A"`.

> [!CAUTION]
> **Exclude Numeric Ticks**: Do NOT annotate standard numeric axis ticks or tick numbers (e.g., `"0.0"`, `"0.2"`, `"0.4"`, `"10"`, `"100"`). Only annotate tick labels if they represent categorical entities (e.g., model names along the x-axis of a bar chart).

### 2.2 What Counts as a Scientific Entity?
Extract entities explicitly named in the paper context (referring paragraphs or caption):
- **`METHOD` / `MODEL`**: Named architectures, algorithms, or systems (e.g., `"Graph Convolutional Network"`, `"Transformer"`, `"AdamW"`).
- **`METRIC`**: Evaluated quantities or criteria (e.g., `"Top-1 Accuracy"`, `"Mean Absolute Error"`, `"AUROC"`, `"F1 Score"`).
- **`DATASET`**: Benchmark corpora or physical data collections (e.g., `"ImageNet-1k"`, `"CIFAR-10"`, `"Cora"`).
- **`VARIABLE`**: Experimental or physical independent/dependent parameters (e.g., `"Epoch"`, `"Batch Size"`, `"Window Size"`).
- **`MATERIAL`**: Physical compounds, molecules, cell lines, or biological entities (e.g., `"ADR1"`, `"HIV-1 Protease"`).

### 2.3 When is an Element Linked to an Entity?
An element $e$ is linked to an entity $n$ with relation $R$ if and only if an informed reader of the paper would agree the visual element depicts, represents, or corresponds to that entity:
- **`REPRESENTS`**: Used for `AXIS` or `LABEL` elements representing a metric or variable.
  - *Example*: Axis `"Top-1 Acc. (%)"` $\xrightarrow{\text{REPRESENTS}}$ Entity `"Top-1 Accuracy"`.
- **`CORRESPONDS_TO`**: Used for `LEGEND` or `LABEL` elements denoting a specific model, method, or dataset.
  - *Example*: Legend `"GCN"` $\xrightarrow{\text{CORRESPONDS_TO}}$ Entity `"Graph Convolutional Network"`.

---

## 3. Handling Edge Cases & Pitfalls

1. **Abbreviations & Display Aliases**:
   - Figure elements almost always abbreviate entity names (e.g., `"GCN"` for `"Graph Convolutional Network"`, `"MAE"` for `"Mean Absolute Error"`, `"Acc."` for `"Accuracy"`).
   - **Rule**: Link to the full canonical entity defined in the paragraph. If the paragraph defines `"Mean Absolute Error (MAE)"`, link the axis `"MAE (eV)"` to `"Mean Absolute Error"`.
2. **Unlinked / Generic Elements**:
   - In-figure text like `"x"`, `"y"`, `"Value"`, `"Component 1"`, or `"Method"` that has no specific counterpart in the paper text must be left **unlinked** (`gold_links = []`).
3. **One Element, Multiple Potential Entities**:
   - If a legend says `"Ours (GNN + ResNet)"`, link to the specific model entity `"Ours"` or the composite system if defined. Do not split one box into arbitrary sub-boxes unless visually disjoint.
4. **Entities Appearing Only in the Caption**:
   - If an entity is named only in the caption and missing from the paragraph (e.g. `"Fig. 3: Runtime comparison of ViT and CNN"`), annotate it as an entity with `source: "caption"` and establish the link.

---

## 4. Five Worked Examples

### Example 1: Standard Method Comparison Line Plot
- **Figure ID**: `scicap_sample_01` (Accuracy vs. Epochs)
- **Elements Annotated**:
  - `axis_0`: `[x0, y0, x1, y1]`, text: `"Epoch"`, type: `AXIS`
  - `axis_1`: `[x0, y0, x1, y1]`, text: `"Top-1 Accuracy (%)"`, type: `AXIS`
  - `legend_0`: `[x0, y0, x1, y1]`, text: `"ResNet-50"`, type: `LEGEND`
  - `legend_1`: `[x0, y0, x1, y1]`, text: `"ViT-Base"`, type: `LEGEND`
- **Entities in Text**: `"ResNet-50"` (`METHOD`), `"Vision Transformer (ViT-Base)"` (`METHOD`), `"Top-1 Accuracy"` (`METRIC`), `"Epoch"` (`VARIABLE`).
- **Gold Links**:
  - `legend_0` $\xrightarrow{\text{CORRESPONDS_TO}}$ `"ResNet-50"`
  - `legend_1` $\xrightarrow{\text{CORRESPONDS_TO}}$ `"Vision Transformer (ViT-Base)"`
  - `axis_0` $\xrightarrow{\text{REPRESENTS}}$ `"Epoch"`
  - `axis_1` $\xrightarrow{\text{REPRESENTS}}$ `"Top-1 Accuracy"`

### Example 2: Window Size vs. Accuracy Bar Chart
- **Figure ID**: `scicap_7424364`
- **Elements Annotated**:
  - `axis_0`: text: `"Window Size"`, type: `AXIS`
  - `axis_1`: text: `"Accuracy Score"`, type: `AXIS`
- **Entities in Text**: `"window size"` (`VARIABLE`), `"accuracy in distance estimation"` (`METRIC`), `"mean absolute error (MAE)"` (`METRIC` - distractor).
- **Gold Links**:
  - `axis_0` $\xrightarrow{\text{REPRESENTS}}$ `"window size"`
  - `axis_1` $\xrightarrow{\text{REPRESENTS}}$ `"accuracy in distance estimation"`
  - Note: `"mean absolute error"` is a distractor mentioned in paragraph but not in this figure.

### Example 3: Ablation Bar Chart with Abbreviations
- **Elements Annotated**:
  - `legend_0`: text: `"w/o DA"`, type: `LEGEND`
  - `legend_1`: text: `"Full Model"`, type: `LEGEND`
- **Entities in Text**: `"Data Augmentation (DA)"`, `"Proposed Framework"`.
- **Gold Links**:
  - `legend_0` $\xrightarrow{\text{CORRESPONDS_TO}}$ `"Data Augmentation (DA)"`
  - `legend_1` $\xrightarrow{\text{CORRESPONDS_TO}}$ `"Proposed Framework"`

### Example 4: Attention Matrix Heatmap
- **Elements Annotated**:
  - `label_0`: text: `"Layer 6 Attention"`, type: `LABEL`
  - `axis_0`: text: `"Source Token Position"`, type: `AXIS`
  - `axis_1`: text: `"Target Token Position"`, type: `AXIS`
- **Gold Links**:
  - `label_0` $\xrightarrow{\text{REPRESENTS}}$ `"Multi-Head Attention"`

### Example 5: Generic Axes (Negative Example)
- **Elements Annotated**:
  - `axis_0`: text: `"Dimension 1"`, type: `AXIS`
  - `axis_1`: text: `"Dimension 2"`, type: `AXIS`
- **Paper Text**: Mentions t-SNE projection of representations without specific named axes.
- **Gold Links**: Empty (`[]`). (Correct negative rejection).

---

## 5. Quality Control & Inter-Annotator Agreement

- Every annotator completes a qualification batch of 10 figures.
- A random sample of 40 figures is independently annotated by a secondary annotator.
- Target Agreement: **Cohen's $\kappa \ge 0.70$** on pair link decisions; Micro-F1 $\ge 0.80$.
