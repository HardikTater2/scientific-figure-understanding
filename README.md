# Multi-Modal Scientific Visualization Understanding with Hierarchical Context Grounding and Cross-Modal Retrieval

Research prototype for technical document understanding, figure layout extraction, 4-level hierarchical retrieval, graph-based entity grounding, grounded caption generation, and SCQ composite evaluation on Apple Silicon hardware.

## System Architecture

```
Paper PDF / Image / Query
         │
         ├── Document Ingestion (PyMuPDF) -> Hierarchy (Paper -> Section -> Paragraph -> Entity/FigureRef)
         ├── Scientific NER (SciBERT / Rule Abstraction) -> [METHOD, METRIC, DATASET, VARIABLE, ...]
         ├── 4-Level Hierarchical Indexing (FAISS + Sentence Transformers) -> Weighted Context Score
         ├── Figure Processing (OpenCV / EasyOCR) -> Visual Elements (Axes, Legends, Labels)
         ├── Entity Grounding Graph (NetworkX) -> Nodes (Elements + Entities), Edges (REPRESENTS, etc.)
         ├── Grounded Caption Generator -> Technically Grounded Caption
         ├── Hallucination / Evidence Verification -> Claims Status (SUPPORTED / CONTRADICTED / UNKNOWN)
         └── SCQ Composite Evaluation -> Composite Quality Metric
```

## Implementation Status Matrix

| Component | Status | Implementation Details |
|---|---|---|
| Document Ingestion | **IMPLEMENTED** | PyMuPDF structural parser (Paper -> Section -> Paragraph -> Figure Refs) |
| Scientific NER | **IMPLEMENTED** | SciBERT & Rule-based hybrid abstraction layer |
| Hierarchical Retrieval | **IMPLEMENTED** | 4-Level indexing (Paper, Section, Paragraph, Entity) with FAISS & Sentence-Transformers |
| Figure Processing | **IMPLEMENTED** | OpenCV layout detection + visual element extractor |
| Entity Grounding Graph | **IMPLEMENTED** | NetworkX heterogeneous grounding graph connecting elements & entities |
| Grounded Caption Generation | **IMPLEMENTED** | Evidence-grounded multimodal template captioning |
| Hallucination & Fact Check | **IMPLEMENTED** | Claim extraction & evidence overlap verification |
| SCQ Composite Evaluation | **IMPLEMENTED** | Configurable SCQ metric calculation |
| FastAPI REST Server | **IMPLEMENTED** | API endpoints in `api/main.py` |
| Demo Web UI | **IMPLEMENTED** | Lightweight web interface |
| Cross-Modal Tri-Modal Alignment | **PARTIALLY IMPLEMENTED** | Pretrained CLIP embeddings (Phase 2 expansion ready) |
| GNN Message Passing | **PLANNED** | PyTorch Geometric GNN optional module (Phase 3) |
| 3D Visualization Pathway | **PLANNED** | PointNet++ & Multi-view renderer interface (Phase 4) |

## Quick Start & Reproduction

### Prerequisites
- Python 3.10+
- Apple Silicon Mac (M4 Pro / MPS device support enabled)

### Run MVP Pipeline Command
```bash
python pipelines/run_pipeline.py --paper sample.pdf --figure figure.png --query "What does this figure show?"
```

### Run FastAPI Backend
```bash
uvicorn api.main:app --reload --port 8000
```
