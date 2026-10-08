"""
FastAPI Backend Server for Multi-Modal Scientific Visualization Understanding
Includes CORS middleware, file upload endpoints, SciCap Dataset loader, Phase 3 PyG GNN,
and Phase 4 3D Visualization Branch (PointNet + Multi-View + Multimodal Retriever).
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import sys
import shutil
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import get_config
from src.ingestion.pdf_parser import PDFIngestor
from src.ingestion.scicap_loader import SciCapDatasetLoader
from src.nlp.scientific_ner import ScientificNERExtractor
from src.retrieval.hierarchical_indexer import HierarchicalIndexer
from src.vision.figure_processor import FigureProcessor
from src.grounding.entity_graph import EntityGroundingGraph
from src.generation.caption_generator import GroundedCaptionGenerator
from src.evaluation.hallucination_checker import HallucinationChecker
from src.vision.threed_detector import ThreeDVisualizationDetector
from src.models.vision.pointnet_extractor import PointNetExtractor, load_point_cloud_fixture
from src.vision.multiview_renderer import MultiViewRenderer
from src.retrieval.multimodal_retriever import MultimodalRetriever
from pipelines.run_pipeline import run_end_to_end_pipeline

app = FastAPI(
    title="Multi-Modal Scientific Visualization Understanding API",
    version="0.1.0-mvp",
    description="Backend API for PDF ingestion, SciCap benchmark, PyG GNN, 3D PointNet, Multi-View 3D rendering, and multimodal retrieval."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    paper_path: str = "demo/sample_paper.pdf"
    figure_path: str = "demo/sample_figure.png"
    query: str = "What does this figure show regarding accuracy and training epochs?"

class ThreeDDetectRequest(BaseModel):
    figure_path: str = "demo/dataset/samples/scicap_003.png"
    caption_text: Optional[str] = "3D point cloud protein structure"
    paper_context: Optional[str] = "AlphaFold predicted 3D atomic coordinates"

scicap_loader = SciCapDatasetLoader()
threed_detector = ThreeDVisualizationDetector()
pointnet_extractor = PointNetExtractor()
multiview_renderer = MultiViewRenderer()
multimodal_retriever = MultimodalRetriever()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Multi-Modal Scientific Visualization Understanding API",
        "version": "0.1.0-mvp",
        "device": get_config().get("system.device", "mps"),
        "phases_active": ["Phase 1 (MVP)", "Phase 2 (CLIP)", "Phase 3 (PyG GNN)", "Phase 4 (3D PointNet + Multi-View)"]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "api", "3d_branch": "PointNet + MultiView Active"}

@app.post("/detect-3d")
def detect_3d(req: ThreeDDetectRequest):
    try:
        res = threed_detector.detect_3d_content(req.figure_path, req.caption_text or "", req.paper_context or "")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-pointcloud-3d")
def extract_pointcloud_3d(filepath: str = "demo/dataset/3d_samples/alphafold_t1087_3d.json"):
    try:
        pts = load_point_cloud_fixture(filepath)
        z_3d = pointnet_extractor.extract_point_cloud_embedding(pts)
        return {
            "num_points": len(pts),
            "embedding_dim": len(z_3d),
            "point_cloud_embedding": z_3d.tolist(),
            "status": "IMPLEMENTED (PointNet 3D Feature Extractor)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scicap/samples")
def list_scicap_samples():
    return {
        "dataset_name": scicap_loader.metadata.get("dataset_name"),
        "samples": scicap_loader.list_samples()
    }

@app.get("/scicap/sample/{sample_id}")
def get_scicap_sample(sample_id: str):
    sample = scicap_loader.get_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail=f"SciCap sample '{sample_id}' not found.")
    doc_struct = scicap_loader.convert_sample_to_document_structure(sample)
    return {"sample": sample, "document_structure": doc_struct}

@app.post("/scicap/run-sample/{sample_id}")
def run_scicap_sample(sample_id: str):
    sample = scicap_loader.get_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail=f"SciCap sample '{sample_id}' not found.")
    
    fig_path = os.path.join(scicap_loader.dataset_dir, "samples", sample["image_filename"])
    query = f"Explain {sample['sample_id']} figure demonstrating {sample['paper_title']}"
    
    # 3D Detection
    det_3d = threed_detector.detect_3d_content(fig_path, sample["ground_truth_caption"], sample["paragraph_text"])
    
    # Run end-to-end pipeline with PyG GNN
    pipeline_res = run_end_to_end_pipeline("demo/sample_paper.pdf", fig_path, query)
    
    # Add 3D branch analysis
    if det_3d["is_3d"]:
        pts = load_point_cloud_fixture("demo/dataset/3d_samples/alphafold_t1087_3d.json")
        z_3d = pointnet_extractor.extract_point_cloud_embedding(pts)
        z_mv = multiview_renderer.extract_multiview_embedding(fig_path)
        pipeline_res["3d_branch_analysis"] = {
            "3d_detection": det_3d,
            "pointnet_embedding_norm": float(np.linalg.norm(z_3d)),
            "multiview_embedding_norm": float(np.linalg.norm(z_mv)),
            "status": "IMPLEMENTED (3D PointNet & Multi-View Visual Embedding)"
        }
    else:
        pipeline_res["3d_branch_analysis"] = {
            "3d_detection": det_3d,
            "status": "NOT APPLICABLE (2D Figure detected)"
        }

    # Evaluate against SciCap Ground Truth
    eval_res = scicap_loader.evaluate_pipeline_output(sample_id, pipeline_res)
    pipeline_res["scicap_evaluation"] = eval_res
    
    return pipeline_res

@app.post("/run-pipeline")
def api_run_pipeline(req: QueryRequest):
    try:
        res = run_end_to_end_pipeline(req.paper_path, req.figure_path, req.query)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
