"""
Unit and Integration Tests for Phase 4 — 3D Scientific Visualization Branch Expansion.
Verifies 3D content detection, PointNet 3D feature extraction, multi-view perspective rendering,
and multimodal retrieval (Text + 2D + 3D + PyG GNN).
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.threed_detector import ThreeDVisualizationDetector
from src.models.vision.pointnet_extractor import PointNetExtractor, load_point_cloud_fixture
from src.vision.multiview_renderer import MultiViewRenderer
from src.retrieval.multimodal_retriever import MultimodalRetriever
from src.grounding.entity_graph import EntityGroundingGraph

def test_threed_detector():
    detector = ThreeDVisualizationDetector()
    res_3d = detector.detect_3d_content(
        "demo/dataset/samples/scicap_003.png",
        caption_text="3D point cloud molecular rendering of protein T1087 with pLDDT score"
    )
    assert res_3d["is_3d"] == True
    assert res_3d["visualization_type"] == "3D_MOLECULAR_SURFACE"

    res_2d = detector.detect_3d_content(
        "demo/dataset/samples/scicap_001.png",
        caption_text="Top-1 accuracy comparison of ViT-Base vs ResNet-50 across epochs"
    )
    assert res_2d["is_3d"] == False
    print("✓ test_threed_detector passed")

def test_pointnet_extractor():
    extractor = PointNetExtractor(in_channels=3, out_dim=64)
    pts = load_point_cloud_fixture("demo/dataset/3d_samples/alphafold_t1087_3d.json", num_points=1024)
    
    assert pts.shape == (1024, 3)
    z_3d = extractor.extract_point_cloud_embedding(pts)

    assert z_3d.shape == (64,)
    assert np.isclose(np.linalg.norm(z_3d), 1.0, atol=1e-3)
    print("✓ test_pointnet_extractor passed")

def test_multiview_renderer():
    renderer = MultiViewRenderer(num_views=4, out_dim=64)
    views = renderer.generate_views("demo/dataset/samples/scicap_003.png")
    assert len(views) == 4

    mv_emb = renderer.extract_multiview_embedding("demo/dataset/samples/scicap_003.png")
    assert mv_emb.shape == (64,)
    assert np.isclose(np.linalg.norm(mv_emb), 1.0, atol=1e-3)
    print("✓ test_multiview_renderer passed")

def test_multimodal_retriever():
    retriever = MultimodalRetriever()
    q_emb = np.random.randn(64)
    v2d_emb = np.random.randn(64)
    v3d_emb = np.random.randn(64)
    gnn_emb = np.random.randn(64)

    candidates = [
        {
            "figure_id": "scicap_003_3d_protein",
            "paper_title": "AlphaFold Structure Prediction",
            "is_3d": True,
            "text_embedding": q_emb,
            "2d_embedding": v2d_emb,
            "3d_embedding": v3d_emb,
            "gnn_embedding": gnn_emb
        },
        {
            "figure_id": "scicap_001_2d_plot",
            "paper_title": "ResNet vs ViT Accuracy",
            "is_3d": False,
            "text_embedding": q_emb,
            "2d_embedding": v2d_emb,
            "3d_embedding": None,
            "gnn_embedding": gnn_emb
        }
    ]

    results = retriever.retrieve_multimodal(q_emb, v2d_emb, v3d_emb, gnn_emb, candidates)
    assert len(results) == 2
    assert "multimodal_score" in results[0]
    print("✓ test_multimodal_retriever passed")

def test_end_to_end_phase4_pipeline():
    # 1. Detect 3D content
    detector = ThreeDVisualizationDetector()
    det_res = detector.detect_3d_content("demo/dataset/samples/scicap_003.png", "3D molecular render of protein T1087")

    # 2. Extract 3D PointNet embedding
    extractor = PointNetExtractor()
    pts = load_point_cloud_fixture("demo/dataset/3d_samples/alphafold_t1087_3d.json")
    z_3d = extractor.extract_point_cloud_embedding(pts)

    # 3. Integrate with EntityGroundingGraph
    elements = [{"id": "3d_mesh", "type": "CHART_REGION", "text": "3D Point Cloud Mesh (AlphaFold T1087)"}]
    entities = [{"text": "AlphaFold", "type": "MODEL"}, {"text": "T1087", "type": "MATERIAL"}]
    
    graph_builder = EntityGroundingGraph()
    g = graph_builder.build_graph(elements, entities, run_gnn=True)
    summary = graph_builder.get_grounding_summary()

    assert summary["num_nodes"] > 0
    assert summary["gnn_enhanced"] == True
    print("✓ test_end_to_end_phase4_pipeline passed")

if __name__ == "__main__":
    test_threed_detector()
    test_pointnet_extractor()
    test_multiview_renderer()
    test_multimodal_retriever()
    test_end_to_end_phase4_pipeline()
    print("\nALL PHASE 4 3D BRANCH UNIT TESTS PASSED!")
