"""
Tri-Modal & Multimodal Retriever Module.
Integrates Text, 2D Visual, 3D Point-Cloud/Multi-View, and PyG GNN Graph Embeddings
for unified cross-modal scientific retrieval.
"""

from typing import Dict, Any, List, Optional
import numpy as np

class MultimodalRetriever:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.weights = self.config.get("retrieval.multimodal_weights", {
            "w_text": 0.40,
            "w_2d": 0.25,
            "w_3d": 0.20,
            "w_gnn": 0.15
        })

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculates cosine similarity between normalized vector embeddings.
        """
        if vec_a is None or vec_b is None:
            return 0.0
        vec_a = np.array(vec_a, dtype=np.float32)
        vec_b = np.array(vec_b, dtype=np.float32)
        
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        dot = float(np.dot(vec_a / norm_a, vec_b / norm_b))
        return float(round(max(dot, 0.0), 4))

    def retrieve_multimodal(
        self,
        query_text_emb: np.ndarray,
        figure_2d_emb: Optional[np.ndarray] = None,
        figure_3d_emb: Optional[np.ndarray] = None,
        gnn_graph_emb: Optional[np.ndarray] = None,
        candidate_figures: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs unified multimodal retrieval across Text, 2D Visual, 3D Point-Cloud, and GNN modalities.
        """
        candidates = candidate_figures or []
        results = []

        w_text = self.weights.get("w_text", 0.40)
        w_2d = self.weights.get("w_2d", 0.25)
        w_3d = self.weights.get("w_3d", 0.20)
        w_gnn = self.weights.get("w_gnn", 0.15)

        for cand in candidates:
            # 1. Text Similarity
            c_text_emb = cand.get("text_embedding", query_text_emb)
            s_text = self.compute_similarity(query_text_emb, c_text_emb)

            # 2. 2D Image Visual Similarity
            c_2d_emb = cand.get("2d_embedding")
            s_2d = self.compute_similarity(figure_2d_emb, c_2d_emb) if figure_2d_emb is not None and c_2d_emb is not None else 0.50

            # 3. 3D Point-Cloud / Multi-View Similarity
            c_3d_emb = cand.get("3d_embedding")
            is_cand_3d = cand.get("is_3d", False)
            if is_cand_3d and figure_3d_emb is not None and c_3d_emb is not None:
                s_3d = self.compute_similarity(figure_3d_emb, c_3d_emb)
            else:
                s_3d = 0.0  # Zero weight impact for non-3D figures

            # 4. GNN Graph Similarity
            c_gnn_emb = cand.get("gnn_embedding")
            s_gnn = self.compute_similarity(gnn_graph_emb, c_gnn_emb) if gnn_graph_emb is not None and c_gnn_emb is not None else 0.50

            # Dynamic weight adjustment for 3D vs 2D figures
            if not is_cand_3d or figure_3d_emb is None:
                # Re-distribute 3D weight to 2D & Text for 2D figures
                eff_w_text = w_text + w_3d * 0.5
                eff_w_2d = w_2d + w_3d * 0.5
                eff_w_3d = 0.0
                eff_w_gnn = w_gnn
            else:
                eff_w_text, eff_w_2d, eff_w_3d, eff_w_gnn = w_text, w_2d, w_3d, w_gnn

            total_score = (
                eff_w_text * s_text +
                eff_w_2d * s_2d +
                eff_w_3d * s_3d +
                eff_w_gnn * s_gnn
            )

            results.append({
                "figure_id": cand.get("figure_id", "fig_cand"),
                "paper_title": cand.get("paper_title", "Scientific Document"),
                "is_3d": is_cand_3d,
                "multimodal_score": float(round(total_score, 4)),
                "modality_breakdown": {
                    "text_similarity": s_text,
                    "visual_2d_similarity": s_2d,
                    "pointcloud_3d_similarity": s_3d,
                    "gnn_graph_similarity": s_gnn
                }
            })

        results.sort(key=lambda x: x["multimodal_score"], reverse=True)
        return results
