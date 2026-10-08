"""
GNN Pipeline & Self-Supervised Training Pipeline Module.
Converts NetworkX graphs to PyG, runs self-supervised GNN link prediction training,
and attaches learned node/graph embeddings back to the grounding graph.
"""

from typing import Dict, Any, List, Tuple
import torch
import torch.optim as optim
import networkx as nx
import numpy as np

from src.grounding.pyg_converter import PyGGraphConverter
from src.models.grounding.gnn_grounding import GNNGroundingModel

class GNNGroundingPipeline:
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        gnn_type: str = "GAT",
        lr: float = 0.01,
        epochs: int = 30,
        device: str = None
    ):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        self.lr = lr
        self.epochs = epochs
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.converter = PyGGraphConverter(feature_dim=feature_dim)
        self.model = GNNGroundingModel(
            in_channels=feature_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            gnn_type=gnn_type
        ).to(self.device)

    def train_and_embed(self, graph: nx.DiGraph) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Runs conversion, self-supervised GNN training, and attaches learned node embeddings to NetworkX graph.
        Returns:
            - graph: NetworkX graph with node['gnn_embedding'] attached
            - metrics: Dict containing loss history and GNN summary
        """
        pyg_data, node2idx, idx2node = self.converter.convert_to_pyg_data(graph)
        
        if pyg_data.num_nodes == 0 or pyg_data.edge_index.numel() == 0:
            # Fallback for empty/single node graphs
            metrics = {
                "epochs_trained": 0,
                "final_loss": 0.0,
                "gnn_type": self.gnn_type,
                "status": "PARTIAL (Insufficient edges for GNN message passing)"
            }
            return graph, metrics

        pyg_data = pyg_data.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Self-Supervised Link Prediction Training Loop
        self.model.train()
        loss_history = []
        
        pos_edge_index = pyg_data.edge_index
        num_pos = pos_edge_index.size(1)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            z = self.model(pyg_data.x, pos_edge_index)

            # Positive edge predictions
            src_pos, tgt_pos = pos_edge_index[0], pos_edge_index[1]
            pos_scores = self.model.predict_link(z[src_pos], z[tgt_pos])
            pos_loss = -torch.log(pos_scores + 1e-8).mean()

            # Sample negative edges for contrastive link prediction
            neg_src = torch.randint(0, pyg_data.num_nodes, (num_pos,), device=self.device)
            neg_tgt = torch.randint(0, pyg_data.num_nodes, (num_pos,), device=self.device)
            neg_scores = self.model.predict_link(z[neg_src], z[neg_tgt])
            neg_loss = -torch.log(1.0 - neg_scores + 1e-8).mean()

            total_loss = pos_loss + neg_loss
            total_loss.backward()
            optimizer.step()

            loss_history.append(float(total_loss.item()))

        # Inference: Extract learned embeddings
        self.model.eval()
        with torch.no_grad():
            z_final = self.model(pyg_data.x, pos_edge_index)
            graph_emb = self.model.get_graph_embedding(z_final).cpu().numpy()[0]

        # Attach GNN embeddings back into NetworkX graph nodes
        z_np = z_final.cpu().numpy()
        for idx, node_id in idx2node.items():
            if graph.has_node(node_id):
                graph.nodes[node_id]["gnn_embedding"] = z_np[idx].tolist()

        # Compute GNN-enhanced element-to-entity grounding similarity scores
        gnn_grounding_pairs = []
        for u, v, d in graph.edges(data=True):
            if u in node2idx and v in node2idx:
                u_idx, v_idx = node2idx[u], node2idx[v]
                sim = float(np.dot(z_np[u_idx], z_np[v_idx]))
                d["gnn_grounding_score"] = float(round(sim, 4))
                gnn_grounding_pairs.append({
                    "source": u,
                    "target": v,
                    "relation": d.get("relation", ""),
                    "gnn_similarity": float(round(sim, 4))
                })

        metrics = {
            "epochs_trained": self.epochs,
            "initial_loss": float(round(loss_history[0], 4)) if loss_history else 0.0,
            "final_loss": float(round(loss_history[-1], 4)) if loss_history else 0.0,
            "gnn_type": self.gnn_type,
            "gnn_grounding_pairs": gnn_grounding_pairs,
            "graph_embedding": graph_emb.tolist(),
            "status": "IMPLEMENTED (PyTorch Geometric GNN Message Passing)"
        }

        return graph, metrics
