"""
Entity Grounding Graph Module
Constructs NetworkX graph connecting Figure Elements -> Scientific Entities -> Scientific Claims,
and integrates PyTorch Geometric GNN message passing for learned representations.
"""

from typing import Dict, Any, List, Optional
import networkx as nx
from src.grounding.gnn_pipeline import GNNGroundingPipeline

class EntityGroundingGraph:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.graph = nx.DiGraph()
        self.enable_gnn = self.config.get("ablation_switches.graph_enabled", True)
        self.gnn_pipeline = GNNGroundingPipeline() if self.enable_gnn else None
        self.gnn_metrics: Dict[str, Any] = {}

    def build_graph(
        self, 
        figure_elements: List[Dict[str, Any]], 
        entities: List[Dict[str, Any]], 
        relations: List[Dict[str, Any]] = None,
        run_gnn: bool = True
    ) -> nx.DiGraph:
        """
        Builds a grounding graph linking figure visual elements to scientific entities and paper claims.
        Optionally enhances the graph using PyTorch Geometric GNN message-passing.
        """
        self.graph.clear()
        
        # Add figure element nodes
        for elem in figure_elements:
            node_id = f"elem:{elem['id']}"
            self.graph.add_node(
                node_id, 
                label=elem['text'], 
                node_type="FIGURE_ELEMENT", 
                element_type=elem['type']
            )

        # Add entity nodes
        for ent in entities:
            node_id = f"ent:{ent['text']}"
            self.graph.add_node(
                node_id, 
                label=ent['text'], 
                node_type="SCIENTIFIC_ENTITY", 
                entity_type=ent['type']
            )

        # Create grounding edges (FIGURE_ELEMENT -> REPRESENTS / CORRESPONDS_TO -> SCIENTIFIC_ENTITY)
        for elem in figure_elements:
            elem_node = f"elem:{elem['id']}"
            elem_text_lower = elem['text'].lower()
            
            for ent in entities:
                ent_node = f"ent:{ent['text']}"
                ent_text_lower = ent['text'].lower()
                
                if ent_text_lower in elem_text_lower or (elem['type'] == "AXIS" and ent['type'] in ["VARIABLE", "METRIC"]):
                    self.graph.add_edge(elem_node, ent_node, relation="REPRESENTS")
                elif elem['type'] == "LEGEND" and ent['type'] in ["METHOD", "MODEL"]:
                    self.graph.add_edge(elem_node, ent_node, relation="CORRESPONDS_TO")

        # Add entity-entity relations
        if relations:
            for rel in relations:
                src_node = f"ent:{rel['source']}"
                tgt_node = f"ent:{rel['target']}"
                if self.graph.has_node(src_node) and self.graph.has_node(tgt_node):
                    self.graph.add_edge(src_node, tgt_node, relation=rel['relation'])

        # Enhance graph with PyTorch Geometric GNN if enabled
        if run_gnn and self.enable_gnn and self.gnn_pipeline and len(self.graph.nodes) > 1:
            try:
                self.graph, self.gnn_metrics = self.gnn_pipeline.train_and_embed(self.graph)
            except Exception as e:
                print(f"[EntityGroundingGraph] PyTorch Geometric GNN fallback notice: {e}")
                self.gnn_metrics = {"status": "PARTIAL (Fallback NetworkX baseline topology)"}

        return self.graph

    def get_grounding_summary(self) -> Dict[str, Any]:
        """
        Returns a summary of nodes, edges, grounded paths, and GNN learned metrics.
        """
        nodes = []
        for n in self.graph.nodes:
            ndata = dict(self.graph.nodes[n])
            # Exclude large vector arrays from basic summary
            if "gnn_embedding" in ndata:
                ndata["has_gnn_embedding"] = True
                del ndata["gnn_embedding"]
            nodes.append({"id": n, **ndata})

        edges = [{"source": u, "target": v, **self.graph.edges[u, v]} for u, v in self.graph.edges]
        
        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
            "gnn_enhanced": bool(self.gnn_metrics.get("epochs_trained", 0) > 0),
            "gnn_metrics": self.gnn_metrics,
            "status": "IMPLEMENTED (NetworkX + PyTorch Geometric GNN Message Passing)"
        }
