"""
Hierarchical Retrieval & Indexing Module.
Supports 4 levels: Paper, Section, Paragraph, Entity level.
Uses sentence-transformers + FAISS / cosine similarity with configurable weights and robust offline fallback.
"""

from typing import Dict, Any, List
import numpy as np

class HierarchicalIndexer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", weights: Dict[str, float] = None):
        self.weights = weights or {
            "w_paper": 0.15,
            "w_section": 0.25,
            "w_paragraph": 0.40,
            "w_entity": 0.20
        }
        self.use_st = False
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_embedding_dimension() if hasattr(self.model, "get_embedding_dimension") else self.model.get_sentence_embedding_dimension()
            self.faiss = faiss
            self.indexes = {
                "paper": faiss.IndexFlatIP(self.dimension),
                "section": faiss.IndexFlatIP(self.dimension),
                "paragraph": faiss.IndexFlatIP(self.dimension),
                "entity": faiss.IndexFlatIP(self.dimension)
            }
            self.use_st = True
        except Exception as e:
            print(f"[HierarchicalIndexer] HuggingFace/SentenceTransformer unavailable ({e}). Using lightweight vector overlap fallback engine.")
            self.use_st = False

        self.chunks = {
            "paper": [],
            "section": [],
            "paragraph": [],
            "entity": []
        }

    def index_document(self, doc_structure: Dict[str, Any], entities: List[Dict[str, Any]] = None):
        """
        Indexes document chunks at paper, section, paragraph, and entity levels.
        """
        paper_id = doc_structure.get("paper_id", "doc_1")
        
        # 1. Paper level chunk
        full_text = f"{doc_structure.get('title', '')} " + " ".join(
            [p["text"] for sec in doc_structure.get("sections", []) for p in sec.get("paragraphs", [])]
        )
        self._add_chunk("paper", full_text[:2000], {"paper_id": paper_id, "chunk_type": "paper", "text": full_text[:2000]})

        # 2. Section level chunks
        for sec in doc_structure.get("sections", []):
            sec_text = f"{sec.get('title', '')}: " + " ".join([p["text"] for p in sec.get("paragraphs", [])])
            self._add_chunk("section", sec_text[:1000], {
                "paper_id": paper_id,
                "section_id": sec.get("section_id"),
                "chunk_type": "section",
                "text": sec_text[:1000]
            })
            
            # 3. Paragraph level chunks
            for p in sec.get("paragraphs", []):
                self._add_chunk("paragraph", p["text"], {
                    "paper_id": paper_id,
                    "section_id": sec.get("section_id"),
                    "paragraph_id": p.get("paragraph_id"),
                    "chunk_type": "paragraph",
                    "text": p["text"]
                })

        # 4. Entity level chunks
        if entities:
            for ent in entities:
                self._add_chunk("entity", ent["text"], {
                    "paper_id": paper_id,
                    "entity_type": ent["type"],
                    "chunk_type": "entity",
                    "text": ent["text"]
                })

    def _add_chunk(self, level: str, text: str, metadata: Dict[str, Any]):
        if not text.strip():
            return
        if self.use_st:
            try:
                emb = self.model.encode([text], normalize_embeddings=True)
                self.indexes[level].add(np.array(emb, dtype=np.float32))
            except Exception:
                self.use_st = False
        self.chunks[level].append(metadata)

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Performs 4-level retrieval and applies configurable context weighting.
        """
        results = {}

        if self.use_st:
            try:
                query_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
                for level in ["paper", "section", "paragraph", "entity"]:
                    idx = self.indexes[level]
                    if idx.ntotal == 0:
                        results[level] = []
                        continue
                    scores, indices = idx.search(query_emb, min(top_k, idx.ntotal))
                    level_results = []
                    for score, idx_val in zip(scores[0], indices[0]):
                        if idx_val < len(self.chunks[level]):
                            chunk_info = dict(self.chunks[level][idx_val])
                            chunk_info["score"] = float(score)
                            level_results.append(chunk_info)
                    results[level] = level_results
            except Exception:
                self.use_st = False

        if not self.use_st:
            # Word-overlap vector fallback
            query_words = set(query.lower().split())
            for level in ["paper", "section", "paragraph", "entity"]:
                level_chunks = self.chunks[level]
                scored = []
                for chunk in level_chunks:
                    text_words = set(chunk["text"].lower().split())
                    overlap = len(query_words.intersection(text_words)) / max(len(query_words), 1)
                    c_info = dict(chunk)
                    c_info["score"] = float(min(0.5 + overlap * 0.45, 0.95))
                    scored.append(c_info)
                scored.sort(key=lambda x: x["score"], reverse=True)
                results[level] = scored[:top_k]

        # Calculate weighted hierarchical context score
        weighted_score = 0.0
        weighted_score += self.weights.get("w_paper", 0.15) * (results["paper"][0]["score"] if results["paper"] else 0.0)
        weighted_score += self.weights.get("w_section", 0.25) * (results["section"][0]["score"] if results["section"] else 0.0)
        weighted_score += self.weights.get("w_paragraph", 0.40) * (results["paragraph"][0]["score"] if results["paragraph"] else 0.0)
        weighted_score += self.weights.get("w_entity", 0.20) * (results["entity"][0]["score"] if results["entity"] else 0.0)

        return {
            "query": query,
            "weighted_score": float(weighted_score),
            "level_results": results
        }
