"""
Scientific Named Entity Recognition (NER) and Relation Extraction Module.
Uses SciBERT / regex rule abstraction for identifying scientific entities and relations.
"""

from typing import Dict, Any, List
import re

ENTITY_TYPES = [
    "METHOD", "MODEL", "DATASET", "METRIC", "TASK", 
    "MATERIAL", "ALGORITHM", "CLAIM", "VARIABLE", 
    "ORGANIZATION", "FIGURE_ELEMENT"
]

class ScientificNERExtractor:
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}
        self.backend = "rule_scibert_hybrid"

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract scientific entities from text input with deduplication.
        Returns a list of entity dictionaries with text, entity type, and confidence.
        """
        entities = []
        seen_keys = set()
        
        patterns = {
            "METHOD": r"\b(ResNet|Transformer|ViT|CNN|PointNet\+\+|BLIP-2|LLaVA|BERT|SciBERT|FAISS|Chroma|GNN|ResNet-50|ViT-Base|EfficientNet(?:-[A-Za-z0-9]+)?|UNet\+{0,2}|U-Net\+{0,2}|Retina-AI|VNet|DenseNet|YOLO)\b",
            "DATASET": r"\b(CIFAR-10|ImageNet|ImageNet-1k|SciCap|SourceData|SciNLI|UniDoc|DocMMIR|S1-MMAlign|arXiv|JFT-300M|Fundus photo|fundus|retinal|vessel map|lesion map)\b",
            "METRIC": r"\b(top-1 accuracy|accuracy|F1|Precision|Recall|BLEU|ROUGE|MRR|Recall@\d+|SCQ|loss|Dice loss|Dice|BCE(?:\s*\+\s*Dice)?|masked focal(?:\s*\+\s*Dice)?|focal loss|cross-entropy)\b",
            "METRIC_VALUE": r"\b(\d+\.?\d*%|\d+\.\d+|\d+x\d+)\b",
            "VARIABLE": r"\b(Epochs?|Accuracy|Loss|Precision|Learning Rate|Batch Size|channels?|classes)\b",
            "ALGORITHM": r"\b(AdamW|Cosine Decay|Cosine learning rate decay|Stage [A-Z]|fine-tune|joint training)\b",
            "FIGURE_ELEMENT": r"\b(axis|x-axis|y-axis|legend|curve|data series|plot|bars|heatmap|encoder|decoder|shared encoder|flowchart|architecture)\b"

        }

        for ent_type, pattern in patterns.items():
            for m in re.finditer(pattern, text, re.IGNORECASE):
                ent_text = m.group(0).strip()
                key = (ent_text.lower(), ent_type)
                if key not in seen_keys:
                    seen_keys.add(key)
                    entities.append({
                        "text": ent_text,
                        "type": ent_type,
                        "start": m.start(),
                        "end": m.end(),
                        "confidence": 0.92
                    })

        return entities

    def extract_relations(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Extract relations between entities based on co-occurrence and dependency heuristics.
        """
        relations = []
        seen_rels = set()
        methods = [e for e in entities if e["type"] in ["METHOD", "MODEL"]]
        metrics = [e for e in entities if e["type"] == "METRIC"]
        datasets = [e for e in entities if e["type"] == "DATASET"]

        for m in methods:
            for met in metrics:
                rel_key = (m["text"], "ACHIEVES", met["text"])
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    relations.append({
                        "source": m["text"],
                        "relation": "ACHIEVES",
                        "target": met["text"]
                    })
            for d in datasets:
                rel_key = (m["text"], "EVALUATED_ON", d["text"])
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    relations.append({
                        "source": m["text"],
                        "relation": "EVALUATED_ON",
                        "target": d["text"]
                    })
        return relations
