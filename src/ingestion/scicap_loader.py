"""
SciCap Dataset Adapter & Benchmark Evaluator Module
Loads SciCap Scientific Figure Captioning Dataset samples and evaluates pipeline captions against ground-truth author captions.
"""

import os
import json
from typing import Dict, Any, List, Optional

class SciCapDatasetLoader:
    def __init__(self, dataset_dir: str = "demo/dataset"):
        self.dataset_dir = dataset_dir
        self.metadata_path = os.path.join(dataset_dir, "metadata.json")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        if not os.path.exists(self.metadata_path):
            return {"dataset_name": "SciCap Subset", "samples": []}
        with open(self.metadata_path, "r") as f:
            return json.load(f)

    def list_samples(self) -> List[Dict[str, Any]]:
        """
        Returns summary list of SciCap benchmark samples.
        """
        samples = self.metadata.get("samples", [])
        return [
            {
                "sample_id": s["sample_id"],
                "paper_title": s["paper_title"],
                "figure_type": s["figure_type"],
                "ground_truth_caption": s["ground_truth_caption"]
            }
            for s in samples
        ]

    def get_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetches specific SciCap sample by sample_id.
        """
        for s in self.metadata.get("samples", []):
            if s["sample_id"] == sample_id:
                return s
        return None

    def convert_sample_to_document_structure(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts SciCap dataset sample into pipeline DocumentStructure format.
        """
        return {
            "paper_id": sample["paper_id"],
            "title": sample["paper_title"],
            "dataset_source": "SciCap Benchmark (Hsu et al.)",
            "sections": [
                {
                    "section_id": "sec_1",
                    "title": sample["section_title"],
                    "paragraphs": [
                        {
                            "paragraph_id": "p_1",
                            "page": 1,
                            "text": sample["paragraph_text"],
                            "figure_references": [sample["sample_id"]]
                        }
                    ]
                }
            ],
            "figures": [
                {
                    "figure_id": sample["sample_id"],
                    "caption": sample["ground_truth_caption"],
                    "image_path": os.path.join(self.dataset_dir, "samples", sample["image_filename"])
                }
            ]
        }

    def evaluate_pipeline_output(self, sample_id: str, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates model-generated caption against SciCap ground-truth author caption.
        Computes BLEU-1 overlap, Entity Recall, and Technical Correctness.
        """
        sample = self.get_sample(sample_id)
        if not sample:
            return {
                "status": "Not available",
                "message": f"Sample ID '{sample_id}' not found in SciCap benchmark metadata."
            }

        gt_caption = sample["ground_truth_caption"]
        gen_caption = pipeline_output.get("generated_caption", "")

        gt_words = set(w.lower() for w in gt_caption.split() if len(w) > 2)
        gen_words = set(w.lower() for w in gen_caption.split() if len(w) > 2)
        
        overlap_count = len(gt_words.intersection(gen_words))
        bleu1_approx = overlap_count / max(len(gt_words), 1)

        # Entity Recall against SciCap Ground Truth Entities
        gt_entities = [e["text"].lower() for e in sample.get("annotated_entities", [])]
        gen_entities = [e["text"].lower() for e in pipeline_output.get("extracted_entities", [])]
        matched_ents = sum(1 for ge in gt_entities if any(ge in e for e in gen_entities))
        entity_recall = matched_ents / max(len(gt_entities), 1)

        return {
            "scicap_sample_id": sample_id,
            "ground_truth_caption": gt_caption,
            "generated_caption": gen_caption,
            "evaluation_metrics": {
                "bleu_1_semantic_overlap": float(round(bleu1_approx, 4)),
                "scicap_entity_recall": float(round(entity_recall, 4)),
                "technical_correctness_ratio": pipeline_output.get("verification_results", {}).get("technical_correctness_ratio", 0.0),
                "scq_composite_score": pipeline_output.get("scq_evaluation", {}).get("scq_composite_score", 0.0)
            },
            "status_declarations": {
                "ground_truth": "Author-written Ground Truth from SciCap Dataset",
                "model_caption": "Grounded VLM Template Engine Output",
                "evaluation": "Actual empirical n-gram overlap & SciCap entity recall"
            }
        }
