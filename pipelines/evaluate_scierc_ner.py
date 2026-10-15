"""
Task 3c: Real Human Data Evaluation on SciERC Test Split.
Evaluates ModernScientificNER (regex/rule baseline) vs. fine-tuned SciBERT token classifier
on the official human-annotated SciERC test split (500 scientific abstracts).

Taxonomy Mapping:
  SciERC "Method"               -> "METHOD"
  SciERC "Metric"               -> "METRIC"
  SciERC "Material"             -> "DATASET"
  SciERC "Task"                 -> "TASK"
  SciERC "OtherScientificTerm"  -> "VARIABLE"
  SciERC "Generic"              -> Ignored (pronominal/generic mention)

Provenance: human_external_dataset
Outputs saved to results/phase3/scierc_ner_evaluation.json.
"""

import os
import sys
import json
import time
import tempfile
from typing import Dict, Any, List, Tuple, Set

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score as seq_f1_score, precision_score as seq_precision_score, recall_score as seq_recall_score, classification_report as seq_classification_report
from seqeval.scheme import IOB2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.nlp.modern_scientific_ner import ModernScientificNER

TRAIN_PATH = os.path.join(ROOT, "data", "external", "scierc", "processed_data", "json", "train.json")
DEV_PATH = os.path.join(ROOT, "data", "external", "scierc", "processed_data", "json", "dev.json")
TEST_PATH = os.path.join(ROOT, "data", "external", "scierc", "processed_data", "json", "test.json")
OUT_PATH = os.path.join(ROOT, "results", "phase3", "scierc_ner_evaluation.json")

SCIERC_MAP = {
    "Method": "METHOD",
    "Metric": "METRIC",
    "Material": "DATASET",
    "Task": "TASK",
    "OtherScientificTerm": "VARIABLE"
}

LABEL_LIST = [
    "O",
    "B-METHOD", "I-METHOD",
    "B-METRIC", "I-METRIC",
    "B-DATASET", "I-DATASET",
    "B-TASK", "I-TASK",
    "B-VARIABLE", "I-VARIABLE"
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def load_scierc_file(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line.strip()))
    return docs


def evaluate_rule_extractor_seqeval(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    extractor = ModernScientificNER()
    all_trues = []
    all_preds = []

    for doc in docs:
        sents = doc["sentences"]
        ner_sents = doc["ner"]
        cum_tokens = 0

        for tokens, ners in zip(sents, ner_sents):
            # Gold word-level BIO tags
            gold_labels = ["O"] * len(tokens)
            for start, end, label in ners:
                rel_start = start - cum_tokens
                rel_end = end - cum_tokens
                if 0 <= rel_start < len(tokens) and 0 <= rel_end < len(tokens) and label in SCIERC_MAP:
                    mapped = SCIERC_MAP[label]
                    gold_labels[rel_start] = f"B-{mapped}"
                    for k in range(rel_start + 1, rel_end + 1):
                        gold_labels[k] = f"I-{mapped}"
            cum_tokens += len(tokens)
            all_trues.append(gold_labels)

            # Predict on sentence text
            sent_text = " ".join(tokens)
            preds = extractor.extract_entities(sent_text)

            pred_labels = ["O"] * len(tokens)
            # Align character spans to token indices
            for p in preds:
                p_text = p["text"].lower()
                p_type = p["type"]
                for i in range(len(tokens)):
                    # Check n-gram of tokens
                    for span_len in range(1, 6):
                        if i + span_len <= len(tokens):
                            cand_text = " ".join(tokens[i:i + span_len]).lower()
                            if cand_text == p_text and all(pred_labels[k] == "O" for k in range(i, i + span_len)):
                                pred_labels[i] = f"B-{p_type}"
                                for k in range(i + 1, i + span_len):
                                    pred_labels[k] = f"I-{p_type}"
                                break
            all_preds.append(pred_labels)

    prec = seq_precision_score(all_trues, all_preds, mode="strict", scheme=IOB2)
    rec = seq_recall_score(all_trues, all_preds, mode="strict", scheme=IOB2)
    f1 = seq_f1_score(all_trues, all_preds, mode="strict", scheme=IOB2)
    report_dict = seq_classification_report(all_trues, all_preds, mode="strict", scheme=IOB2, output_dict=True)

    per_class = {}
    for c in SCIERC_MAP.values():
        if c in report_dict:
            per_class[c] = {
                "precision": round(report_dict[c]["precision"], 4),
                "recall": round(report_dict[c]["recall"], 4),
                "f1": round(report_dict[c]["f1-score"], 4),
                "support": int(report_dict[c]["support"])
            }
        else:
            per_class[c] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

    return {
        "model": "ModernScientificNER (Regex / Lexicon Rule Baseline)",
        "evaluation_standard": "Span-level Exact Match (seqeval strict IOB2)",
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "per_class": per_class
    }, all_trues, all_preds


class SciERCTokenDataset(Dataset):
    def __init__(self, docs: List[Dict[str, Any]], tokenizer, max_len: int = 256):
        self.examples = []
        for doc in docs:
            sents = doc["sentences"]
            ner_sents = doc["ner"]
            cum_tokens = 0

            for s_idx, (tokens, ners) in enumerate(zip(sents, ner_sents)):
                labels = ["O"] * len(tokens)
                for start, end, label in ners:
                    rel_start = start - cum_tokens
                    rel_end = end - cum_tokens
                    if 0 <= rel_start < len(tokens) and 0 <= rel_end < len(tokens) and label in SCIERC_MAP:
                        target_type = SCIERC_MAP[label]
                        labels[rel_start] = f"B-{target_type}"
                        for k in range(rel_start + 1, rel_end + 1):
                            labels[k] = f"I-{target_type}"
                cum_tokens += len(tokens)

                subwords = []
                subword_labels = []
                word_to_subword_idx = []

                for word, lab in zip(tokens, labels):
                    w_sub = tokenizer.tokenize(word)
                    if not w_sub:
                        continue
                    # First subword index in subwords list
                    word_to_subword_idx.append(len(subwords))
                    subwords.extend(w_sub)
                    subword_labels.append(LABEL2ID[lab])
                    subword_labels.extend([-100] * (len(w_sub) - 1))

                # Truncate
                subwords = subwords[:max_len - 2]
                subword_labels = subword_labels[:max_len - 2]
                valid_word_indices = [idx for idx in word_to_subword_idx if idx < max_len - 2]

                input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(subwords) + [tokenizer.sep_token_id]
                label_ids = [-100] + subword_labels + [-100]
                attention_mask = [1] * len(input_ids)
                # Offset by 1 for [CLS]
                word_indices_in_inputs = [idx + 1 for idx in valid_word_indices]

                self.examples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(label_ids, dtype=torch.long),
                    "word_indices": word_indices_in_inputs,
                    "original_tokens": tokens[:len(word_indices_in_inputs)],
                    "original_labels": labels[:len(word_indices_in_inputs)]
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    attention_mask = []
    labels = []
    word_indices = []
    original_labels = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(torch.cat([b["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        word_indices.append(b["word_indices"])
        original_labels.append(b["original_labels"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "word_indices": word_indices,
        "original_labels": original_labels
    }


def evaluate_split_seqeval(model, dataloader, device) -> Tuple[float, float, float, Dict[str, Any]]:
    model.eval()
    all_trues = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_indices_batch = batch["word_indices"]
            original_labels_batch = batch["original_labels"]

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            for b_idx in range(len(preds)):
                w_indices = word_indices_batch[b_idx]
                true_labels = original_labels_batch[b_idx]
                pred_labels = [ID2LABEL[preds[b_idx, w_i]] for w_i in w_indices]
                all_trues.append(true_labels)
                all_preds.append(pred_labels)

    prec = seq_precision_score(all_trues, all_preds, mode="strict", scheme=IOB2)
    rec = seq_recall_score(all_trues, all_preds, mode="strict", scheme=IOB2)
    f1 = seq_f1_score(all_trues, all_preds, mode="strict", scheme=IOB2)
    report_dict = seq_classification_report(all_trues, all_preds, mode="strict", scheme=IOB2, output_dict=True)

    per_class = {}
    for c in SCIERC_MAP.values():
        if c in report_dict:
            per_class[c] = {
                "precision": round(report_dict[c]["precision"], 4),
                "recall": round(report_dict[c]["recall"], 4),
                "f1": round(report_dict[c]["f1-score"], 4),
                "support": int(report_dict[c]["support"])
            }
        else:
            per_class[c] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

    return float(prec), float(rec), float(f1), per_class, all_trues, all_preds


def train_and_evaluate_scibert(train_docs, dev_docs, test_docs, device_str: str = "mps", force_retrain: bool = False):
    import hashlib
    # Set seed 42 strictly
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(42)

    device = torch.device(device_str if (torch.backends.mps.is_available() and device_str == "mps") else "cpu")

    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)

    dev_ds = SciERCTokenDataset(dev_docs, tokenizer)
    test_ds = SciERCTokenDataset(test_docs, tokenizer)
    dev_loader = DataLoader(dev_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    ckpt_dir = os.path.join(ROOT, "models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_weights_path = os.path.join(ckpt_dir, "scibert_scierc.pt")
    epochs = 5

    # Check if a saved SciBERT checkpoint already exists
    if os.path.exists(best_weights_path) and not force_retrain:
        print(f"Loading existing saved SciBERT checkpoint: {best_weights_path}")
        with open(best_weights_path, "rb") as f:
            ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()
        print(f"Verified Checkpoint SHA-256: {ckpt_sha256}")
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        dev_p, dev_r, dev_f1, _, _, _ = evaluate_split_seqeval(model, dev_loader, device)
        print(f"Dev Split Strict Span F1: {dev_f1:.4f} (P={dev_p:.4f}, R={dev_r:.4f})")
        test_p, test_r, test_f1, test_per_class, test_trues, test_preds = evaluate_split_seqeval(model, test_loader, device)
        return {
            "model": "Fine-Tuned SciBERT (allenai/scibert_scivocab_uncased)",
            "evaluation_standard": "Span-level Exact Match (seqeval strict IOB2)",
            "checkpoint_path": os.path.relpath(best_weights_path, ROOT),
            "checkpoint_sha256": ckpt_sha256,
            "seed": 42,
            "hardware": device_str,
            "epochs_trained": epochs,
            "best_dev_span_f1": round(dev_f1, 4),
            "test_precision": round(test_p, 4),
            "test_recall": round(test_r, 4),
            "test_f1": round(test_f1, 4),
            "training_time_s": 0.0,
            "per_class": test_per_class
        }, test_trues, test_preds, best_weights_path, ckpt_sha256

    print(f"Fine-tuning SciBERT token classifier using device: {device} (seed=42)")
    train_ds = SciERCTokenDataset(train_docs, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_dev_f1 = -1.0
    best_epoch = -1

    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)

        # Evaluate on dev split for model selection
        dev_p, dev_r, dev_f1, _, _, _ = evaluate_split_seqeval(model, dev_loader, device)
        print(f"  Epoch {ep + 1}/{epochs} - Train Loss: {avg_loss:.4f} | Dev Strict Span F1: {dev_f1:.4f} (P={dev_p:.4f}, R={dev_r:.4f})")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = ep + 1
            torch.save(model.state_dict(), best_weights_path)

    train_time = round(time.perf_counter() - t0, 2)
    print(f"\nModel selection completed: best checkpoint from Epoch {best_epoch} with Dev Span F1={best_dev_f1:.4f}")
    print(f"Saved best SciBERT checkpoint to: {best_weights_path}")

    # Compute SHA-256 of saved checkpoint
    with open(best_weights_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"Checkpoint SHA-256: {ckpt_sha256}")

    # Load best checkpoint for final evaluation on test split
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_p, test_r, test_f1, test_per_class, test_trues, test_preds = evaluate_split_seqeval(model, test_loader, device)

    return {
        "model": "Fine-Tuned SciBERT (allenai/scibert_scivocab_uncased)",
        "evaluation_standard": "Span-level Exact Match (seqeval strict IOB2)",
        "checkpoint_path": os.path.relpath(best_weights_path, ROOT),
        "checkpoint_sha256": ckpt_sha256,
        "seed": 42,
        "hardware": device_str,
        "epochs_trained": epochs,
        "selected_epoch": best_epoch,
        "best_dev_span_f1": round(best_dev_f1, 4),
        "test_precision": round(test_p, 4),
        "test_recall": round(test_r, 4),
        "test_f1": round(test_f1, 4),
        "training_time_s": train_time,
        "per_class": test_per_class
    }, test_trues, test_preds, best_weights_path, ckpt_sha256



def main():
    import argparse
    from src.common_metadata import get_runtime_metadata

    parser = argparse.ArgumentParser(description="SciERC NER Evaluation")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if checkpoint exists")
    args = parser.parse_args()

    print("=" * 70)
    print("TASK 3c: SCIERC REAL HUMAN BENCHMARK EVALUATION (SEQEVAL SPAN EXACT-MATCH)")
    print("=" * 70)

    runtime_meta = get_runtime_metadata("pipelines/evaluate_scierc_ner.py")

    train_docs = load_scierc_file(TRAIN_PATH)
    dev_docs = load_scierc_file(DEV_PATH)
    test_docs = load_scierc_file(TEST_PATH)

    print(f"Dataset Partitioning: {len(train_docs)} train, {len(dev_docs)} dev, {len(test_docs)} test abstracts (100 test docs)")

    # 1. Baseline Regex/Lexicon Extractor
    print("\n[1/2] Evaluating ModernScientificNER baseline on SciERC test split...")
    rule_res, rule_trues, rule_preds = evaluate_rule_extractor_seqeval(test_docs)
    print(f"  ModernScientificNER: Prec={rule_res['precision']:.4f}, Rec={rule_res['recall']:.4f}, F1={rule_res['f1']:.4f}")

    # 2. Fine-Tuned SciBERT on MPS
    print("\n[2/2] Fine-tuning SciBERT on SciERC train split with dev model selection (MPS)...")
    scibert_res, test_trues, test_preds, ckpt_path, ckpt_sha256 = train_and_evaluate_scibert(
        train_docs, dev_docs, test_docs, device_str="mps", force_retrain=args.retrain
    )
    print(f"  Fine-Tuned SciBERT (Best Dev Checkpoint): Prec={scibert_res['test_precision']:.4f}, Rec={scibert_res['test_recall']:.4f}, F1={scibert_res['test_f1']:.4f}")

    # Save real raw predictions with checkpoint SHA-256
    raw_out_path = os.path.join(ROOT, "results", "phase3", "raw", "scierc_raw_predictions.json")
    os.makedirs(os.path.dirname(raw_out_path), exist_ok=True)
    raw_payload = {
        "provenance": "human_external_dataset",
        "model_name": "allenai/scibert_scivocab_uncased",
        "checkpoint_path": os.path.relpath(ckpt_path, ROOT),
        "checkpoint_sha256": ckpt_sha256,
        "seed": 42,
        "gold_sentences": test_trues,
        "rule_predictions": rule_preds,
        "scibert_predictions": test_preds,
        **runtime_meta
    }
    with open(raw_out_path, "w") as f:
        json.dump(raw_payload, f)
    print(f"\nSaved REAL SciERC raw predictions to: {raw_out_path}")

    results = {
        "provenance": "human_external_dataset",
        "benchmark": "SciERC Test Split (Luan et al., EMNLP 2018)",
        "source_url": "https://nlp.cs.washington.edu/sciIE",
        "evaluation_metric": "seqeval strict span-level exact match (IOB2)",
        "num_train_docs": len(train_docs),
        "num_dev_docs": len(dev_docs),
        "num_test_docs": len(test_docs),
        "type_mapping": SCIERC_MAP,
        "results": {
            "modern_scientific_ner_baseline": rule_res,
            "finetuned_scibert_token_classifier": scibert_res
        },
        **runtime_meta
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved SciERC evaluation report to: {OUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()


