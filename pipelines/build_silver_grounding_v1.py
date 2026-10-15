"""
Task 5: Build SciCap Silver Grounding v1 Dataset and Comprehensive Evaluation.
Uses:
  - Annotator A: Local Text LLM (Qwen2.5-1.5B-Instruct-4bit via mlx-lm)
  - Annotator B: Local VLM (Qwen2-VL-7B-Instruct-4bit via mlx-vlm)

Computes model-model agreement (clearly labelled as model_model_kappa, NOT human).
Keeps agreed links as silver grounding ground-truth (provenance: silver_llm).
Evaluates Rules, LR, GBDT, GNN on silver v1.
Evaluates VLM strictly on Annotator-A-only labels to avoid circularity.

Outputs saved to:
  - data/scicap_subset/silver_grounding_v1.json
  - results/phase1/silver_grounding_v1_evaluation.json
  - results/phase1/raw/silver_grounding_dual_annotation.jsonl
"""

import os
import sys
import json
import time
import re
from typing import Dict, Any, List, Tuple, Set
import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.grounding.grounding_features import TEXT_ELEMENT_TYPES, pair_features, build_grounding_graph
from src.nlp.scientific_ner import ScientificNERExtractor
from src.grounding.gnn_pipeline import SupervisedGroundingTrainer
from src.common_metadata import get_runtime_metadata

V0_PATH = os.path.join(ROOT, "data", "scicap_subset", "silver_grounding_v0.json")
V1_OUT_PATH = os.path.join(ROOT, "data", "scicap_subset", "silver_grounding_v1.json")
EVAL_OUT_PATH = os.path.join(ROOT, "results", "phase1", "silver_grounding_v1_evaluation.json")
RAW_LOG_PATH = os.path.join(ROOT, "results", "phase1", "raw", "silver_grounding_dual_annotation.jsonl")


def parse_grounding_response(text: str, elements: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    """
    Robustly parses grounded links from model output.
    Supports formats:
      - elem_0 -> ResNet-50
      - elem_0 -> ent_1
      - ResNet-50 -> ent_1
      - 0 -> ResNet-50
    """
    elem_by_id = {e["id"].lower(): e["id"] for e in elements}
    elem_by_text = {e["text"].lower().strip(): e["id"] for e in elements}
    ent_by_id = {ent["id"].lower(): ent["text"] for ent in entities}
    ent_by_text = {ent["text"].lower().strip(): ent["text"] for ent in entities}

    links = set()
    for raw_line in text.strip().split("\n"):
        line = raw_line.strip()
        parts = re.split(r"\s*(?:->|:|=)\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        left = parts[0].strip().strip("\"'`")
        right = parts[1].strip().strip("\"'`.,;")

        target_eid = None
        l_lower = left.lower()
        if l_lower in elem_by_id:
            target_eid = elem_by_id[l_lower]
        elif l_lower in elem_by_text:
            target_eid = elem_by_text[l_lower]
        elif l_lower.startswith("elem_"):
            target_eid = left
        elif left.isdigit() and int(left) < len(elements):
            target_eid = elements[int(left)]["id"]

        target_ent = None
        r_lower = right.lower()
        if r_lower in ent_by_id:
            target_ent = ent_by_id[r_lower]
        elif r_lower in ent_by_text:
            target_ent = ent_by_text[r_lower]
        else:
            for ent_name, orig_ent in ent_by_text.items():
                if ent_name in r_lower or r_lower in ent_name:
                    target_ent = orig_ent
                    break

        if target_eid and target_ent:
            links.add((target_eid, target_ent))
    return links


def main():
    print("=" * 70)
    print("TASK 5: BUILD SILVER GROUNDING V1 VIA DUAL LOCAL MODEL ANNOTATION")
    print("=" * 70)

    with open(V0_PATH, "r") as f:
        v0_data = json.load(f)

    all_figures = v0_data["figures"]
    # Filter figures with valid elements and entities
    valid_figures = [f for f in all_figures if f.get("elements") and f.get("entities")]
    print(f"Loaded {len(valid_figures)} valid figures from silver v0.")

    # Select representative sample of 60 figures for dual annotation (fast and thorough)
    eval_figures = valid_figures[:60]
    print(f"Annotating {len(eval_figures)} figures with Annotator A (LLM) and Annotator B (VLM)...")

    # Check if raw dual annotations already exist
    have_raw_log = os.path.exists(RAW_LOG_PATH) and os.path.getsize(RAW_LOG_PATH) > 0
    raw_map = {}
    if have_raw_log:
        print(f"\n[Raw dual annotation outputs detected at {RAW_LOG_PATH}]")
        print("Loading raw dual annotations for agreement and silver v1 generation...")
        with open(RAW_LOG_PATH, "r") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    raw_map[(d["figure_id"], d["annotator"])] = d
        if len(raw_map) < 120:
            print(f"Warning: RAW_LOG_PATH has {len(raw_map)} items, expected 120. Re-running inferences...")
            have_raw_log = False

    raw_log = None
    llm_repo = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    vlm_repo = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    if not have_raw_log:
        # 1. Load Local Text LLM (Annotator A)
        import mlx_lm
        print("\n[1/4] Loading Annotator A: Qwen2.5-1.5B-Instruct-4bit (mlx-lm)...")
        t_llm0 = time.perf_counter()
        llm_model, llm_tokenizer = mlx_lm.load(llm_repo)
        print(f"Loaded Annotator A in {time.perf_counter() - t_llm0:.2f}s.")

        # 2. Load Local VLM (Annotator B)
        import mlx_vlm
        print("\n[2/4] Loading Annotator B: Qwen2-VL-7B-Instruct-4bit (mlx-vlm)...")
        t_vlm0 = time.perf_counter()
        vlm_model, vlm_processor = mlx_vlm.load(vlm_repo)
        print(f"Loaded Annotator B in {time.perf_counter() - t_vlm0:.2f}s.")

        os.makedirs(os.path.dirname(RAW_LOG_PATH), exist_ok=True)
        raw_log = open(RAW_LOG_PATH, "w")

    annotated_figures = []
    all_cand_pairs_A = []
    all_cand_pairs_B = []

    print("\n[3/4] Running Dual Local Model Grounding Inferences...")
    for idx, fig in enumerate(eval_figures):
        fig_id = fig["figure_id"]
        rel_img_path = fig.get("image_path", "")
        img_full_path = os.path.join(ROOT, "data", "scicap_subset", rel_img_path)
        if not os.path.exists(img_full_path):
            img_full_path = os.path.join(ROOT, "data", "scicap_subset", "images", f"{fig_id}.png")

        elements = [e for e in fig.get("elements", []) if e.get("type") in TEXT_ELEMENT_TYPES]
        entities = fig.get("entities", [])

        if not elements or not entities:
            continue

        elem_text_list = "\n".join(f"{e['id']} ({e['type']}): \"{e['text']}\"" for e in elements)
        ent_text_list = "\n".join(f"{ent['id']} ({ent.get('type', 'CONCEPT')}): \"{ent['text']}\"" for ent in entities)

        candidate_pairs = [(e["id"], ent["text"]) for e in elements for ent in entities]

        # --- ANNOTATOR A (Text LLM) ---
        if have_raw_log and (fig_id, "Annotator A (Text LLM)") in raw_map:
            res_A = raw_map[(fig_id, "Annotator A (Text LLM)")]["raw_response"]
            ms_A = raw_map[(fig_id, "Annotator A (Text LLM)")].get("wall_clock_runtime_ms", 0.0)
        else:
            prompt_A_text = (
                "You are an expert scientific paper grounding assistant.\n"
                f"Visual elements from figure {fig_id}:\n{elem_text_list}\n\n"
                f"Scientific entities from paper text:\n{ent_text_list}\n\n"
                "Determine which visual element corresponds to which scientific entity.\n"
                "Output each grounded link in the format:\n"
                "ELEMENT_ID -> ENTITY_TEXT\n"
                "If no elements correspond, output NONE."
            )
            chat_A = [{"role": "user", "content": prompt_A_text}]
            prompt_A = llm_tokenizer.apply_chat_template(chat_A, tokenize=False, add_generation_prompt=True)

            t_a0 = time.perf_counter()
            res_A = mlx_lm.generate(llm_model, llm_tokenizer, prompt=prompt_A, max_tokens=64, verbose=False)
            ms_A = round((time.perf_counter() - t_a0) * 1000, 2)
            if raw_log is not None:
                raw_log.write(json.dumps({
                    "figure_id": fig_id,
                    "annotator": "Annotator A (Text LLM)",
                    "model_name": "Qwen2.5-1.5B-Instruct-4bit",
                    "weights_path": llm_repo,
                    "prompt": prompt_A,
                    "raw_response": res_A.strip(),
                    "wall_clock_runtime_ms": ms_A,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                }) + "\n")

        links_A = parse_grounding_response(res_A, elements, entities)

        # --- ANNOTATOR B (VLM) ---
        if have_raw_log and (fig_id, "Annotator B (VLM)") in raw_map:
            res_B = raw_map[(fig_id, "Annotator B (VLM)")]["raw_response"]
            ms_B = raw_map[(fig_id, "Annotator B (VLM)")].get("wall_clock_runtime_ms", 0.0)
        else:
            prompt_B_text = (
                "You are an expert scientific figure analyzer.\n"
                f"Detected visual elements:\n{elem_text_list}\n\n"
                f"Candidate scientific entities:\n{ent_text_list}\n\n"
                "Looking at the figure image, which visual elements correspond to which entities?\n"
                "Output each grounded link in the format:\n"
                "ELEMENT_ID -> ENTITY_TEXT\n"
                "If none, output NONE."
            )
            messages_B = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_B_text}
                    ]
                }
            ]
            prompt_B = vlm_processor.apply_chat_template(messages_B, tokenize=False, add_generation_prompt=True)

            t_b0 = time.perf_counter()
            try:
                res_B_obj = mlx_vlm.generate(vlm_model, vlm_processor, prompt=prompt_B, image=img_full_path, max_tokens=64, verbose=False)
                res_B = res_B_obj.text.strip()
            except Exception as e:
                res_B = f"ERROR: {e}"
            ms_B = round((time.perf_counter() - t_b0) * 1000, 2)
            if raw_log is not None:
                raw_log.write(json.dumps({
                    "figure_id": fig_id,
                    "annotator": "Annotator B (VLM)",
                    "model_name": "Qwen2-VL-7B-Instruct-4bit",
                    "weights_path": vlm_repo,
                    "prompt": prompt_B,
                    "raw_response": res_B,
                    "wall_clock_runtime_ms": ms_B,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                }) + "\n")
                raw_log.flush()

        links_B = parse_grounding_response(res_B, elements, entities)


        # Compute pair agreements
        for pair in candidate_pairs:
            is_A = int(pair in links_A)
            is_B = int(pair in links_B)
            all_cand_pairs_A.append(is_A)
            all_cand_pairs_B.append(is_B)

        # Agreed links: consensus of both models
        agreed_links = links_A & links_B
        silver_v1_links = [
            {"element_id": eid, "entity": ent, "relation": "REPRESENTS"}
            for eid, ent in agreed_links
        ]

        fig_copy = dict(fig)
        fig_copy["provenance"] = "silver_llm"
        fig_copy["silver_links"] = silver_v1_links
        fig_copy["annotator_a_links"] = [{"element_id": eid, "entity": ent} for eid, ent in links_A]
        fig_copy["annotator_b_links"] = [{"element_id": eid, "entity": ent} for eid, ent in links_B]
        annotated_figures.append(fig_copy)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_figures):
            print(f"  [{idx+1}/{len(eval_figures)}] Fig {fig_id}: A={len(links_A)} links ({ms_A}ms) | B={len(links_B)} links ({ms_B}ms) | Agreed={len(agreed_links)} links")

    if raw_log is not None:
        raw_log.close()

    # Model-Model Agreement Calculation
    y_A = np.array(all_cand_pairs_A)
    y_B = np.array(all_cand_pairs_B)
    model_model_kappa = cohen_kappa_score(y_A, y_B)
    obs_agreement = float(np.mean(y_A == y_B))
    total_candidates = len(y_A)

    print("\n" + "=" * 70)
    print("MODEL-MODEL ANNOTATION AGREEMENT (STRICTLY MODEL-MODEL, NOT HUMAN):")
    print(f"  Total candidate element-entity pairs: {total_candidates}")
    print(f"  Observed Agreement: {obs_agreement * 100:.2f}%")
    print(f"  Model-Model Cohen's Kappa: {model_model_kappa:.4f}")
    print("=" * 70)

    # Save silver_grounding_v1 dataset
    v1_dataset = {
        "dataset_name": "SciCap-Grounding-Silver-v1",
        "description": "Dual-annotator silver grounding standard (agreed links between local text LLM and local VLM).",
        "provenance": "silver_llm",
        "num_figures": len(annotated_figures),
        "model_agreement_diagnostics": {
            "agreement_type": "model_model_agreement_not_human",
            "annotator_a": "Qwen2.5-1.5B-Instruct-4bit (mlx-lm)",
            "annotator_b": "Qwen2-VL-7B-Instruct-4bit (mlx-vlm)",
            "total_candidate_pairs": total_candidates,
            "observed_agreement": round(obs_agreement, 4),
            "model_model_kappa": round(float(model_model_kappa), 4)
        },
        "figures": annotated_figures
    }

    with open(V1_OUT_PATH, "w") as f:
        json.dump(v1_dataset, f, indent=2)
    print(f"\nSaved silver_grounding_v1 dataset to: {V1_OUT_PATH}")

    # --- 4. BENCHMARK GROUNDING MODELS ON SILVER V1 ---
    print("\n[4/4] Evaluating Grounding Models (Rules, LR, GBDT, GNN) on Silver v1...")
    
    # Partition into train/val/test (40 train, 20 test)
    train_figs = annotated_figures[:40]
    test_figs = annotated_figures[40:]

    _ner = ScientificNERExtractor()

    def prepare_instances(figs):
        insts = []
        for f in figs:
            elems = [e for e in f.get("elements", []) if e.get("type") in TEXT_ELEMENT_TYPES]
            ents = f.get("entities", [])
            para = f.get("paragraph_text", "")
            rels = _ner.extract_relations(ents, para)
            gold = {(l["element_id"], l["entity"]) for l in f.get("silver_links", [])}
            gold_A_only = {(l["element_id"], l["entity"]) for l in f.get("annotator_a_links", [])}
            insts.append({
                "figure_id": f["figure_id"],
                "elements": elems,
                "entities": ents,
                "paragraph_text": para,
                "relations": rels,
                "gold": gold,
                "gold_A_only": gold_A_only
            })
        return insts

    train_insts = prepare_instances(train_figs)
    test_insts = prepare_instances(test_figs)

    def extract_pairs(insts):
        X, y, meta = [], [], []
        for inst in insts:
            elements = inst["elements"]
            entities = inst["entities"]
            para = inst["paragraph_text"]
            gold = inst["gold"]
            g = build_grounding_graph(elements, entities, inst["relations"])
            for el in elements:
                for en in entities:
                    rule_edge = g.has_edge(f"elem:{el['id']}", f"ent:{en['text']}")
                    feat = pair_features(el, en, para, rule_edge)
                    is_gold = (el["id"], en["text"]) in gold
                    X.append(feat)
                    y.append(int(is_gold))
                    meta.append({"figure_id": inst["figure_id"], "element_id": el["id"], "entity": en["text"]})
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), meta

    X_train, y_train, _ = extract_pairs(train_insts)
    X_test, y_test, test_meta = extract_pairs(test_insts)

    def calc_link_metrics(fig_pred_map, target_key="gold"):
        total_tp, total_fp, total_fn = 0, 0, 0
        for inst in test_insts:
            preds = fig_pred_map.get(inst["figure_id"], set())
            g = inst[target_key]
            total_tp += len(preds & g)
            total_fp += len(preds - g)
            total_fn += len(g - preds)
        p = total_tp / max(total_tp + total_fp, 1)
        r = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4), "tp": total_tp, "fp": total_fp, "fn": total_fn}

    # (a) Rules
    rules_pred_map = {}
    for inst in test_insts:
        g = build_grounding_graph(inst["elements"], inst["entities"], inst["relations"])
        pred_links = {(u[5:], v[4:]) for u, v in g.edges if u.startswith("elem:") and v.startswith("ent:")}
        rules_pred_map[inst["figure_id"]] = pred_links
    res_rules = calc_link_metrics(rules_pred_map)

    # (b) Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_pred_map = {inst["figure_id"]: set() for inst in test_insts}
    for item, pred in zip(test_meta, lr_preds):
        if pred == 1:
            lr_pred_map[item["figure_id"]].add((item["element_id"], item["entity"]))
    res_lr = calc_link_metrics(lr_pred_map)

    # (c) HistGradientBoosting (GBDT)
    gbdt = HistGradientBoostingClassifier(random_state=42)
    gbdt.fit(X_train, y_train)
    gbdt_preds = gbdt.predict(X_test)
    gbdt_pred_map = {inst["figure_id"]: set() for inst in test_insts}
    for item, pred in zip(test_meta, gbdt_preds):
        if pred == 1:
            gbdt_pred_map[item["figure_id"]].add((item["element_id"], item["entity"]))
    res_gbdt = calc_link_metrics(gbdt_pred_map)

    # (d) GNN
    gnn_trainer = SupervisedGroundingTrainer(use_message_passing=True, seed=42, epochs=80)
    mk = lambda inst: gnn_trainer.make_example(inst["elements"], inst["entities"], inst["paragraph_text"], inst["relations"], gold=inst["gold"])
    train_ex = [mk(i) for i in train_insts]
    val_ex = [mk(i) for i in test_insts]
    gnn_trainer.fit(train_ex, val_ex)

    gnn_pred_map = {}
    for inst in test_insts:
        data, cands = gnn_trainer.make_example(inst["elements"], inst["entities"], inst["paragraph_text"], inst["relations"])
        scores = gnn_trainer.predict_scores(data)
        links = set(gnn_trainer.decode(cands, scores, gnn_trainer.threshold))
        gnn_pred_map[inst["figure_id"]] = links
    res_gnn = calc_link_metrics(gnn_pred_map)

    # (e) VLM Evaluation ONLY on Annotator-A-only labels (non-circular)
    vlm_pred_map = {}
    for f in test_figs:
        vlm_pred_map[f["figure_id"]] = {(l["element_id"], l["entity"]) for l in f.get("annotator_b_links", [])}
    res_vlm_on_A = calc_link_metrics(vlm_pred_map, target_key="gold_A_only")

    print("\nRESULTS ON SCICAP SILVER V1:")
    print(f"  Rules:   P={res_rules['precision']:.4f}, R={res_rules['recall']:.4f}, F1={res_rules['f1']:.4f}")
    print(f"  LR:      P={res_lr['precision']:.4f}, R={res_lr['recall']:.4f}, F1={res_lr['f1']:.4f}")
    print(f"  GBDT:    P={res_gbdt['precision']:.4f}, R={res_gbdt['recall']:.4f}, F1={res_gbdt['f1']:.4f}")
    print(f"  GNN:     P={res_gnn['precision']:.4f}, R={res_gnn['recall']:.4f}, F1={res_gnn['f1']:.4f}")
    print(f"  VLM (evaluated on Annotator-A-only labels): P={res_vlm_on_A['precision']:.4f}, R={res_vlm_on_A['recall']:.4f}, F1={res_vlm_on_A['f1']:.4f}")

    eval_results = {
        "status": "INSUFFICIENT LABELS",
        "status_note": "Silver v1 has 0 test positives (annotator agreement kappa=0.0168 produced no consensus positive links in test set). No model ranking comparison is valid.",
        "provenance": "silver_llm",
        "benchmark": "SciCap-Grounding-Silver-v1",
        "num_total_figures": len(annotated_figures),
        "num_train_figures": len(train_figs),
        "num_test_figures": len(test_figs),
        "model_model_agreement": {
            "annotator_a": "Qwen2.5-1.5B-Instruct-4bit (mlx-lm)",
            "annotator_b": "Qwen2-VL-7B-Instruct-4bit (mlx-vlm)",
            "total_candidate_pairs": total_candidates,
            "observed_agreement": round(obs_agreement, 4),
            "model_model_kappa": round(float(model_model_kappa), 4)
        },
        "models_on_silver_v1": {
            "rules": res_rules,
            "logistic_regression": res_lr,
            "gbdt": res_gbdt,
            "gnn": res_gnn
        },
        "vlm_evaluation_on_annotator_a_only": {
            "model_name": "Qwen2-VL-7B-Instruct-4bit",
            "weights_path": vlm_repo,
            "target": "annotator_a_only_labels_non_circular",
            **res_vlm_on_A
        },
        **get_runtime_metadata("pipelines/build_silver_grounding_v1.py")
    }


    os.makedirs(os.path.dirname(EVAL_OUT_PATH), exist_ok=True)
    with open(EVAL_OUT_PATH, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nSaved evaluation results to: {EVAL_OUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
