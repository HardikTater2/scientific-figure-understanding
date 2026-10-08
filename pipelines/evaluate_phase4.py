"""
Phase 4 evaluation: 3D figure detection and figure <-> 3D geometry alignment on SciFig-Controlled.

Trains (on the train split) and evaluates (on the held-out test split):
  1. Figure-type / 3D detector (CNN image classifier), compared with the text-keyword detector.
  2. Figure image <-> geometry contrastive alignment (PointNet and multi-view encoders), compared
     with identically initialised but untrained encoders and with chance.

Writes checkpoints to models/checkpoints/ and results to results/phase4_results.json.
All numbers are measured in this run; nothing is filled in by hand.

Usage:
    python pipelines/evaluate_phase4.py [--benchmark data/benchmark] [--seed 42]
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.benchmark_generator import load_benchmark
from src.models.vision.cross_modal_3d import (
    CrossModal3DAligner, evaluate_alignment, evaluate_figure_classifier, save_checkpoint,
    train_alignment, train_figure_classifier,
)
from src.vision.threed_detector import ThreeDVisualizationDetector


VAL_MOLECULES = {"1A28", "2XHE"}


def hardware_info():
    return {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "torch": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu"}


def keyword_detector_eval(figs):
    det = ThreeDVisualizationDetector(checkpoint_path="")  # text cues only
    pred = np.array([det.detect_3d_content("", f["caption"], f["paragraph_text"])["is_3d"] for f in figs])
    gold = np.array([f["is_3d"] for f in figs])
    tp, fp, fn = int((pred & gold).sum()), int((pred & ~gold).sum()), int((~pred & gold).sum())
    return {"is_3d_accuracy": round(float((pred == gold).mean()), 4),
            "is_3d_precision": round(tp / max(tp + fp, 1), 4),
            "is_3d_recall": round(tp / max(tp + fn, 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classifier-epochs", type=int, default=25)
    ap.add_argument("--alignment-epochs", type=int, default=60)
    ap.add_argument("--out", default="results/phase4_results.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    meta = load_benchmark(a.benchmark)
    root = meta["root"]
    train = [f for f in meta["figures"] if f["split"] == "train"]
    test = [f for f in meta["figures"] if f["split"] == "test"]

    print("[1/2] Training figure-type classifier ...")
    clf, clf_train = train_figure_classifier(root, train, epochs=a.classifier_epochs, seed=a.seed)
    save_checkpoint(clf, clf_train, "models/checkpoints/figure_classifier.pt")
    clf_test = evaluate_figure_classifier(clf, root, test)
    kw_test = keyword_detector_eval(test)
    print(f"  CNN   is_3d acc={clf_test['is_3d_accuracy']}  type acc={clf_test['figure_type_accuracy']}")
    print(f"  Words is_3d acc={kw_test['is_3d_accuracy']}")

    print("[2/2] Training figure <-> geometry alignment ...")
    # Model selection on a validation split carved from train (val molecules are structure-disjoint).
    vrng = np.random.default_rng(1)
    is_val = [(f.get("pdb_id") in VAL_MOLECULES) if f["figure_type"] == "protein" else bool(vrng.random() < 0.2)
              for f in train]
    fit = [f for f, v in zip(train, is_val) if not v]
    val = [f for f, v in zip(train, is_val) if v]
    selection = {}
    for aug in (True, False):
        m, _ = train_alignment(root, fit, epochs=a.alignment_epochs, seed=a.seed, augment_rotation=aug)
        selection[f"rotation_augmentation={aug}"] = evaluate_alignment(m, root, val)["results"]["pointnet+multiview"]
        print(f"  val (rotation_aug={aug}): {selection[f'rotation_augmentation={aug}']}")
    best_aug = max((True, False), key=lambda g: selection[f"rotation_augmentation={g}"]["mrr"])
    torch.manual_seed(a.seed)
    untrained = CrossModal3DAligner()
    base = evaluate_alignment(untrained, root, test)
    aligner, al_train = train_alignment(root, train, epochs=a.alignment_epochs, seed=a.seed, augment_rotation=best_aug)
    save_checkpoint(aligner, al_train, "models/checkpoints/cross_modal_3d.pt")
    al_test = evaluate_alignment(aligner, root, test)
    for k, v in al_test["results"].items():
        print(f"  {k:<20} R@1={v['recall@1']} R@5={v['recall@5']} MRR={v['mrr']}  "
              f"(untrained R@1={base['results'][k]['recall@1']}, chance={al_test['chance_recall@1']})")

    report = {
        "benchmark": {"name": meta["dataset_name"], "version": meta["version"], "path": root,
                      "generator_seed": meta["seed"], "num_train": len(train), "num_test": len(test),
                      "test_molecules_structure_disjoint": meta["test_molecules"]},
        "seed": a.seed,
        "hardware": hardware_info(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "figure_classifier": {"architecture": "4-block CNN (16-32-64-64), 128x128 RGB input, 7 classes",
                              "training": clf_train, "test": clf_test,
                              "text_keyword_baseline_test": kw_test},
        "geometry_alignment": {
            "architecture": {"image": "FigureImageEncoder (CNN) -> 64-d",
                             "point_cloud": "PointNet (vanilla, 1024 points) -> 64-d",
                             "multi_view": "6 orthographic views (occupancy/near/far depth) -> shared CNN -> max-pool -> 64-d",
                             "loss": "multi-positive InfoNCE, image<->pointnet + image<->multiview"},
            "model_selection_on_validation": {"val_molecules": sorted(VAL_MOLECULES), "num_fit": len(fit),
                                              "num_val": len(val), "criterion": "val MRR (pointnet+multiview)",
                                              "candidates": selection, "selected_rotation_augmentation": best_aug},
            "training": al_train, "test_trained": al_test, "test_untrained_baseline": base},
        "caveats": [
            "SciFig-Controlled is synthetic (matplotlib renders); in-domain accuracy does not transfer to real paper figures.",
            "Protein test structures are disjoint from training structures; surfaces and 3D scatters are new random instances of the same parametric families.",
            "No pretrained vision weights were available (HuggingFace / download.pytorch.org unreachable); all encoders are trained from scratch.",
        ],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Results written to {a.out}")


if __name__ == "__main__":
    main()
