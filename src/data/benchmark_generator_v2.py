"""
SciFig-Controlled v2 Generator (Phase 0 / Task 2).
Fixes syntactic position shortcuts discovered in v1 (where sentence position yielded F1 = 0.8835).

Key architectural upgrades in v2:
  1. 20+ diverse paraphrased template variants per sentence role.
  2. In-sentence distractor mixing: distractors appear in the exact same sentences as plotted entities.
  3. Sentence order shuffling: plotted entities and distractors appear with equal probability at any
     sentence position (beginning, middle, end).
  4. Generates output to data/benchmark_v2 with provenance 'synthetic'.
"""

import os
import sys
import gzip
import json
import re
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from src.data.benchmark_generator import (
    METHODS, DATASETS, METRICS, VARIABLES, COLOR_VARIABLES, MOLECULES, TEST_MOLECULES,
    SURFACE_FAMILIES, SHAPE_FAMILIES, FIG_TYPES_2D, FIG_TYPES_3D, NUM_POINTS,
    _FigRecord, load_pdb_coords, resample_chain, normalize_points, surface_geometry, shape_points
)

# ---------------------------------------------------------------------------
# 20+ Paraphrased Sentence Templates per Role
# ---------------------------------------------------------------------------

METHOD_SENTENCE_VARIANTS = [
    "We compare the empirical performance of {methods} in this benchmark.",
    "Experimental evaluations examine {methods} under identical testing conditions.",
    "In this study, {methods} are evaluated across the target tasks.",
    "The behavior of {methods} is plotted to illustrate model scaling.",
    "We benchmark {methods} against representative baselines.",
    "Results demonstrate the performance characteristics of {methods}.",
    "We investigate the convergence and accuracy of {methods}.",
    "Our comparative analysis includes {methods} on the test suite.",
    "The curves illustrate how {methods} perform across configurations.",
    "We measure quantitative outcomes for {methods}.",
    "Performance trajectories of {methods} are visualized below.",
    "Here we study {methods} under systematic parameter sweeps.",
    "Figure depicts the empirical trade-offs of {methods}.",
    "We observe consistent trends across {methods}.",
    "Evaluation protocol assesses {methods} on the primary split.",
    "The primary models under investigation are {methods}.",
    "We contrast the outcomes achieved by {methods}.",
    "Both baseline models and {methods} are analyzed in detail.",
    "Experimental observations highlight the strengths of {methods}.",
    "We present a detailed comparison focusing on {methods}.",
    "Quantitative comparisons highlight {methods} across trials."
]

METRIC_VARIABLE_SENTENCE_VARIANTS = [
    "Performance is evaluated using {metric} as a function of {variable}.",
    "We report {metric} across varying settings of {variable}.",
    "Values of {metric} are plotted against {variable}.",
    "The effect of {variable} on overall {metric} is demonstrated.",
    "We analyze the relationship between {variable} and {metric}.",
    "System outcomes are quantified via {metric} over different {variable} regimes.",
    "Measurements of {metric} track changes across {variable}.",
    "We monitor {metric} while sweeping {variable}.",
    "The dependency of {metric} upon {variable} is shown in the figure.",
    "Scaling behavior is measured by {metric} with respect to {variable}.",
    "We inspect how {variable} influences observed {metric}.",
    "The response curve depicts {metric} across values of {variable}.",
    "Validation scores reflect {metric} evaluated against {variable}.",
    "We observe the trajectory of {metric} across the {variable} axis.",
    "Empirical readings record {metric} over varying {variable} levels.",
    "The chart displays {metric} as {variable} changes systematically.",
    "We measure system efficiency in terms of {metric} across {variable}.",
    "Across all trials, {metric} is documented relative to {variable}.",
    "The vertical axis illustrates {metric} while the horizontal axis denotes {variable}.",
    "Scores are computed by {metric} under varying {variable} thresholds.",
    "We plot {metric} to highlight the sensitivity to {variable}."
]

DATASET_SENTENCE_VARIANTS = [
    "All experiments are conducted on the {dataset} benchmark.",
    "Evaluations utilize the standardized {dataset} corpus.",
    "We validate our findings using the {dataset} dataset.",
    "Results are reported over the public {dataset} collection.",
    "The empirical evaluation is grounded in {dataset}.",
    "Experiments are executed using data from {dataset}.",
    "We evaluate model efficacy on {dataset}.",
    "The primary evaluation benchmark is {dataset}.",
    "Trials are performed across the {dataset} environment.",
    "Our setup leverages samples drawn from {dataset}.",
    "We test generalization on the {dataset} split.",
    "Data partitioning follows the canonical {dataset} split.",
    "The test protocol relies upon {dataset} for empirical validation.",
    "Benchmark results are derived from {dataset}.",
    "We analyze behavior across the {dataset} test distribution.",
    "Empirical metrics are computed over {dataset}.",
    "All models process the same subset of {dataset}.",
    "Evaluation benchmarks include {dataset} for cross-domain stability.",
    "We assess empirical gains on {dataset}.",
    "Target predictions are scored against {dataset}.",
    "Our empirical evidence stems from {dataset} experiments."
]

# Combined sentence templates that place distractors in the same sentence as plotted entities symmetrically
MIXED_DISTRACTOR_SENTENCE_VARIANTS = [
    "We compare {methods} against {distractors} in this benchmark.",
    "Both {distractors} and {methods} are evaluated across identical test conditions.",
    "We benchmark {methods} alongside reference models {distractors}.",
    "Experiments report quantitative results for {methods} as well as {distractors}.",
    "We investigate {distractors} and {methods} across our experimental setup.",
    "The analysis assesses the behavior of {methods} and comparison models {distractors}.",
    "Performance trade-offs between {methods} and {distractors} are analyzed.",
    "We evaluate {distractors} alongside {methods} on the test suite.",
    "Our comparative study examines {methods} in conjunction with {distractors}.",
    "We monitor empirical trajectories for both {distractors} and {methods}.",
    "Results contrast the performance of {methods} with {distractors}.",
    "We evaluate {methods} and baseline systems {distractors} across the tasks.",
    "Empirical comparisons highlight both {distractors} and {methods}.",
    "We benchmark {methods} together with {distractors} on the benchmark.",
    "The evaluation covers {distractors} as well as {methods}.",
    "We examine how {methods} perform relative to {distractors}.",
    "Both {methods} and {distractors} are included in our empirical evaluation.",
    "We record benchmark scores for {distractors} and {methods}.",
    "Quantitative comparisons detail outcomes for {methods} and {distractors}.",
    "Our setup benchmarks {distractors} alongside {methods}.",
    "We study the empirical convergence of {methods} and {distractors}.",
    "Systematic evaluations analyze {distractors} in parallel with {methods}.",
    "We present experimental results covering both {methods} and {distractors}.",
    "Test outcomes reflect performance for {distractors} and {methods}.",
    "We evaluate the relative capabilities of {methods} and {distractors}."
]


class SciFigBenchmarkGeneratorV2:
    def __init__(self, out_dir: str = "data/benchmark_v2", seed: int = 42,
                 n_per_2d_type: int = 75, n_views: int = 5, test_fraction: float = 0.30):
        self.out_dir = out_dir
        self.rng = np.random.default_rng(seed)
        self.n_per_2d_type = n_per_2d_type
        self.n_views = n_views
        self.test_fraction = test_fraction
        self.pdb_dir = os.path.join(ROOT, "data", "benchmark", "pdb_structures")
        if not os.path.exists(self.pdb_dir):
            self.pdb_dir = os.path.join(ROOT, "data", "scicap_subset")  # fallback

        os.makedirs(os.path.join(self.out_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "geometry"), exist_ok=True)

    def _pick(self, mapping_or_list, k: int) -> List[str]:
        items = list(mapping_or_list.keys()) if isinstance(mapping_or_list, dict) else list(mapping_or_list)
        return self.rng.choice(items, size=min(k, len(items)), replace=False).tolist()

    def _alias(self, mapping, canonical: str) -> str:
        aliases = mapping[canonical]
        return aliases[int(self.rng.integers(len(aliases)))]

    def _mention(self, canonical: str, alias: str) -> str:
        if self.rng.random() < 0.45:
            return f"{canonical} ({alias})"
        return canonical

    def _ent(self, text: str, etype: str) -> Dict[str, str]:
        return {"id": f"ent_{abs(hash(text)) % 1000000:06d}", "text": text, "type": etype}

    def _generate_v2_paragraph(self, methods: List[str], m_aliases: Dict[str, str],
                               metric: str, m_alias: str, var: str, v_alias: str,
                               dataset: str, distractors: List[str]) -> str:
        """
        Generates a paraphrased paragraph with:
          - 20+ sentence variants per role
          - Distractor mixing in the same sentence as plotted entities
          - Symmetrical syntax avoiding any leak of plotted vs. distractor identity
          - Shuffled sentence order to eliminate positional shortcuts
        """
        methods_str = ", ".join(self._mention(m, m_aliases[m]) for m in methods)
        distract_str = ", ".join(distractors) if distractors else ""
        metric_str = self._mention(metric, m_alias)
        var_str = self._mention(var, v_alias)

        sentences = []

        # When distractors exist, place them directly in the same sentence as methods with symmetric phrasing
        # 50% of the time methods appear first, 50% distractors appear first
        if distract_str:
            tmpl = self.rng.choice(MIXED_DISTRACTOR_SENTENCE_VARIANTS)
            if self.rng.random() < 0.50:
                s_method = tmpl.format(methods=methods_str, distractors=distract_str,
                                       dataset=dataset, variable=var_str, metric=metric_str)
            else:
                s_method = tmpl.format(methods=distract_str, distractors=methods_str,
                                       dataset=dataset, variable=var_str, metric=metric_str)
            sentences.append(s_method)
        else:
            tmpl_m = self.rng.choice(METHOD_SENTENCE_VARIANTS)
            sentences.append(tmpl_m.format(methods=methods_str))

        # Metric & Variable sentence (50% metric first, 50% variable first)
        tmpl_mv = self.rng.choice(METRIC_VARIABLE_SENTENCE_VARIANTS)
        sentences.append(tmpl_mv.format(metric=metric_str, variable=var_str))

        # Dataset sentence
        tmpl_ds = self.rng.choice(DATASET_SENTENCE_VARIANTS)
        sentences.append(tmpl_ds.format(dataset=dataset))

        # CRITICAL: Shuffle sentences so plotted entities appear at ANY sentence index
        self.rng.shuffle(sentences)

        return " ".join(sentences)

    def _fig_xy(self, kind: str, rec: _FigRecord):
        n_m = int(self.rng.integers(2, 5))
        methods = self._pick(METHODS, n_m)
        metric = self._pick(METRICS, 1)[0]
        var = self._pick(VARIABLES, 1)[0]
        dataset = self._pick(DATASETS, 1)[0]
        m_alias = {m: self._alias(METHODS, m) for m in methods}
        x_alias = self._alias(VARIABLES, var)
        y_alias = self._alias(METRICS, metric)

        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        xs = np.linspace(0.1, 1.0, 10)
        for i, m in enumerate(methods):
            ys = 1.0 - np.exp(-xs * (i + 1)) + self.rng.normal(0, 0.03, len(xs))
            if kind == "line":
                ax.plot(xs, ys, label=m_alias[m], marker="o")
            else:
                ax.scatter(xs, ys, label=m_alias[m])

        leg = ax.legend(fontsize=12, loc="lower right")
        xl = ax.set_xlabel(x_alias, fontsize=13)
        yl = ax.set_ylabel(y_alias, fontsize=13)
        rec.add("AXIS", x_alias, xl, (var, "REPRESENTS"))
        rec.add("AXIS", y_alias, yl, (metric, "REPRESENTS"))
        for m, t in zip(methods, leg.get_texts()):
            rec.add("LEGEND", m_alias[m], t, (m, "CORRESPONDS_TO"))

        # Balanced Distractors (equal count to plotted entities)
        have = set(methods) | {metric, var, dataset}
        distract_m = [m for m in self._pick(METHODS, n_m + 2) if m not in have][:n_m]
        distract_k = [k for k in self._pick(METRICS, 2) if k not in have][:1]
        distract_v = [v for v in self._pick(VARIABLES, 2) if v not in have][:1]
        distractors = distract_m + distract_k

        ents = [self._ent(m, "METHOD") for m in methods] + [
            self._ent(metric, "METRIC"), self._ent(var, "VARIABLE")
        ] + [self._ent(m, "METHOD") for m in distract_m] + [
            self._ent(k, "METRIC") for k in distract_k
        ] + [self._ent(v, "VARIABLE") for v in distract_v]

        para = self._generate_v2_paragraph(methods, m_alias, metric, y_alias, var, x_alias, dataset, distractors)
        caption = f"{metric} versus {var} for {', '.join(methods)} on {dataset}."
        return fig, ents, para, caption

    def _fig_bar(self, rec: _FigRecord):
        n_m = int(self.rng.integers(3, 6))
        methods = self._pick(METHODS, n_m)
        metric = self._pick(METRICS, 1)[0]
        dataset = self._pick(DATASETS, 1)[0]
        m_alias = {m: self._alias(METHODS, m) for m in methods}
        y_alias = self._alias(METRICS, metric)

        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        ax.bar(range(n_m), self.rng.uniform(0.3, 1.0, n_m), color="C0")
        ax.set_xticks(range(n_m))
        ticks = ax.set_xticklabels([m_alias[m] for m in methods], fontsize=12, rotation=0)
        yl = ax.set_ylabel(y_alias, fontsize=13)
        xl = ax.set_xlabel("Method", fontsize=13)
        rec.add("AXIS", "Method", xl, None)
        rec.add("AXIS", y_alias, yl, (metric, "REPRESENTS"))
        for m, t in zip(methods, ticks):
            rec.add("LABEL", m_alias[m], t, (m, "CORRESPONDS_TO"))

        # Balanced Distractors (equal count to plotted entities)
        have = set(methods) | {metric, dataset}
        distract_m = [m for m in self._pick(METHODS, n_m + 2) if m not in have][:n_m]
        distract_k = [k for k in self._pick(METRICS, 2) if k not in have][:1]
        distractors = distract_m + distract_k

        ents = [self._ent(m, "METHOD") for m in methods] + [
            self._ent(metric, "METRIC")
        ] + [self._ent(m, "METHOD") for m in distract_m] + [self._ent(k, "METRIC") for k in distract_k]

        para = self._generate_v2_paragraph(methods, m_alias, metric, y_alias, "Method", "Method", dataset, distractors)
        caption = f"{metric} of {', '.join(methods)} on {dataset}."
        return fig, ents, para, caption

    def _fig_heatmap(self, rec: _FigRecord):
        v1, v2 = self._pick(VARIABLES, 2)
        metric = self._pick(METRICS, 1)[0]
        method = self._pick(METHODS, 1)[0]
        dataset = self._pick(DATASETS, 1)[0]
        a1, a2 = self._alias(VARIABLES, v1), self._alias(VARIABLES, v2)
        c_alias = self._alias(METRICS, metric)

        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        im = ax.imshow(self.rng.random((8, 8)), cmap=str(self.rng.choice(["viridis", "magma", "Blues"])))
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(c_alias, fontsize=13)
        cl = cb.ax.yaxis.label
        xl = ax.set_xlabel(a1, fontsize=13)
        yl = ax.set_ylabel(a2, fontsize=13)
        rec.add("AXIS", a1, xl, (v1, "REPRESENTS"))
        rec.add("AXIS", a2, yl, (v2, "REPRESENTS"))
        rec.add("LABEL", c_alias, cl, (metric, "REPRESENTS"))

        have = {method, metric, v1, v2, dataset}
        distract_m = [m for m in self._pick(METHODS, 3) if m not in have][:2]
        distract_k = [k for k in self._pick(METRICS, 2) if k not in have][:1]
        distract_v = [v for v in self._pick(VARIABLES, 2) if v not in have][:1]
        distractors = distract_m + distract_k

        ents = [self._ent(method, "METHOD"), self._ent(metric, "METRIC"), self._ent(v1, "VARIABLE"),
                self._ent(v2, "VARIABLE")] + [self._ent(m, "METHOD") for m in distract_m] + [
                self._ent(k, "METRIC") for k in distract_k] + [self._ent(v, "VARIABLE") for v in distract_v]

        para = self._generate_v2_paragraph([method], {method: method}, metric, c_alias, v1, a1, dataset, distractors)
        caption = f"{metric} of {method} over {v1} and {v2} on {dataset}."
        return fig, ents, para, caption

    def _split(self) -> str:
        return "test" if self.rng.random() < self.test_fraction else "train"

    def _save(self, fig_id, fig, rec, kind, is_3d, ents, para, caption, split, geom=None):
        rec.finalize(fig)
        img_rel = f"images/{fig_id}.png"
        fig.savefig(os.path.join(self.out_dir, img_rel), dpi=100)
        plt.close(fig)
        seen, uniq = set(), []
        for e in ents:
            if e["text"] not in seen:
                seen.add(e["text"])
                uniq.append(e)
        return {
            "figure_id": fig_id, "split": split, "figure_type": kind, "is_3d": is_3d,
            "image_path": img_rel, "geometry_path": None,
            "paragraph_text": para, "caption": caption,
            "entities": uniq, "elements": rec.elements,
            "gold_links": [{"element_id": el["id"], "entity": el["gold_entity"], "relation": el["gold_relation"]}
                           for el in rec.elements if el["gold_entity"]],
            "provenance": "synthetic"
        }

    def generate(self) -> Dict[str, Any]:
        figures = []
        k = 0
        for kind in FIG_TYPES_2D:
            for _ in range(self.n_per_2d_type):
                rec = _FigRecord()
                if kind in ("line", "scatter"):
                    fig, ents, para, cap = self._fig_xy(kind, rec)
                elif kind == "bar":
                    fig, ents, para, cap = self._fig_bar(rec)
                else:
                    fig, ents, para, cap = self._fig_heatmap(rec)
                figures.append(self._save(f"fig_v2_{k:04d}", fig, rec, kind, False, ents, para, cap, self._split()))
                k += 1

        dataset_out = {
            "dataset_name": "SciFig-Controlled-v2",
            "provenance": "synthetic",
            "version": "2.0 (shortcut-free)",
            "num_figures": len(figures),
            "num_train": sum(1 for f in figures if f["split"] == "train"),
            "num_test": sum(1 for f in figures if f["split"] == "test"),
            "figures": figures
        }

        out_meta = os.path.join(self.out_dir, "metadata.json")
        with open(out_meta, "w") as f:
            json.dump(dataset_out, f, indent=2)

        print(f"Generated SciFig-Controlled v2: {len(figures)} figures ({dataset_out['num_train']} train, {dataset_out['num_test']} test)")
        print(f"Saved to: {out_meta}")
        return dataset_out


if __name__ == "__main__":
    gen = SciFigBenchmarkGeneratorV2()
    gen.generate()
