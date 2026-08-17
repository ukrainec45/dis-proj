"""Aggregate result directories into thesis-ready tables and paired statistics."""

import csv
import json
from pathlib import Path

import numpy as np


NUMERIC = ("runtime_ms", "peak_memory_kib", "solution_count", "normalized_hypervolume",
           "additive_epsilon", "union_front_recall", "best_f1_time_s",
           "best_f2_nav_deficit_s", "best_f3_visibility_deficit_s")


def _read_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            row["feasible"] = row["feasible"].lower() == "true"
            for key in NUMERIC:
                if row.get(key, "") not in ("", None):
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (None, None, None)
    return tuple(float(v) for v in np.quantile(values, (0.25, 0.5, 0.75)))


def _bootstrap_median_ci(values, seed=2026, samples=2000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    medians = [np.median(rng.choice(values, len(values), replace=True)) for _ in range(samples)]
    return tuple(float(v) for v in np.quantile(medians, (0.025, 0.975)))


def aggregate(rows, group):
    output = []
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method and row["feasible"]]
        record = {"group": group, "method": method, "instances": len(subset),
                  "feasibility_rate": float(np.mean([row["feasible"] for row in rows if row["method"] == method]))}
        for metric in ("normalized_hypervolume", "union_front_recall", "runtime_ms", "peak_memory_kib", "solution_count"):
            values = [row[metric] for row in subset if row.get(metric) is not None]
            q1, median, q3 = _quantiles(values)
            ci_low, ci_high = _bootstrap_median_ci(values)
            record.update({f"{metric}_q1": q1, f"{metric}_median": median,
                           f"{metric}_q3": q3, f"{metric}_ci95_low": ci_low,
                           f"{metric}_ci95_high": ci_high})
        output.append(record)
    return output


def paired_statistics(rows, metric, first="emoa_star", second="repeated_weighted_astar"):
    keyed = {(row["scenario"], row["method"]): row for row in rows if row["feasible"]}
    pairs = [(key[0], keyed[(key[0], first)][metric], keyed[(key[0], second)][metric])
             for key in keyed if key[1] == first and (key[0], second) in keyed and
             keyed[(key[0], first)].get(metric) is not None and keyed[(key[0], second)].get(metric) is not None]
    if not pairs:
        return {"metric": metric, "n": 0}
    a = np.asarray([pair[1] for pair in pairs])
    b = np.asarray([pair[2] for pair in pairs])
    diff = a - b
    nonzero = diff[np.abs(diff) > 1e-12]
    result = {"metric": metric, "n": len(pairs), "median_difference": float(np.median(diff)),
              "superiority_rate": float(np.mean(diff > 0))}
    if len(nonzero):
        try:
            from scipy.stats import rankdata, wilcoxon
            statistic, pvalue = wilcoxon(nonzero, alternative="two-sided", method="auto")
            ranks = rankdata(np.abs(nonzero))
            rank_biserial = float(np.sum(np.sign(nonzero) * ranks) / np.sum(ranks))
            result.update({"wilcoxon_statistic": float(statistic), "p_value": float(pvalue),
                           "rank_biserial_effect": rank_biserial})
        except ImportError:
            result["p_value"] = None
    return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(rows, analysis_dir):
    """Write distributions usable in the thesis results chapter when matplotlib exists."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    feasible = [row for row in rows if row["feasible"]]
    methods = sorted({row["method"] for row in feasible})
    if not methods:
        return
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, metric, title in zip(
            axes, ("normalized_hypervolume", "union_front_recall", "runtime_ms"),
            ("Normalized hypervolume", "Union-front recall", "Planning time")):
        data = [[row[metric] for row in feasible if row["method"] == method and row.get(metric) is not None]
                for method in methods]
        axis.boxplot(data, tick_labels=methods, showmeans=True)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
        if metric == "runtime_ms":
            axis.set_ylabel("ms")
        else:
            axis.set_ylim(-0.05, 1.05)
    figure.savefig(analysis_dir / "quality_runtime_distributions.png", dpi=160)
    plt.close(figure)

    ablation = [row for row in feasible if row.get("navigation_model") in
                ("full", "visual_only", "terrain_only")]
    models = sorted({row["navigation_model"] for row in ablation})
    if len(models) >= 2:
        figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        values = [[row.get("balanced_f2") for row in ablation
                   if row["navigation_model"] == model and row.get("balanced_f2") is not None]
                  for model in models]
        axis.boxplot(values, tick_labels=models, showmeans=True)
        axis.set_title("Balanced-route navigation deficit ablation")
        axis.set_ylabel("f2 navigation deficit [s]")
        figure.savefig(analysis_dir / "navigation_ablation.png", dpi=160)
        plt.close(figure)


def write_scaling_figure(rows, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method in sorted({row["method"] for row in feasible}):
        subset = sorted((row for row in feasible if row["method"] == method),
                        key=lambda row: float(row.get("grid_scale") or 1))
        x = [float(row.get("grid_scale") or 1) for row in subset]
        axes[0].plot(x, [row["runtime_ms"] for row in subset], marker="o", label=method)
        axes[1].plot(x, [row["solution_count"] for row in subset], marker="o", label=method)
    axes[0].set_title("Scaling: planning time")
    axes[0].set_xlabel("grid scale")
    axes[0].set_ylabel("ms")
    axes[1].set_title("Scaling: front size")
    axes[1].set_xlabel("grid scale")
    for axis in axes:
        axis.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyze(run_dir):
    run_dir = Path(run_dir)
    rows = _read_rows(run_dir / "summary.csv")
    synthetic_rows = [row for row in rows if row.get("case_kind") == "synthetic_campaign"]
    real_rows = [row for row in rows if row.get("case_kind") == "real_aoi"]
    # Initial control runs are retained as a third group for auditability.
    control_rows = [row for row in rows if row not in synthetic_rows and row not in real_rows]
    analysis_dir = run_dir / "analysis"
    write_csv(analysis_dir / "synthetic_aggregate.csv", aggregate(synthetic_rows, "synthetic"))
    write_csv(analysis_dir / "real_aoi_aggregate.csv", aggregate(real_rows, "real_aoi"))
    write_csv(analysis_dir / "control_aggregate.csv", aggregate(control_rows, "control"))
    paired = []
    for group, group_rows in (("synthetic", synthetic_rows), ("real_aoi", real_rows)):
        for metric in ("normalized_hypervolume", "union_front_recall", "runtime_ms"):
            record = paired_statistics(group_rows, metric)
            record["group"] = group
            paired.append(record)
    write_csv(analysis_dir / "paired_tests.csv", paired)
    protocol = """# Evaluation protocol\n\nMethods: EMOA*, time-only A*, repeated weighted A* (15 fixed simplex weights).\n\nAll methods share CostMap constraints. Hypervolume is normalized within an instance; paired Wilcoxon tests compare EMOA* and repeated weighted A* at alpha=0.05.\n"""
    (analysis_dir / "methods_protocol.md").write_text(protocol, encoding="utf-8")
    _write_figures(rows, analysis_dir)
    return {"rows": len(rows), "analysis_dir": str(analysis_dir)}
