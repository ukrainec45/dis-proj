"""Command-line runner for reproducible pre-flight planning comparisons.

Example:
    python -m scripts.benchmark.run_benchmark --run-id synthetic-v1
    python -m scripts.benchmark.run_benchmark --run-id real-aoi-01 \
        --scenario realistic --map zhytomyr=path/to/planning_layers.npz
"""

import argparse
import hashlib
import platform
import shutil
import sys
import tracemalloc
from pathlib import Path

import numpy as np

from .baselines import METHODS, QUARTER_SIMPLEX_WEIGHTS
from .metrics import scenario_metrics
from .report import (TOPSIS_PROFILES, result_json, selected_paths, write_json, write_scenario_figure,
                     write_summary, write_summary_figure)
from .scenarios import BUILTIN_SCENARIOS, builtin_cases, npz_case


def _parse_named_map(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("map must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("map must use NAME=PATH")
    return name, path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_manifest(case):
    item = {"name": case.name, "source": case.source, "metadata": case.metadata or {}}
    source = Path(case.source)
    if source.is_file():
        item["sha256"] = _sha256(source)
    return item


def _summary_row(case, result, metric, reference):
    costs = np.asarray([cost for _, cost in result.solutions], dtype=float)
    row = {
        "scenario": case.name,
        "source": case.source,
        "method": result.method,
        "feasible": result.feasible,
        "runtime_ms": round(result.runtime_ms, 6),
        "n_expanded": result.n_expanded,
        "n_generated": result.n_generated,
        "peak_memory_kib": result.peak_memory_kib,
        "solution_count": len(result.solutions),
        "best_f1_time_s": float(costs[:, 0].min()) if len(costs) else None,
        "best_f2_nav_deficit_s": float(costs[:, 1].min()) if len(costs) else None,
        "best_f3_visibility_deficit_s": float(costs[:, 2].min()) if len(costs) else None,
        "hypervolume": metric["hypervolume"],
        "normalized_hypervolume": metric["normalized_hypervolume"],
        "additive_epsilon": metric["additive_epsilon"],
        "union_front_recall": metric["union_front_recall"],
        "reference_f1": None if reference is None else float(reference[0]),
        "reference_f2": None if reference is None else float(reference[1]),
        "reference_f3": None if reference is None else float(reference[2]),
    }
    for other, value in metric["coverage"].items():
        row[f"coverage_of_{other}"] = value
    row.update(case.metadata or {})
    return row


def _run_with_memory(method, cost_map, measure_memory=False):
    """Time an uninstrumented run; optionally measure allocations in a probe run."""
    result = method(cost_map)
    if not measure_memory:
        return result
    tracemalloc.start()
    try:
        method(cost_map)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    result.peak_memory_kib = round(peak / 1024.0, 3)
    return result


def run_cases(cases, output_dir, measure_memory=False):
    """Run all methods and write complete artifacts below an existing directory."""
    output_dir = Path(output_dir)
    rows = []
    for case in cases:
        results = [_run_with_memory(method, case.cost_map, measure_memory) for method in METHODS]
        shared_front, reference, metrics = scenario_metrics(results)
        case_dir = output_dir / "runs" / case.name
        time_only = next(result for result in results if result.method == "time_only_astar")
        baseline_selections = selected_paths(time_only.solutions)
        for result in results:
            payload = result_json(result)
            payload["scenario"] = case.name
            payload["source"] = case.source
            payload["case_metadata"] = case.metadata or {}
            payload["scenario_reference_point"] = None if reference is None else reference.tolist()
            payload["shared_non_dominated_front"] = shared_front.tolist()
            payload["metrics"] = metrics[result.method]
            write_json(case_dir / f"{result.method}.json", payload)
            row = _summary_row(case, result, metrics[result.method], reference)
            for profile, selection in selected_paths(result.solutions).items():
                baseline = baseline_selections.get(profile)
                row.update({
                    f"{profile}_f1": selection["cost"][0],
                    f"{profile}_f2": selection["cost"][1],
                    f"{profile}_f3": selection["cost"][2],
                    f"{profile}_time_premium_vs_time_only": None if baseline is None else
                        selection["cost"][0] - baseline["cost"][0],
                    f"{profile}_nav_improvement_vs_time_only": None if baseline is None else
                        baseline["cost"][1] - selection["cost"][1],
                    f"{profile}_visibility_improvement_vs_time_only": None if baseline is None else
                        baseline["cost"][2] - selection["cost"][2],
                })
            rows.append(row)
        write_scenario_figure(output_dir / "figures" / f"{case.name}.png", case, results)
    write_summary(output_dir / "summary.csv", rows)
    write_summary_figure(output_dir / "figures" / "summary.png", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark GPS-denied UAV pre-flight planners")
    parser.add_argument("--scenario", action="append", choices=sorted(BUILTIN_SCENARIOS),
                        help="built-in scenario; repeat to select several (default: all)")
    parser.add_argument("--map", action="append", type=_parse_named_map, default=[],
                        metavar="NAME=PATH", help="CostMap-compatible NPZ archive; repeat as needed")
    parser.add_argument("--output-dir", default="results/benchmarks",
                        help="parent directory for versioned benchmark artifacts")
    parser.add_argument("--run-id", required=True, help="unique, version-controlled result directory name")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run-id directory")
    args = parser.parse_args(argv)

    cases = builtin_cases(args.scenario) if args.scenario else builtin_cases()
    names = {case.name for case in cases}
    for name, path in args.map:
        if name in names:
            parser.error(f"duplicate benchmark case name: {name}")
        cases.append(npz_case(name, path))
        names.add(name)

    output_dir = Path(args.output_dir) / args.run_id
    if output_dir.exists():
        if not args.overwrite:
            parser.error(f"result directory already exists: {output_dir}; use --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    manifest = {
        "run_id": args.run_id,
        "python": sys.version,
        "platform": platform.platform(),
        "methods": ["emoa_star", "time_only_astar", "repeated_weighted_astar"],
        "repeated_weighted_astar_weights": [list(weight) for weight in QUARTER_SIMPLEX_WEIGHTS],
        "topsis_profiles": {name: list(weights) for name, weights in TOPSIS_PROFILES.items()},
        "cases": [_case_manifest(case) for case in cases],
    }
    write_json(output_dir / "manifest.json", manifest)
    rows = run_cases(cases, output_dir)
    print(f"Wrote {len(rows)} method results for {len(cases)} case(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
