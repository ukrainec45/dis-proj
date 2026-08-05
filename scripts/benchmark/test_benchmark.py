"""Tests for baseline fairness, metrics, and persistent benchmark artifacts."""

import numpy as np
import pytest

from scripts.benchmark.baselines import (QUARTER_SIMPLEX_WEIGHTS, run_emoa,
                                          run_repeated_weighted_astar,
                                          run_time_only_astar)
from scripts.benchmark.metrics import coverage, hypervolume, reference_point, scenario_metrics
from scripts.benchmark.run_benchmark import main
from scripts.moa import synthetic
from scripts.moa.domination import dominates


def test_quarter_simplex_has_fifteen_normalized_weights():
    assert len(QUARTER_SIMPLEX_WEIGHTS) == 15
    assert all(sum(weight) == pytest.approx(1.0) for weight in QUARTER_SIMPLEX_WEIGHTS)
    assert (1.0, 0.0, 0.0) in QUARTER_SIMPLEX_WEIGHTS


def test_time_only_baseline_matches_emoa_fastest_route():
    cost_map = synthetic.tailwind_map()
    exact = run_emoa(cost_map)
    baseline = run_time_only_astar(cost_map)
    assert exact.feasible and baseline.feasible
    assert baseline.solutions[0][1][0] == pytest.approx(min(cost[0] for _, cost in exact.solutions))


def test_repeated_weighted_front_is_mutually_non_dominated_and_feasible():
    cost_map = synthetic.lake_map()
    result = run_repeated_weighted_astar(cost_map)
    assert result.feasible
    for path, cost in result.solutions:
        assert path[0] == cost_map.start and path[-1] == cost_map.goal
        assert not any(dominates(other, cost) for _, other in result.solutions if other != cost)


def test_scalar_baselines_keep_integrated_energy_and_turn_constraints():
    cost_map = synthetic.realistic_map()
    for result in (run_time_only_astar(cost_map), run_repeated_weighted_astar(cost_map)):
        for path, cost in result.solutions:
            assert cost[0] <= cost_map.usable_flight_time_s + 1e-9
            assert all(not cost_map.is_occupied(cell) for cell in path)


def test_metrics_have_known_coverage_and_positive_hypervolume():
    front_a = [([], (1.0, 4.0, 2.0)), ([], (2.0, 2.0, 1.0))]
    front_b = [([], (3.0, 5.0, 3.0))]
    assert coverage(front_a, front_b) == 1.0
    reference = reference_point(np.asarray([cost for _, cost in front_a + front_b]))
    assert hypervolume(np.asarray([cost for _, cost in front_a]), reference) > 0.0


def test_scenario_metrics_share_one_reference_point():
    results = [run_emoa(synthetic.lake_map()), run_time_only_astar(synthetic.lake_map())]
    _, reference, metrics = scenario_metrics(results)
    assert reference is not None
    assert set(metrics) == {result.method for result in results}
    assert all("hypervolume" in metric for metric in metrics.values())


def test_runner_writes_versioned_tables_paths_and_figures(tmp_path):
    pytest.importorskip("matplotlib")
    status = main(["--scenario", "lake", "--run-id", "smoke", "--output-dir", str(tmp_path)])
    out = tmp_path / "smoke"
    assert status == 0
    assert (out / "manifest.json").is_file()
    assert (out / "summary.csv").is_file()
    assert (out / "runs" / "lake" / "emoa_star.json").is_file()
    assert (out / "figures" / "lake.png").is_file()
    assert (out / "figures" / "summary.png").is_file()
