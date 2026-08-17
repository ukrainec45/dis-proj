"""Tests for baseline fairness, metrics, and persistent benchmark artifacts."""

import numpy as np
import pytest

from scripts.benchmark.baselines import (QUARTER_SIMPLEX_WEIGHTS, run_emoa,
                                          run_repeated_weighted_astar,
                                          run_time_only_astar)
from scripts.benchmark.metrics import coverage, hypervolume, reference_point, scenario_metrics
from scripts.benchmark.models import PlannerResult
from scripts.benchmark.report import _visibility_scale
from scripts.benchmark.run_benchmark import main
from scripts.benchmark.campaign import (apply_navigation_ablation, apply_vehicle,
                                        apply_weather, synthetic_campaign)
from scripts.benchmark.oracle import validate_oracle_suite
from scripts.benchmark.analysis import analyze, write_csv as write_analysis_csv
from scripts.benchmark import run_campaign
from scripts.benchmark.scenarios import BenchmarkCase
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


def test_visibility_colour_scale_is_shared_and_never_negative():
    all_zero = [PlannerResult("a", [([], (1.0, 2.0, 0.0))], 0.0)]
    mixed = [PlannerResult("a", [([], (1.0, 2.0, 4.5))], 0.0),
             PlannerResult("b", [([], (2.0, 1.0, 0.0))], 0.0)]
    assert _visibility_scale(all_zero) == (0.0, 1.0)
    assert _visibility_scale(mixed) == (0.0, 4.5)


def test_scenario_metrics_share_one_reference_point():
    results = [run_emoa(synthetic.lake_map()), run_time_only_astar(synthetic.lake_map())]
    _, reference, metrics = scenario_metrics(results)
    assert reference is not None
    assert set(metrics) == {result.method for result in results}
    assert all("hypervolume" in metric for metric in metrics.values())


def test_parameterized_synthetic_campaign_has_four_families_and_two_vehicles():
    cases = synthetic_campaign(variants=1, seed=4)
    assert len(cases) == 8
    assert {case.metadata["family"] for case in cases} == {
        "lake", "foggy_valley", "terrain", "realistic"
    }
    assert {case.metadata["vehicle"] for case in cases} == {"multirotor", "fixed_wing_vtol"}


def test_weather_vehicle_and_navigation_ablation_are_explicit():
    cm = synthetic.lake_map()
    weather = apply_weather(cm, "crosswind_clear")
    assert not np.array_equal(weather.wind_field, cm.wind_field)
    assert apply_vehicle(cm, "multirotor").min_turn_radius_m == 0.0
    assert apply_navigation_ablation(cm, "full") is cm
    with pytest.raises(ValueError, match="requires"):
        apply_navigation_ablation(cm, "visual_only")


def test_small_oracle_suite_matches_emoa_exactly():
    rows = validate_oracle_suite(range(2))
    assert all(row["exact_match"] for row in rows)


def test_analysis_writes_aggregate_and_paired_evidence(tmp_path):
    rows = []
    for scenario, emoa_hv, weighted_hv in (("s1", 1.0, 0.8), ("s2", 1.0, 0.7)):
        for method, hv in (("emoa_star", emoa_hv), ("repeated_weighted_astar", weighted_hv)):
            rows.append({"scenario": scenario, "method": method, "case_kind": "synthetic_campaign",
                         "feasible": "True", "runtime_ms": "10", "peak_memory_kib": "20",
                         "solution_count": "2", "normalized_hypervolume": str(hv),
                         "additive_epsilon": "0", "union_front_recall": str(hv),
                         "best_f1_time_s": "1", "best_f2_nav_deficit_s": "2",
                         "best_f3_visibility_deficit_s": "3"})
    write_analysis_csv(tmp_path / "summary.csv", rows)
    outcome = analyze(tmp_path)
    assert outcome["rows"] == 4
    assert (tmp_path / "analysis" / "synthetic_aggregate.csv").is_file()
    assert (tmp_path / "analysis" / "paired_tests.csv").is_file()


def test_campaign_runner_writes_oracle_and_aggregate_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_campaign, "synthetic_campaign", lambda variants, seed: [
        BenchmarkCase("smoke", synthetic.foggy_valley_map(), "synthetic:smoke",
                      {"case_kind": "synthetic_campaign", "family": "fog", "seed": seed,
                       "weather": "calm_clear", "vehicle": "multirotor", "grid_scale": 1})
    ])
    status = run_campaign.main(["--run-id", "campaign-smoke", "--output-dir", str(tmp_path),
                                "--synthetic-variants", "1"])
    out = tmp_path / "campaign-smoke"
    assert status == 0
    assert (out / "correctness_oracle.csv").is_file()
    assert (out / "analysis" / "synthetic_aggregate.csv").is_file()


def test_runner_writes_versioned_tables_paths_and_figures(tmp_path):
    pytest.importorskip("matplotlib")
    status = main(["--scenario", "foggy_valley", "--run-id", "smoke", "--output-dir", str(tmp_path)])
    out = tmp_path / "smoke"
    assert status == 0
    assert (out / "manifest.json").is_file()
    assert (out / "summary.csv").is_file()
    assert (out / "runs" / "foggy_valley" / "emoa_star.json").is_file()
    assert (out / "figures" / "foggy_valley.png").is_file()
    assert (out / "figures" / "summary.png").is_file()
