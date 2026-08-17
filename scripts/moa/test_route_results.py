import csv

import numpy as np

from .moa_star import EmoaStarLateBS
from .route_results import (build_route_results, load_route_results, route_path,
                            save_route_results, write_characteristics_csv)
from .synthetic import demo_map
from .visualize_results import load_characteristics_csv, plot_characteristics


def test_route_results_round_trip_and_characteristics(tmp_path):
    cost_map = demo_map()
    solver = EmoaStarLateBS(cost_map)
    solutions = solver.solve()
    output = tmp_path / "routes.npz"

    saved = save_route_results(output, cost_map, solutions, (0.5, 0.3, 0.2),
                               solver.n_expanded, solver.n_generated)
    loaded = load_route_results(output)

    assert loaded["costs"].shape == (len(solutions), 3)
    assert np.array_equal(route_path(loaded, 0), np.asarray(solutions[0][0]))
    assert int(loaded["topsis_best"]) == int(saved["topsis_best"])
    assert np.all(loaded["horizontal_length_m"] > 0)
    assert np.all((loaded["average_navigation_quality"] >= 0) &
                  (loaded["average_navigation_quality"] <= 1))

    csv_path = tmp_path / "characteristics.csv"
    write_characteristics_csv(csv_path, loaded)
    with csv_path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(solutions)
    assert sum(row["topsis_selected"] == "True" for row in rows) == 1


def test_zero_duration_route_has_perfect_average_quality():
    cost_map = demo_map()
    cost_map.goal = cost_map.start
    results = build_route_results(cost_map, [([cost_map.start], (0.0, 0.0, 0.0))])
    assert results["average_navigation_quality"][0] == 1.0
    assert results["average_visibility"][0] == 1.0
    assert results["average_ground_speed_mps"][0] == 0.0


def test_characteristics_dashboard_is_created(tmp_path):
    cost_map = demo_map()
    solutions = EmoaStarLateBS(cost_map).solve()
    results = build_route_results(cost_map, solutions)
    csv_path = tmp_path / "route_characteristics.csv"
    write_characteristics_csv(csv_path, results)
    loaded_from_csv = load_characteristics_csv(csv_path)
    output = tmp_path / "dashboard.png"

    assert plot_characteristics(loaded_from_csv, output, top_routes=3) == output
    assert output.is_file()
    assert output.stat().st_size > 0
