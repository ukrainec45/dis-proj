"""pytest suite for the MOA* pre-flight planner.

Run with:  python -m pytest scripts/validate.py -v
"""

import numpy as np
import pytest

from . import synthetic
from .domination import dominates, non_dominated_indices, weakly_dominates
from .edge_costs import (CostMap, NEIGHBOR_OFFSETS, edge_objectives,
                         turn_is_feasible, turn_radius_m)
from .heuristics import compute_heuristics, compute_time_to_landing_with_turns
from .moa_star import EmoaStarLateBS, bs_check, bs_filter
from .topsis import topsis


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _tiny_map(nav=0.9, vis=1.0, wind=(0.0, 0.0), dem=0.0):
    H, W = 3, 3
    dem = np.full((H, W), dem, dtype=np.float32)
    nav = np.full((H, W), nav, dtype=np.float32)
    vis = np.full((H, W), vis, dtype=np.float32)
    wind = np.tile(np.array(wind, dtype=np.float32), (H, W, 1))
    occ = np.zeros((H, W), dtype=bool)
    return CostMap(dem=dem, nav_density=nav, visibility=vis, wind_field=wind,
                   occupancy=occ, resolution_m=10.0, start=(0, 0), goal=(2, 2),
                   v_air=15.0, v_max=12.0)


def _path_cost(path, cost_map, v_air, v_max):
    total = np.zeros(3)
    for a, b in zip(path[:-1], path[1:]):
        c = edge_objectives(a, b, cost_map, v_air, v_max)
        assert c is not None, f"edge {a}->{b} unexpectedly infeasible"
        total += np.asarray(c)
    return tuple(total)


def _brute_force_front(cost_map, v_air, v_max):
    """Enumerate all simple paths and return the non-dominated cost set."""
    start, goal = cost_map.start, cost_map.goal
    H, W = cost_map.shape
    costs = []

    def dfs(node, previous, visited, acc):
        if node == goal:
            costs.append(tuple(acc))
            return
        x, y = node
        for dx, dy in NEIGHBOR_OFFSETS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            nxt = (nx, ny)
            if nxt in visited or cost_map.is_occupied(nxt):
                continue
            if previous is not None and not turn_is_feasible(previous, node, nxt, cost_map):
                continue
            c = edge_objectives(node, nxt, cost_map, v_air, v_max)
            if c is None:
                continue
            dfs(nxt, node, visited | {nxt}, [a + b for a, b in zip(acc, c)])

    dfs(start, None, {start}, [0.0, 0.0, 0.0])
    return [costs[i] for i in non_dominated_indices(costs)]


def _dedupe_sols(sols, ndigits=6):
    """Drop float-noise duplicates from the solver front (for comparisons)."""
    costs = np.array([c for _, c in sols])
    keep = non_dominated_indices(np.round(costs, ndigits))
    return [(p, c) for i, (p, c) in enumerate(sols) if i in keep]


# ----------------------------------------------------------------------------
# edge cost units
# ----------------------------------------------------------------------------
class TestEdgeCosts:
    def test_calm_air_time(self):
        cm = _tiny_map()
        c = edge_objectives((0, 0), (1, 0), cm, cm.v_air, cm.v_max)
        assert c is not None
        t = 10.0 / 15.0  # distance 10 m at 15 m/s
        assert c[0] == pytest.approx(t, abs=1e-9)

    def test_tailwind_reduces_time(self):
        cm = _tiny_map(wind=(8.0, 0.0))
        calm = edge_objectives((0, 0), (1, 0), cm, cm.v_air, cm.v_max)
        tail = edge_objectives((0, 0), (1, 0), cm, cm.v_air, cm.v_max)
        cm2 = _tiny_map()
        calm = edge_objectives((0, 0), (1, 0), cm2, cm2.v_air, cm2.v_max)
        assert tail[0] < calm[0]

    def test_headwind_increases_time(self):
        cm = _tiny_map(wind=(-8.0, 0.0))
        head = edge_objectives((0, 0), (1, 0), cm, cm.v_air, cm.v_max)
        cm2 = _tiny_map()
        calm = edge_objectives((0, 0), (1, 0), cm2, cm2.v_air, cm2.v_max)
        assert head[0] > calm[0]

    def test_crosswind_increases_time(self):
        cm = _tiny_map(wind=(0.0, 8.0))
        cross = edge_objectives((0, 0), (1, 0), cm, cm.v_air, cm.v_max)
        cm2 = _tiny_map()
        calm = edge_objectives((0, 0), (1, 0), cm2, cm2.v_air, cm2.v_max)
        assert cross[0] > calm[0]

    def test_north_wind_is_tailwind_for_a_northbound_grid_move(self):
        cm = _tiny_map(wind=(0.0, 8.0))
        northbound = edge_objectives((1, 1), (1, 0), cm, cm.v_air, cm.v_max)
        cm_calm = _tiny_map()
        calm = edge_objectives((1, 1), (1, 0), cm_calm, cm_calm.v_air, cm_calm.v_max)
        assert northbound[0] < calm[0]

    def test_nav_deficit(self):
        cm_rich = _tiny_map(nav=1.0)
        cm_poor = _tiny_map(nav=0.0)
        rich = edge_objectives((0, 0), (1, 0), cm_rich, cm_rich.v_air, cm_rich.v_max)
        poor = edge_objectives((0, 0), (1, 0), cm_poor, cm_poor.v_air, cm_poor.v_max)
        assert rich[1] == pytest.approx(0.0, abs=1e-9)
        assert poor[1] == pytest.approx(poor[0], abs=1e-9)  # (1 - 0) * t

    def test_visibility_deficit(self):
        cm = _tiny_map(vis=1.0)
        c = edge_objectives((0, 0), (1, 0), cm, cm.v_air, cm.v_max)
        assert c[2] == pytest.approx(0.0, abs=1e-9)

    def test_wind_at_vmax_infeasible(self):
        cm = _tiny_map(wind=(12.0, 0.0))
        assert edge_objectives((0, 0), (1, 0), cm, cm.v_air, cm.v_max) is None

    def test_wind_below_vmax_feasible(self):
        cm = _tiny_map(wind=(11.9, 0.0))
        assert edge_objectives((0, 0), (1, 0), cm, cm.v_air, cm.v_max) is not None

    def test_wind_above_vmax_is_infeasible(self):
        """A forecast value beyond the aircraft limit closes the corridor."""
        cm = _tiny_map(wind=(12.1, 0.0))
        assert edge_objectives((0, 0), (1, 0), cm, cm.v_air, cm.v_max) is None

    def test_occupied_target_infeasible(self):
        cm = _tiny_map()
        cm.occupancy[0, 1] = True  # cell (x=1, y=0)
        assert edge_objectives((0, 0), (1, 0), cm, cm.v_air, cm.v_max) is None


# ----------------------------------------------------------------------------
# dominance helpers
# ----------------------------------------------------------------------------
class TestDomination:
    def test_weakly(self):
        assert weakly_dominates((1, 2), (1, 2))
        assert weakly_dominates((1, 2), (3, 4))
        assert not weakly_dominates((3, 2), (1, 4))

    def test_strict(self):
        assert dominates((1, 2), (3, 4))
        assert not dominates((1, 2), (1, 2))
        assert not dominates((3, 2), (1, 4))

    def test_non_dominated(self):
        assert non_dominated_indices([(1, 4), (2, 3), (3, 1), (4, 0), (5, 5)]) == [0, 1, 2, 3]


class TestBS:
    def test_check_dominated(self):
        assert bs_check([(1, 2), (3, 1)], (2, 2)) is True
        assert bs_check([(1, 2), (3, 1)], (2, 0.5)) is False
        assert bs_check([(1, 2), (3, 1)], (0.5, 5)) is False
        assert bs_check([], (2, 2)) is False

    def test_filter_keeps_sorted_and_removes_dominated(self):
        assert bs_filter([(1, 2), (3, 1)], (2, 2)) == [(1, 2), (2, 2), (3, 1)]
        assert bs_filter([(1, 2), (3, 1)], (1, 3)) == [(1, 2), (1, 3), (3, 1)]
        assert bs_filter([], (1, 1)) == [(1, 1)]
        assert bs_filter([(2, 2)], (1, 1)) == [(1, 1)]


# ----------------------------------------------------------------------------
# heuristic
# ----------------------------------------------------------------------------
class TestHeuristic:
    def test_goal_is_zero(self):
        cm = synthetic.lake_map()
        h = compute_heuristics(cm, cm.v_air, cm.v_max)
        assert np.allclose(h[cm.goal[1], cm.goal[0]], 0.0, atol=1e-9)

    def test_consistent(self):
        cm = synthetic.brute_map(seed=5)
        h = compute_heuristics(cm, cm.v_air, cm.v_max)
        H, W = cm.shape
        for y in range(H):
            for x in range(W):
                if cm.is_occupied((x, y)) or not np.all(np.isfinite(h[y, x])):
                    continue
                hv = h[y, x]
                for dx, dy in NEIGHBOR_OFFSETS:
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < W and 0 <= ny < H) or cm.is_occupied((nx, ny)):
                        continue
                    c = edge_objectives((x, y), (nx, ny), cm, cm.v_air, cm.v_max)
                    if c is None:
                        continue
                    assert hv[0] <= c[0] + h[ny, nx, 0] + 1e-6
                    assert hv[1] <= c[1] + h[ny, nx, 1] + 1e-6
                    assert hv[2] <= c[2] + h[ny, nx, 2] + 1e-6


# ----------------------------------------------------------------------------
# planner scenarios
# ----------------------------------------------------------------------------
class TestPlannerScenarios:
    def test_lake_tradeoff(self):
        cm = synthetic.lake_map()
        sols = EmoaStarLateBS(cm).solve()
        assert len(sols) >= 2
        costs = np.array([c for _, c in sols])
        # mutual non-dominance
        for i in range(len(costs)):
            for j in range(len(costs)):
                if i != j:
                    assert not dominates(costs[j], costs[i])
        # a genuine f1/f2 tradeoff exists
        assert any(a[0] < b[0] and a[1] > b[1]
                   for a in costs for b in costs)

    def test_nfz_detour(self):
        cm = synthetic.nfz_map()
        sols = EmoaStarLateBS(cm).solve()
        assert len(sols) >= 1
        for path, _ in sols:
            for cell in path:
                assert not cm.is_occupied(cell)

    def test_tailwind_routing(self):
        cm = synthetic.tailwind_map()
        sols = EmoaStarLateBS(cm).solve()
        assert len(sols) >= 1
        min_f1 = min(c[0] for _, c in sols)
        # all-calm straight route: 9 north + 14 east steps at 15 m/s
        straight_time = 23 * (10.0 / 15.0)
        assert min_f1 < straight_time - 1.0

    def test_no_feasible_path(self):
        cm = synthetic.walled_map()
        sols = EmoaStarLateBS(cm).solve()
        assert sols == []

    def test_pareto_front_matches_brute_force(self):
        for seed in (3, 7):
            cm = synthetic.brute_map(seed=seed)
            solver = EmoaStarLateBS(cm)
            sols = solver.solve()
            expected = _brute_force_front(cm, cm.v_air, cm.v_max)
            got = {tuple(np.round(c, 6)) for _, c in sols}
            exp = {tuple(np.round(np.asarray(c), 6)) for c in expected}
            assert got == exp, f"seed {seed}: got {got}, expected {exp}"

    def test_infeasible_edge_never_used(self):
        cm = synthetic.nfz_map()
        sols = EmoaStarLateBS(cm).solve()
        for path, _ in sols:
            for a, b in zip(path[:-1], path[1:]):
                assert edge_objectives(a, b, cm, cm.v_air, cm.v_max) is not None


class TestDemoMap:
    """The purpose-built demo map must yield a clean, readable 3-point front."""

    def test_clean_front(self):
        cm = synthetic.demo_map()
        sols = _dedupe_sols(EmoaStarLateBS(cm).solve())
        assert len(sols) == 3
        costs = np.array([c for _, c in sols])
        for i in range(len(costs)):
            for j in range(len(costs)):
                if i != j:
                    assert not dominates(costs[j], costs[i])
        # the fastest route crosses the lake, has the worst navigation deficit
        # and pays the fog penalty; the detours are clean in f3
        i_fast = int(np.argmin(costs[:, 0]))
        assert costs[i_fast, 1] == costs[:, 1].max()
        assert costs[i_fast, 2] > 0.5
        assert np.allclose(costs[np.arange(len(costs)) != i_fast, 2], 0.0,
                           atol=1e-9)
        # a genuine f1/f2 trade-off exists
        assert any(a[0] < b[0] and a[1] > b[1] for a in costs for b in costs)


class TestTerrainMap:
    """The land-cover-style map must have a realistic quality spread and a
    genuine spatial trade-off between flight time and navigation deficit."""

    def test_realistic_field(self):
        cm = synthetic.terrain_map()
        nav = cm.nav_density
        assert nav.min() == 0.0                       # the featureless lake
        assert np.ptp(nav) > 0.3                      # genuine quality spread
        assert len(np.unique(np.round(nav, 4))) > 10  # heterogeneous, not flat
        assert (nav < 0.4).sum() > 0                  # some cells score badly
        assert (nav > 0.8).sum() > 0                  # some cells score well
        assert np.all((nav >= 0.0) & (nav <= 0.95))
        # deterministic across builds
        assert np.array_equal(nav, synthetic.terrain_map().nav_density)

    def test_lake_is_largest_component(self):
        from . import visualize
        cm = synthetic.terrain_map()
        labels, n, biggest = visualize._low_nav_components(cm)
        assert n >= 2                       # separate poor-terrain patches
        assert biggest >= 1
        lake = np.zeros(cm.shape, dtype=bool)
        lake[3:8, 5:12] = True
        assert np.array_equal(labels[lake], np.full(lake.sum(), biggest))

    def test_front_non_dominated(self):
        cm = synthetic.terrain_map()
        sols = _dedupe_sols(EmoaStarLateBS(cm).solve())
        assert len(sols) >= 2
        costs = np.array([c for _, c in sols])
        for i in range(len(costs)):
            for j in range(len(costs)):
                if i != j:
                    assert not dominates(costs[j], costs[i])
        # the trade-off is real: shortest path has the worst navigation deficit
        i_fast = int(np.argmin(costs[:, 0]))
        assert costs[i_fast, 1] == costs[:, 1].max()
        assert any(a[0] < b[0] and a[1] > b[1] for a in costs for b in costs)

    def test_map_plot_runs_terrain(self, tmp_path):
        pytest.importorskip("matplotlib")
        from . import visualize
        assert visualize.map_plot(synthetic.terrain_map(), str(tmp_path)) == 0
        assert any(tmp_path.iterdir())


class TestRealisticMap:
    """End-to-end scenario exercising the combined physical constraints."""

    def test_realistic_demo_is_feasible_and_constraint_compliant(self):
        cm = synthetic.realistic_map()
        sols = EmoaStarLateBS(cm).solve()
        assert sols
        assert cm.min_turn_radius_m == 10.0
        assert cm.landing_sites is not None and cm.landing_sites.sum() == 2
        for path, cost in sols:
            assert cost[0] <= cm.usable_flight_time_s + 1e-9
            for cell in path:
                assert not cm.is_occupied(cell)
                assert cm.altitude_m(cell) <= cm.z_max_m
            for previous, vertex, successor in zip(path, path[1:], path[2:]):
                assert turn_is_feasible(previous, vertex, successor, cm)

    def test_realistic_demo_has_a_pareto_tradeoff(self):
        sols = EmoaStarLateBS(synthetic.realistic_map()).solve()
        costs = np.asarray([cost for _, cost in sols])
        assert len(costs) >= 2
        assert np.ptp(costs[:, 2]) > 0.0  # both clear and fog-affected routes
        assert any(a[0] < b[0] and (a[1] > b[1] or a[2] > b[2])
                   for a in costs for b in costs)

    def test_realistic_map_plot_runs(self, tmp_path):
        pytest.importorskip("matplotlib")
        from . import visualize
        assert visualize.map_plot(synthetic.realistic_map(), str(tmp_path)) == 0
        assert any(tmp_path.iterdir())


class TestVisualizeSmoke:
    """The three visualization modes must run without error."""

    def test_trace_runs(self, capsys):
        from . import visualize
        visualize.trace(synthetic.demo_map(), limit=5)
        out = capsys.readouterr().out
        assert "FINAL PARETO FRONT" in out
        assert "TOPSIS" in out

    def test_map_plot_runs(self, tmp_path):
        pytest.importorskip("matplotlib")
        from . import visualize
        assert visualize.map_plot(synthetic.demo_map(), str(tmp_path)) == 0
        assert any(tmp_path.iterdir())

    def test_anim_runs(self, tmp_path):
        pytest.importorskip("matplotlib")
        pytest.importorskip("PIL")
        from . import visualize
        assert visualize.anim(synthetic.demo_map(), str(tmp_path)) == 0
        assert any(tmp_path.iterdir())


# ----------------------------------------------------------------------------
# TOPSIS
# ----------------------------------------------------------------------------
class TestTopsis:
    def test_ranks_best_ideal(self):
        costs = np.array([[10.0, 10.0], [5.0, 5.0], [8.0, 3.0]])
        order, closeness = topsis(costs, np.array([0.5, 0.5]))
        assert order[0] == 1  # dominates in both objectives
        assert closeness.min() >= 0.0
        assert closeness.max() <= 1.0

    def test_single_row(self):
        order, closeness = topsis(np.array([[3.0, 4.0]]), np.array([0.5, 0.5]))
        assert order.tolist() == [0]
        assert closeness[0] == pytest.approx(0.5)


# ----------------------------------------------------------------------------
# CostMap validation
# ----------------------------------------------------------------------------
class TestCostMapValidation:
    def test_occupied_start_raises(self):
        cm = synthetic.nfz_map()
        cm.occupancy[cm.start[1], cm.start[0]] = True
        with pytest.raises(ValueError):
            CostMap(**cm.__dict__)

    def test_occupied_goal_raises(self):
        cm = synthetic.nfz_map()
        cm.occupancy[cm.goal[1], cm.goal[0]] = True
        with pytest.raises(ValueError):
            CostMap(**cm.__dict__)

    def test_start_equal_to_goal_returns_zero_cost_route(self):
        cm = _tiny_map()
        cm.goal = cm.start
        assert EmoaStarLateBS(cm).solve() == [([cm.start], (0.0, 0.0, 0.0))]


# ----------------------------------------------------------------------------
# manuscript feasibility constraints
# ----------------------------------------------------------------------------
class TestFlightConstraints:
    def test_altitude_limit_blocks_high_intermediate_cell(self):
        cm = _tiny_map()
        cm.dem[1, 1] = 100.0
        cm.z_max_m = 50.0
        assert edge_objectives((0, 0), (1, 1), cm, cm.v_air, cm.v_max) is None

    def test_invalid_airspeed_limits_raise(self):
        cm = _tiny_map()
        cm.v_air_min_mps = 20.0
        with pytest.raises(ValueError, match="at least"):
            CostMap(**cm.__dict__)
        cm = _tiny_map()
        cm.v_air_max_mps = 10.0
        with pytest.raises(ValueError, match="exceed"):
            CostMap(**cm.__dict__)

    def test_solver_rejects_out_of_range_airspeed_override(self):
        cm = _tiny_map()
        cm.v_air_max_mps = 16.0
        with pytest.raises(ValueError, match="at least"):
            EmoaStarLateBS(cm, v_air=0.0)
        with pytest.raises(ValueError, match="exceed"):
            EmoaStarLateBS(cm, v_air=17.0)

    def test_unattainable_crosswind_is_infeasible(self):
        cm = _tiny_map(wind=(0.0, 6.0))
        assert edge_objectives((0, 0), (1, 0), cm, 5.0, cm.v_max) is None

    def test_headwind_below_wind_limit_can_still_prevent_progress(self):
        """A rescue corridor is unsafe when its headwind exceeds airspeed.

        This deliberately separates the meteorological wind limit from the
        direction-dependent forward-progress requirement.
        """
        cm = _tiny_map(wind=(-11.9, 0.0))
        assert edge_objectives((0, 0), (1, 0), cm, v_air=10.0, v_max=cm.v_max) is None

    def test_3d_wind_time_uses_unit_horizontal_track(self):
        cm = _tiny_map(wind=(0.0, 8.0))
        cm.dem[0, 1] = 10.0
        c = edge_objectives((0, 0), (1, 0), cm, cm.v_air, cm.v_max)
        # 3-D constraint: (V_h)^2 + 8^2 + (V_h)^2 = 15^2.
        v_horizontal = np.sqrt((15.0**2 - 8.0**2) / 2.0)
        assert c[0] == pytest.approx(10.0 / v_horizontal, abs=1e-9)

    def test_energy_budget_rejects_route_beyond_usable_time(self):
        cm = _tiny_map()
        cm.battery_energy_wh = 0.9
        cm.energy_reserve_wh = 0.0
        cm.cruise_power_w = 3600.0  # usable time = 0.9 s
        assert EmoaStarLateBS(cm).solve() == []
        cm.battery_energy_wh = 2.0  # two diagonal edges take ~= 1.886 s
        assert EmoaStarLateBS(cm).solve()

    def test_emergency_landing_reserve_is_enforced(self):
        cm = _tiny_map()
        cm.battery_energy_wh = 1.0
        cm.energy_reserve_wh = 0.0
        cm.cruise_power_w = 3600.0
        cm.landing_sites = np.zeros(cm.shape, dtype=bool)
        cm.landing_sites[0, 0] = True
        # Reaching the goal fits the route budget, but cannot leave enough
        # energy for a return to the only emergency landing cell.
        assert EmoaStarLateBS(cm).solve() == []

    def test_diagonal_cannot_cut_between_no_fly_cells(self):
        cm = _tiny_map()
        cm.occupancy[0, 1] = True
        cm.occupancy[1, 0] = True
        assert EmoaStarLateBS(cm).solve() == []

    def test_turn_radius_geometry(self):
        cm = _tiny_map()
        # Two 10 m cardinal edges with a right-angle turn give R = 10/sqrt(2).
        assert turn_radius_m((0, 0), (1, 0), (1, 1), cm) == pytest.approx(
            10.0 / np.sqrt(2.0), abs=1e-9
        )
        assert np.isinf(turn_radius_m((0, 0), (1, 0), (2, 0), cm))

    def test_minimum_turn_radius_changes_feasibility(self):
        H, W = 2, 3
        cm = CostMap(
            dem=np.zeros((H, W)), nav_density=np.ones((H, W)),
            visibility=np.ones((H, W)), wind_field=np.zeros((H, W, 2)),
            occupancy=np.zeros((H, W), dtype=bool), resolution_m=10.0,
            start=(0, 0), goal=(2, 1), min_turn_radius_m=20.0,
        )
        # Reaching this non-collinear goal needs at least a 45-degree turn,
        # whose radius is below 20 m for 10 m/diagonal grid segments.
        assert EmoaStarLateBS(cm).solve() == []
        cm.min_turn_radius_m = None
        assert EmoaStarLateBS(cm).solve()

    def test_turn_constrained_front_matches_brute_force(self):
        cm = synthetic.brute_map(seed=7)
        cm.min_turn_radius_m = 8.0
        got = {tuple(np.round(c, 6)) for _, c in EmoaStarLateBS(cm).solve()}
        expected = {tuple(np.round(np.asarray(c), 6))
                    for c in _brute_force_front(cm, cm.v_air, cm.v_max)}
        assert got == expected

    def test_emergency_landing_respects_incoming_heading(self):
        cm = _tiny_map()
        cm.min_turn_radius_m = 20.0
        cm.landing_sites = np.zeros(cm.shape, dtype=bool)
        cm.landing_sites[0, 1] = True
        landing_time = compute_time_to_landing_with_turns(cm, cm.v_air, cm.v_max)
        # From (1,1), an eastward arrival cannot make the 90-degree turn north.
        assert not np.isfinite(landing_time.get((1, 1, 1, 0), np.inf))
        # An arrival already heading north can land directly.
        assert np.isfinite(landing_time[(1, 1, 0, -1)])


# ----------------------------------------------------------------------------
# operational scenario regressions
# ----------------------------------------------------------------------------
class TestOperationalScenarios:
    """Small real-world-shaped maps with deterministic behavioural oracles."""

    def test_foggy_valley_and_clear_ridge_are_pareto_alternatives(self):
        """SAR flight: a short foggy valley competes with a clear ridge route."""
        H, W = 3, 5
        dem = np.zeros((H, W))
        nav = np.ones((H, W))
        visibility = np.ones((H, W))
        visibility[1, 1:4] = 0.0  # fog in the direct valley corridor
        cm = CostMap(
            dem=dem, nav_density=nav, visibility=visibility,
            wind_field=np.zeros((H, W, 2)), occupancy=np.zeros((H, W), dtype=bool),
            resolution_m=10.0, start=(0, 1), goal=(4, 1), v_air=15.0, v_max=12.0,
        )
        costs = np.asarray([cost for _, cost in EmoaStarLateBS(cm).solve()])
        assert len(costs) >= 2
        assert costs[:, 2].min() == pytest.approx(0.0)
        assert costs[:, 2].max() > 0.0
        assert any(a[0] < b[0] and a[2] > b[2] for a in costs for b in costs)

    def test_ridge_under_altitude_ceiling_requires_a_pass(self):
        """Mountain response: the high ridge is excluded, but a pass remains."""
        H = W = 3
        dem = np.zeros((H, W))
        dem[1, 1] = 100.0
        cm = CostMap(
            dem=dem, nav_density=np.ones((H, W)), visibility=np.ones((H, W)),
            wind_field=np.zeros((H, W, 2)), occupancy=np.zeros((H, W), dtype=bool),
            resolution_m=10.0, start=(0, 2), goal=(2, 0), v_air=15.0, v_max=12.0,
            cruise_altitude_agl_m=20.0, z_max_m=50.0,
        )
        sols = EmoaStarLateBS(cm).solve()
        assert sols
        assert all((1, 1) not in path for path, _ in sols)
        assert all(cm.altitude_m(cell) <= cm.z_max_m for path, _ in sols for cell in path)

    def test_featureless_water_and_feature_rich_shoreline_trade_time_for_navigation(self):
        """Inspection flight: a lake crossing is fast; the shoreline is localisable."""
        cm = synthetic.lake_map()
        costs = np.asarray([cost for _, cost in EmoaStarLateBS(cm).solve()])
        fastest = costs[np.argmin(costs[:, 0])]
        safest_navigation = costs[np.argmin(costs[:, 1])]
        assert fastest[0] < safest_navigation[0]
        assert fastest[1] > safest_navigation[1]

    def test_integrated_operational_map_preserves_all_hard_constraints(self):
        """End-to-end mission: wind, NFZ, altitude, reserve, and VTOL turns."""
        cm = synthetic.realistic_map()
        sols = EmoaStarLateBS(cm).solve()
        assert sols
        for path, cost in sols:
            assert cost[0] <= cm.usable_flight_time_s + 1e-9
            assert all(not cm.is_occupied(cell) for cell in path)
            assert all(cm.altitude_m(cell) <= cm.z_max_m for cell in path)
            assert all(turn_is_feasible(a, b, c, cm) for a, b, c in zip(path, path[1:], path[2:]))
