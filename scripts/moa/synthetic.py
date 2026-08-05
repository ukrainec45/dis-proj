"""Synthetic cost-map builders for unit tests and the CLI driver.

Each builder returns a :class:`CostMap` with a known correct answer so the
planner can be validated before it is run on real geodata.
"""

import numpy as np

from .edge_costs import CostMap


def _base(height, width, resolution=10.0):
    H, W = height, width
    dem = np.zeros((H, W), dtype=np.float32)
    nav = np.full((H, W), 0.9, dtype=np.float32)
    vis = np.ones((H, W), dtype=np.float32)
    wind = np.zeros((H, W, 2), dtype=np.float32)
    occ = np.zeros((H, W), dtype=bool)
    return dem, nav, vis, wind, occ


def lake_map(height=13, width=19, resolution=25.0):
    """Featureless lake blob + an avoidable fog patch produce a genuine front.

    The lake is a low-``nav_density`` blob in the middle that the direct route
    crosses (fast, poor navigation quality); detours trade extra flight time for
    better navigation quality. The fog patch sits in the top-left corner so one
    of the detours is additionally penalised in visibility.
    """
    dem, nav, vis, wind, occ = _base(height, width, resolution)
    nav[4:9, 5:14] = 0.0      # featureless lake blob (rows 4-8, cols 5-13)
    vis[0:3, 0:5] = 0.2       # fog patch top-left (avoidable)
    start, goal = (0, height - 1), (width - 1, 0)
    return CostMap(dem=dem, nav_density=nav, visibility=vis, wind_field=wind,
                   occupancy=occ, resolution_m=resolution, start=start, goal=goal,
                   v_air=15.0, v_max=12.0)


def demo_map(height=9, width=11, resolution=25.0):
    """Small purpose-built map for the visual demonstration.

    Start ``(0, 8)`` (bottom-left) to goal ``(10, 0)`` (top-right). A low-
    ``nav_density`` lake blob sits across the direct diagonal, with a small fog
    patch inside its lake-crossing corridor. The result is a clean three-point
    Pareto front (mutually non-dominated, monotone f1/f2 tradeoff):

        * straight through the lake -- min f1, high f2, pays f3 (fog)
        * upper detour (short)      -- higher f1, low f2, f3 = 0
        * upper detour (long)       -- highest f1, lowest f2, f3 = 0

    The map is tiny (9x11) so the whole search tree fits in one trace/plot.
    """
    dem, nav, vis, wind, occ = _base(height, width, resolution)
    nav[3:6, 4:9] = 0.0       # lake blob: rows 3-5, cols 4-8
    vis[4, 4:6] = 0.2         # fog on the lake-crossing corridor (row 4)
    start, goal = (0, height - 1), (width - 1, 0)
    return CostMap(dem=dem, nav_density=nav, visibility=vis, wind_field=wind,
                   occupancy=occ, resolution_m=resolution, start=start, goal=goal,
                   v_air=15.0, v_max=12.0)


def terrain_map(height=11, width=15, resolution=10.0, seed=7):
    """Land-cover-like navigation density with a genuine quality spread.

    ``nav_density`` mimics ``max(visual_richness, rugosity)`` from classified
    land cover: a featureless lake (0.0) across the direct corridor ringed by a
    poor-forest collar, plus a separate dense-forest patch (bad, ~0.24-0.26),
    open fields (mediocre, ~0.56), and built-up/roads (good, ~0.86-0.88), each
    an elliptical patch with per-cell noise. Some cells score well, some badly,
    so the planner must trade flight time against navigation quality across the
    whole grid. The forest collar around the lake keeps the Pareto front to a
    readable 4-point monotone trade-off. Seeded so the field is reproducible.
    """
    rng = np.random.default_rng(seed)
    dem, nav, vis, wind, occ = _base(height, width, resolution)
    nav[3:8, 5:12] = 0.0       # featureless lake across the direct corridor
    patches = [
        (8.0, 13.5, 1.3, 0.26),   # dense forest bottom-right (bad)
        (2.5, 10.0, 1.9, 0.56),   # open field (mediocre)
        (8.5, 4.0, 2.0, 0.58),    # open field (mediocre)
        (5.5, 12.0, 1.6, 0.88),   # built-up (good)
        (9.0, 7.0, 1.4, 0.86),    # built-up (good)
        (4.0, 8.0, 2.5, 0.24),    # forest collar painted last (bad ring)
    ]
    yy, xx = np.mgrid[0:height, 0:width]
    for cy, cx, r, value in patches:
        d = np.sqrt(((xx - cx) / r) ** 2 + ((yy - cy) / r) ** 2)
        nav[d <= 1.0] = value
    nav += rng.normal(0.0, 0.03, size=(height, width))
    nav = np.clip(nav, 0.05, 0.95)
    nav[3:8, 5:12] = 0.0       # keep the lake featureless after noise
    start, goal = (0, height - 1), (width - 1, 0)
    return CostMap(dem=dem, nav_density=nav, visibility=vis, wind_field=wind,
                   occupancy=occ, resolution_m=resolution, start=start, goal=goal,
                   v_air=15.0, v_max=12.0)


def realistic_map(height=18, width=28, resolution=20.0):
    """Deterministic operational-style pre-flight planning demonstration.

    The scenario combines the implemented constraints rather than isolating
    them: rolling terrain and an altitude-limited ridge, urban and lakeshore
    navigation quality, fog, two no-fly areas, a spatially varying wind field,
    a battery reserve with emergency landing sites, and a 10 m turn radius.
    It is deliberately synthetic; its purpose is reproducible end-to-end
    validation before substituting geospatial layers.
    """
    dem, nav, vis, wind, occ = _base(height, width, resolution)
    yy, xx = np.mgrid[0:height, 0:width]

    # Ground elevation in a common MSL datum: a broad ridge plus a lower hill.
    dem = (112.0 + 0.45 * xx + 0.25 * yy
           + 48.0 * np.exp(-((xx - 14.0) / 4.8) ** 2 - ((yy - 8.0) / 3.6) ** 2)
           + 18.0 * np.exp(-((xx - 5.0) / 3.0) ** 2 - ((yy - 3.0) / 2.5) ** 2))

    # Featureless water and a foggy valley compete with a longer, feature-rich
    # northern passage.  The values already represent max(phi_vis, phi_ter).
    nav = np.full((height, width), 0.72, dtype=float)
    lake = ((xx - 13.5) / 6.0) ** 2 + ((yy - 11.5) / 2.7) ** 2 <= 1.0
    nav[lake] = 0.08
    nav[(xx < 8) & (yy < 6)] = 0.90                 # built-up/road corridor
    nav[(xx > 20) & (yy > 10)] = 0.48               # sparse woodland
    vis = np.ones((height, width), dtype=float)
    fog = ((xx - 8.0) / 4.5) ** 2 + ((yy - 11.0) / 2.5) ** 2 <= 1.0
    # A local fog cell lies on the fast, tailwind-assisted southern corridor.
    # A one-cell patch keeps the demonstrator front readable while still
    # creating a genuine f1/f3 trade-off with routes that skirt it.
    fog[15, 12] = True
    vis[fog] = 0.35

    # A restricted compound and a narrow no-fly strip leave viable detours.
    occ[6:12, 18:20] = True
    occ[3:5, 10:16] = True

    # Eastward tailwind in the south, changing to a northerly crosswind near
    # the ridge.  Magnitudes stay below the 12 m/s hard limit.
    wind[:, :, 0] = 5.5 * np.exp(-((yy - 14.0) / 3.5) ** 2)
    wind[:, :, 1] = 3.5 * np.exp(-((xx - 14.0) / 5.0) ** 2)

    landing_sites = np.zeros((height, width), dtype=bool)
    landing_sites[16, 1] = True   # launch/recovery area
    landing_sites[2, 25] = True   # alternate landing area near destination
    start, goal = (1, 16), (26, 1)
    return CostMap(
        dem=dem, nav_density=nav, visibility=vis, wind_field=wind,
        occupancy=occ, resolution_m=resolution, start=start, goal=goal,
        v_air=15.0, v_max=12.0,
        v_air_min_mps=10.0, v_air_max_mps=22.0,
        z_max_m=205.0, cruise_altitude_agl_m=55.0,
        battery_energy_wh=18.0, energy_reserve_wh=3.0, cruise_power_w=450.0,
        landing_sites=landing_sites, min_turn_radius_m=10.0,
    )


def nfz_map(height=11, width=17, resolution=10.0):
    """No-fly barrier blocking the direct route; detour around top or bottom."""
    dem, nav, vis, wind, occ = _base(height, width, resolution)
    occ[height // 2 - 2: height // 2 + 2, width // 2] = True  # 4-cell barrier
    start, goal = (0, height - 1), (width - 1, 0)
    return CostMap(dem=dem, nav_density=nav, visibility=vis, wind_field=wind,
                   occupancy=occ, resolution_m=resolution, start=start, goal=goal,
                   v_air=15.0, v_max=12.0)


def tailwind_map(height=10, width=15, resolution=10.0):
    """Strong eastward tailwind confined to the bottom corridor row.

    With v_air = 15 m/s and wind 10 m/s east, eastward edges in the corridor
    reach a ground speed of ~25 m/s while all calm edges move at 15 m/s. The
    time-optimal path therefore runs along the bottom corridor before climbing
    north to the goal, beating the shorter all-calm route.
    """
    dem, nav, vis, wind, occ = _base(height, width, resolution)
    wind[height - 1, :, 0] = 10.0  # east tailwind in the bottom row
    start, goal = (0, height - 1), (width - 1, 0)
    return CostMap(dem=dem, nav_density=nav, visibility=vis, wind_field=wind,
                   occupancy=occ, resolution_m=resolution, start=start, goal=goal,
                   v_air=15.0, v_max=12.0)


def foggy_valley_map(height=3, width=5, resolution=10.0):
    """Search-and-rescue route: short foggy valley versus clear ridge detour."""
    dem, nav, vis, wind, occ = _base(height, width, resolution)
    vis[1, 1:4] = 0.0
    return CostMap(dem=dem, nav_density=nav, visibility=vis, wind_field=wind,
                   occupancy=occ, resolution_m=resolution, start=(0, 1), goal=(4, 1),
                   v_air=15.0, v_max=12.0)


def mountain_pass_map(resolution=10.0):
    """Mountain-response route where a high central ridge exceeds the ceiling."""
    dem, nav, vis, wind, occ = _base(3, 3, resolution)
    dem[1, 1] = 100.0
    return CostMap(dem=dem, nav_density=nav, visibility=vis, wind_field=wind,
                   occupancy=occ, resolution_m=resolution, start=(0, 2), goal=(2, 0),
                   v_air=15.0, v_max=12.0,
                   cruise_altitude_agl_m=20.0, z_max_m=50.0)


def walled_map(height=7, width=7, resolution=10.0):
    """Full no-fly wall: no feasible path exists."""
    dem, nav, vis, wind, occ = _base(height, width, resolution)
    occ[:, width // 2] = True
    start, goal = (0, height - 1), (width - 1, 0)
    return CostMap(dem=dem, nav_density=nav, visibility=vis, wind_field=wind,
                   occupancy=occ, resolution_m=resolution, start=start, goal=goal,
                   v_air=15.0, v_max=12.0)


def brute_map(height=4, width=3, resolution=10.0, seed=3):
    """Small randomised map for exhaustive (brute-force) front comparison."""
    rng = np.random.default_rng(seed)
    dem, nav, vis, wind, occ = _base(height, width, resolution)
    dem[1:, :] += rng.uniform(0.0, 5.0, size=(height - 1, width))
    nav = np.clip(nav + rng.uniform(-0.3, 0.3, size=(height, width)), 0.05, 0.95)
    vis = np.clip(vis + rng.uniform(-0.3, 0.3, size=(height, width)), 0.05, 0.95)
    wind[:, :, 0] = rng.uniform(-3.0, 3.0, size=(height, width))
    wind[:, :, 1] = rng.uniform(-3.0, 3.0, size=(height, width))
    start, goal = (0, 0), (width - 1, height - 1)
    return CostMap(dem=dem, nav_density=nav, visibility=vis, wind_field=wind,
                   occupancy=occ, resolution_m=resolution, start=start, goal=goal,
                   v_air=15.0, v_max=12.0)
