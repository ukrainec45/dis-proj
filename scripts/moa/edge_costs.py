"""Edge cost evaluation for the 2.5D grid planning problem (deficit formulation).

All objectives are time-weighted and minimised (see
``uav_path_planning_context.md``):

    f1(P) = sum_e t(e)
    f2(P) = sum_e (1 - rho_nav(e)) * t(e)
    f3(P) = sum_e (1 - phi_vsb(e)) * t(e)

An edge is infeasible (returns ``None``) if it violates a hard constraint:
terrain occupancy/altitude, wind, or attainable ground speed.
"""

from dataclasses import dataclass

import numpy as np

# 8-connected grid neighbourhood (dx, dy).
NEIGHBOR_OFFSETS = [
    (dx, dy)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if (dx, dy) != (0, 0)
]

M_OBJECTIVES = 3


@dataclass
class CostMap:
    """All raster layers share a common UTM grid; cell (x, y) is column-major.

    ``start`` and ``goal`` are ``(x, y)`` cell indices.
    """

    dem: np.ndarray          # [H, W] ground elevation (m)
    nav_density: np.ndarray  # [H, W] in [0, 1] = max(visual, rugosity)
    visibility: np.ndarray   # [H, W] in [0, 1] (1 = full visibility)
    wind_field: np.ndarray   # [H, W, 2] [u_east, u_north] at cruise altitude
    occupancy: np.ndarray    # [H, W] bool, True = hard blocked cell
    resolution_m: float = 10.0
    start: tuple = (0, 0)
    goal: tuple = (0, 0)
    v_air: float = 15.0
    v_max: float = 12.0
    v_air_min_mps: float = 0.1
    v_air_max_mps: float | None = None
    z_max_m: float | None = None
    cruise_altitude_agl_m: float = 0.0
    battery_energy_wh: float | None = None
    energy_reserve_wh: float = 0.0
    cruise_power_w: float | None = None
    landing_sites: np.ndarray | None = None  # [H, W] bool, emergency landing cells
    min_turn_radius_m: float | None = None

    def __post_init__(self):
        H, W = self.dem.shape
        if self.nav_density.shape != (H, W):
            raise ValueError("nav_density must have shape (H, W)")
        if self.visibility.shape != (H, W):
            raise ValueError("visibility must have shape (H, W)")
        if self.wind_field.shape != (H, W, 2):
            raise ValueError("wind_field must have shape (H, W, 2)")
        if self.occupancy.shape != (H, W):
            raise ValueError("occupancy must have shape (H, W)")
        if self.landing_sites is not None and self.landing_sites.shape != (H, W):
            raise ValueError("landing_sites must have shape (H, W)")
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if self.v_air_min_mps <= 0 or self.v_air <= 0:
            raise ValueError("v_air and v_air_min_mps must be positive")
        if self.v_air < self.v_air_min_mps:
            raise ValueError("v_air must be at least v_air_min_mps")
        if self.v_air_max_mps is not None and self.v_air > self.v_air_max_mps:
            raise ValueError("v_air must not exceed v_air_max_mps")
        if self.v_max <= 0:
            raise ValueError("v_max must be positive")
        if self.min_turn_radius_m is not None and self.min_turn_radius_m < 0:
            raise ValueError("min_turn_radius_m must be non-negative")
        if self.z_max_m is not None and self.cruise_altitude_agl_m < 0:
            raise ValueError("cruise_altitude_agl_m must be non-negative")
        if self.battery_energy_wh is not None:
            if self.battery_energy_wh <= 0 or self.energy_reserve_wh < 0:
                raise ValueError("battery energy must be positive and reserve non-negative")
            if self.energy_reserve_wh >= self.battery_energy_wh:
                raise ValueError("energy reserve must be smaller than battery energy")
            if self.cruise_power_w is None or self.cruise_power_w <= 0:
                raise ValueError("cruise_power_w must be positive when battery energy is set")
        if self.occupancy[self.goal[1], self.goal[0]]:
            raise ValueError("goal cell is occupied")
        if self.occupancy[self.start[1], self.start[0]]:
            raise ValueError("start cell is occupied")
        if self.z_max_m is not None:
            if self.altitude_m(self.start) > self.z_max_m:
                raise ValueError("start altitude exceeds z_max_m")
            if self.altitude_m(self.goal) > self.z_max_m:
                raise ValueError("goal altitude exceeds z_max_m")

    @property
    def shape(self):
        return self.dem.shape

    def is_occupied(self, cell):
        return bool(self.occupancy[cell[1], cell[0]])

    @property
    def usable_flight_time_s(self):
        """Battery time available to the planned route, or infinity if unset."""
        if self.battery_energy_wh is None:
            return np.inf
        return 3600.0 * (self.battery_energy_wh - self.energy_reserve_wh) / self.cruise_power_w

    def altitude_m(self, cell):
        """MSL flight altitude at a cell for a constant-altitude-above-ground plan."""
        return float(self.dem[cell[1], cell[0]]) + self.cruise_altitude_agl_m


def sample_layer(layer, start, end, n=5):
    """Average layer values along the edge at n evenly spaced sample points."""
    values = []
    for t in np.linspace(0.0, 1.0, n):
        x = int(start[0] + t * (end[0] - start[0]))
        y = int(start[1] + t * (end[1] - start[1]))
        values.append(layer[y, x])
    return float(np.mean(values))


def sample_wind(wind_field, start, end):
    """Wind vector sampled at the edge midpoint."""
    x = int(0.5 * (start[0] + end[0]))
    y = int(0.5 * (start[1] + end[1]))
    return np.asarray(wind_field[y, x], dtype=float)


def turn_radius_m(previous, vertex, successor, cost_map):
    """Radius implied by two consecutive horizontal grid segments.

    The value is the osculating-circle approximation formed by the shorter
    adjacent segment and the heading change: ``min(l1,l2)/(2 sin(theta/2))``.
    A straight continuation has infinite radius.  It is deliberately based on
    horizontal geometry because aircraft turn-radius limits are horizontal.
    """
    first = np.array([vertex[0] - previous[0], vertex[1] - previous[1]], dtype=float)
    second = np.array([successor[0] - vertex[0], successor[1] - vertex[1]], dtype=float)
    l1 = float(np.linalg.norm(first)) * cost_map.resolution_m
    l2 = float(np.linalg.norm(second)) * cost_map.resolution_m
    if l1 == 0.0 or l2 == 0.0:
        return np.inf
    cosine = float(np.clip(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)), -1.0, 1.0))
    sin_half_turn = np.sqrt(max((1.0 - cosine) / 2.0, 0.0))
    if sin_half_turn <= 1e-12:
        return np.inf
    return min(l1, l2) / (2.0 * sin_half_turn)


def turn_is_feasible(previous, vertex, successor, cost_map):
    """Whether the heading change at ``vertex`` meets the configured radius."""
    return (cost_map.min_turn_radius_m is None or
            turn_radius_m(previous, vertex, successor, cost_map) + 1e-12 >=
            cost_map.min_turn_radius_m)


def edge_objectives(start, end, cost_map, v_air, v_max):
    """Objective contributions of edge start -> end, or None if infeasible.

    Returns a 3-vector ``(t, (1 - nav) * t, (1 - vsb) * t)`` where ``t`` is the
    flight time on the edge. Uses 3D Euclidean distance (altitude from DEM) and
    the wind-decomposed ground speed.
    """
    if cost_map.occupancy[end[1], end[0]]:
        return None
    if cost_map.z_max_m is not None and cost_map.altitude_m(end) > cost_map.z_max_m:
        return None

    # A diagonal may not pass through the corner shared by two occupied cells.
    # This treats occupancy as closed raster no-fly area rather than point masks.
    if start[0] != end[0] and start[1] != end[1]:
        if (cost_map.is_occupied((end[0], start[1])) or
                cost_map.is_occupied((start[0], end[1]))):
            return None

    wind = sample_wind(cost_map.wind_field, start, end)
    if np.linalg.norm(wind) >= v_max:
        return None

    dx = (end[0] - start[0]) * cost_map.resolution_m
    # Raster rows increase southward, while wind_field stores [east, north].
    # Convert the row displacement to the geographic northing convention.
    dy = -(end[1] - start[1]) * cost_map.resolution_m
    dz = float(cost_map.dem[end[1], end[0]]) - float(cost_map.dem[start[1], start[0]])
    horizontal_distance = np.hypot(dx, dy)
    distance = np.sqrt(horizontal_distance * horizontal_distance + dz * dz)
    if horizontal_distance <= 0.0:
        return (0.0, 0.0, 0.0)

    # Wind is horizontal.  Resolve it against a *unit horizontal* track vector,
    # then solve the 3-D airspeed constraint for horizontal ground speed V_h:
    # (V_h-u_along)^2 + u_cross^2 + (dz/d_h * V_h)^2 = v_air^2.
    track = np.array([dx, dy]) / horizontal_distance
    u_par = float(np.dot(wind, track))
    u_perp = float(wind[0] * track[1] - wind[1] * track[0])
    slope = dz / horizontal_distance
    a = 1.0 + slope * slope
    b = -2.0 * u_par
    c = u_par * u_par + u_perp * u_perp - v_air * v_air
    discriminant = b * b - 4.0 * a * c
    if discriminant <= 0.0:
        return None
    v_horizontal = (-b + np.sqrt(discriminant)) / (2.0 * a)
    if v_horizontal <= 0.0:
        return None
    t = horizontal_distance / v_horizontal

    nav = sample_layer(cost_map.nav_density, start, end)
    vsb = sample_layer(cost_map.visibility, start, end)

    return (t, (1.0 - nav) * t, (1.0 - vsb) * t)
