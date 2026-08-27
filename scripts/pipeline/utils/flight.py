"""Direction-aware edge flight-time calculations."""

import numpy as np


def edge_travel_time_s(dx_m, dy_m, dz_m, wind_uv_mps, airspeed_mps):
    """Return flight time under a 3-D airspeed constraint, or ``None``.

    ``dx_m`` is eastward and ``dy_m`` northward. Wind is ``[east, north]``.
    The model is shared by the legacy MOA* baseline and the new edge pipeline.
    """
    if airspeed_mps <= 0:
        raise ValueError("airspeed_mps must be positive")
    horizontal_distance = float(np.hypot(dx_m, dy_m))
    if horizontal_distance <= 0.0:
        return 0.0
    wind = np.asarray(wind_uv_mps, dtype=float)
    if wind.shape != (2,) or not np.all(np.isfinite(wind)):
        return None
    track = np.array([dx_m, dy_m], dtype=float) / horizontal_distance
    u_parallel = float(np.dot(wind, track))
    u_cross = float(wind[0] * track[1] - wind[1] * track[0])
    slope = float(dz_m) / horizontal_distance
    a = 1.0 + slope * slope
    b = -2.0 * u_parallel
    c = u_parallel * u_parallel + u_cross * u_cross - airspeed_mps * airspeed_mps
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    horizontal_speed = (-b + np.sqrt(discriminant)) / (2.0 * a)
    return horizontal_distance / horizontal_speed if horizontal_speed > 0.0 else None

