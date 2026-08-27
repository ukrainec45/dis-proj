"""Backward-compatible imports for shared visibility helpers."""

from .utils.visibility import (aggregate_to_planner_grid, combine_sources,
                               compute_visibility, gaussian_density,
                               generate_local_smoke_sources,
                               generate_smoke_sources, visibility_from_zones)

__all__ = ["aggregate_to_planner_grid", "combine_sources", "compute_visibility",
           "gaussian_density", "generate_local_smoke_sources",
           "generate_smoke_sources", "visibility_from_zones"]
