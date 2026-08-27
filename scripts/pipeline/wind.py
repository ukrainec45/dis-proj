"""Backward-compatible imports for shared wind helpers."""

from .utils.wind import (generate_synthetic_wind, get_wind_direction,
                         get_wind_speed)

__all__ = ["generate_synthetic_wind", "get_wind_direction", "get_wind_speed"]
