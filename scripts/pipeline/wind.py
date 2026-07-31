"""Synthetic wind-field generation retained from the Colab notebook."""

import numpy as np


def get_wind_speed(u, v):
    return np.sqrt(u**2 + v**2)


def get_wind_direction(u, v):
    """Meteorological direction in degrees (0 north, 90 east)."""
    return (np.degrees(np.arctan2(u, v)) + 360) % 360


def generate_synthetic_wind(rows, cols, base_speed=5.0, base_dir_deg=225,
                            perturbation_scale=1.5, smoothness=2.0, seed=42):
    """Generate the same smooth Gaussian-random-field wind used in the notebook."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.RandomState(seed)
    radians = np.radians(base_dir_deg)
    u_base, v_base = base_speed * np.sin(radians), base_speed * np.cos(radians)
    delta_u = gaussian_filter(rng.normal(0, 1, (rows, cols)), sigma=smoothness) * perturbation_scale
    delta_v = gaussian_filter(rng.normal(0, 1, (rows, cols)), sigma=smoothness) * perturbation_scale
    return u_base + delta_u, v_base + delta_v
