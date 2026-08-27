"""Pinhole geometry used by the synthetic VML benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    model: str
    width_px: int
    height_px: int
    pixel_size_um: float
    focal_length_mm: float
    frame_rate_hz: float

    def __post_init__(self):
        if self.width_px < 2 or self.height_px < 2:
            raise ValueError("camera resolution must be at least 2x2")
        if self.pixel_size_um <= 0 or self.focal_length_mm <= 0:
            raise ValueError("camera pixel size and focal length must be positive")
        if self.frame_rate_hz <= 0:
            raise ValueError("camera frame rate must be positive")

    @property
    def sensor_width_mm(self):
        return self.width_px * self.pixel_size_um / 1000.0

    @property
    def sensor_height_mm(self):
        return self.height_px * self.pixel_size_um / 1000.0

    @property
    def focal_length_px(self):
        return self.focal_length_mm * 1000.0 / self.pixel_size_um

    @property
    def intrinsics(self):
        f = self.focal_length_px
        return np.asarray([[f, 0.0, self.width_px / 2.0],
                           [0.0, f, self.height_px / 2.0],
                           [0.0, 0.0, 1.0]], dtype=float)

    @property
    def horizontal_fov_deg(self):
        return float(np.degrees(2.0 * np.arctan(
            self.sensor_width_mm / (2.0 * self.focal_length_mm))))

    @property
    def vertical_fov_deg(self):
        return float(np.degrees(2.0 * np.arctan(
            self.sensor_height_mm / (2.0 * self.focal_length_mm))))

    def footprint_m(self, altitude_agl_m):
        altitude = float(altitude_agl_m)
        if altitude <= 0:
            raise ValueError("altitude_agl_m must be positive")
        return (altitude * self.sensor_width_mm / self.focal_length_mm,
                altitude * self.sensor_height_mm / self.focal_length_mm)

    def rendered_shape(self, altitude_agl_m, ground_sample_distance_m):
        if ground_sample_distance_m <= 0:
            raise ValueError("ground_sample_distance_m must be positive")
        width_m, height_m = self.footprint_m(altitude_agl_m)
        width = max(32, int(np.ceil(width_m / ground_sample_distance_m)))
        height = max(32, int(np.ceil(height_m / ground_sample_distance_m)))
        return height, width
