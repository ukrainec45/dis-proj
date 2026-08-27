"""Georeferenced nadir rendering from a source orthophoto."""

from __future__ import annotations

import cv2
import numpy as np
from affine import Affine


def nadir_transform(center_xy, yaw_deg, gsd_m, width, height):
    """Return an image-to-map affine for clockwise yaw from north."""
    yaw = np.radians(float(yaw_deg))
    right = np.asarray([np.cos(yaw), -np.sin(yaw)])
    down = np.asarray([-np.sin(yaw), -np.cos(yaw)])
    centre = np.asarray(center_xy, dtype=float)
    origin = centre - right * gsd_m * (width / 2.0) - down * gsd_m * (height / 2.0)
    return Affine(right[0] * gsd_m, down[0] * gsd_m, origin[0],
                  right[1] * gsd_m, down[1] * gsd_m, origin[1])


class OrthophotoSampler:
    def __init__(self, path, target_crs):
        import rasterio

        self._rasterio = rasterio
        self._source = rasterio.open(path)
        if self._source.count < 3:
            self._source.close()
            raise ValueError("VML orthophoto must contain at least three bands")
        self.target_crs = target_crs

    def close(self):
        self._source.close()

    def coverage_at_points(self, points_xy):
        """Return whether source RGB is nonzero at target-CRS map points."""
        from rasterio.warp import transform

        points = np.asarray(points_xy, dtype=float)
        xs, ys = transform(self.target_crs, self._source.crs,
                           points[:, 0].tolist(), points[:, 1].tolist())
        values = np.asarray(list(self._source.sample(zip(xs, ys), indexes=(1, 2, 3))))
        return np.any(values != 0, axis=1)

    def coverage_fraction_around_points(self, points_xy, half_extent_m):
        """Probe a 3x3 stencil covering the largest requested search crop."""
        points = np.asarray(points_xy, dtype=float)
        offsets = np.asarray([(dx, dy) for dy in (-half_extent_m, 0.0, half_extent_m)
                              for dx in (-half_extent_m, 0.0, half_extent_m)])
        probes = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        valid = self.coverage_at_points(probes).reshape(len(points), len(offsets))
        return valid.mean(axis=1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def render(self, transform, height, width):
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        destination = np.zeros((3, int(height), int(width)), dtype=np.uint8)
        for destination_index, source_index in enumerate((1, 2, 3)):
            reproject(
                source=self._rasterio.band(self._source, source_index),
                destination=destination[destination_index],
                src_transform=self._source.transform,
                src_crs=self._source.crs,
                dst_transform=transform,
                dst_crs=self.target_crs,
                src_nodata=self._source.nodata,
                dst_nodata=0,
                resampling=Resampling.bilinear,
            )
        rgb = np.moveaxis(destination, 0, -1)
        # The Pyatka RGB source has no nodata mask. Treat an all-zero pixel as
        # outside coverage; real black pixels are negligible for the coverage
        # guard and remain valid input to the matcher in all other respects.
        valid = np.any(rgb != 0, axis=2)
        return rgb, valid

    def render_query(self, camera, center_xy, altitude_agl_m, yaw_deg, gsd_m):
        height, width = camera.rendered_shape(altitude_agl_m, gsd_m)
        transform = nadir_transform(center_xy, yaw_deg, gsd_m, width, height)
        image, valid = self.render(transform, height, width)
        return image, valid, transform

    def render_reference(self, center_xy, half_extent_m, gsd_m):
        from rasterio.transform import from_origin

        size = max(64, int(np.ceil(2.0 * half_extent_m / gsd_m)))
        left = float(center_xy[0]) - half_extent_m
        top = float(center_xy[1]) + half_extent_m
        transform = from_origin(left, top, gsd_m, gsd_m)
        image, valid = self.render(transform, size, size)
        return image, valid, transform


def degrade_image(image, degradation):
    result = np.asarray(image, dtype=np.float32)
    result = (result - 127.5) * degradation.contrast + 127.5
    if degradation.haze_fraction > 0:
        result = ((1.0 - degradation.haze_fraction) * result +
                  degradation.haze_fraction * 220.0)
    result = np.clip(result, 0, 255).astype(np.uint8)
    if degradation.blur_sigma_px > 0:
        result = cv2.GaussianBlur(result, (0, 0), degradation.blur_sigma_px)
    if degradation.jpeg_quality < 100:
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, degradation.jpeg_quality])
        if not ok:
            raise RuntimeError("OpenCV failed to encode the degraded query")
        result = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR),
                              cv2.COLOR_BGR2RGB)
    return result
