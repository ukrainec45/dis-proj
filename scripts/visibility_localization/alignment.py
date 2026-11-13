"""Yaw-aligned map views; DSM and world coordinates stay in the original frame."""
from dataclasses import dataclass

import cv2
import numpy as np

from .core import digest


def prior_yaw_deg(camera_to_enu):
    """Recover the author's ZYX/NED yaw using the verified nadir camera mounting.

    The input is the simulated/measured camera orientation, never dataset Euler.
    These are the inverse mounting/frame conversions used by read_sample().
    """
    ned_to_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.]])
    camera_to_frd = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    attitude = ned_to_enu.T @ np.asarray(camera_to_enu) @ camera_to_frd.T
    if not np.isfinite(attitude).all() or np.hypot(attitude[0, 0], attitude[1, 0]) < 1e-8:
        raise ValueError('Undefined prior yaw')
    return float(np.degrees(np.arctan2(attitude[1, 0], attitude[0, 0])))


@dataclass
class YawReference:
    """A rotated raster with original map/DSM georeferencing for upstream PnP."""
    original: object
    bgr: np.ndarray
    map_transform: np.ndarray
    yaw_deg: float
    identity: dict

    def __getattr__(self, name):
        return getattr(self.original, name)

    def pixels_to_enu(self, pixels):
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        inverse = cv2.invertAffineTransform(self.map_transform)
        unrotated = np.column_stack([pixels, np.ones(len(pixels))]) @ inverse.T
        return self.original.pixels_to_enu(unrotated)

    def enu_to_pixels(self, enu):
        original = (np.asarray(enu)*[1., -1.]-self.origin)/self.resolution
        return np.append(original, 1.) @ self.map_transform.T


def align_reference(reference, prior, rotate):
    """Use the author's rotation, preserving its interpolation and expanded canvas."""
    yaw = prior_yaw_deg(prior.camera_to_enu)
    image, transform = rotate(reference.bgr, yaw)
    image.flags.writeable = False
    identity = dict(reference.identity, yaw_alignment=digest(dict(
        yaw_deg=yaw, transform=np.asarray(transform), source_shape=reference.bgr.shape)))
    return YawReference(reference, image, np.asarray(transform), yaw, identity)
