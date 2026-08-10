"""Reference-image preparation and ORB landmark extraction."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExtractedFeature:
    col: float
    row: float
    size: float
    angle: float
    response: float
    octave: int
    descriptor: bytes


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "ORB landmark extraction requires opencv-python. Install the landmark generator requirements."
        ) from exc
    return cv2


def prepare_feature_image(image, valid_mask, feature_band=None):
    """Produce a contrast-normalised uint8 image from 1+ GeoTIFF bands."""
    if image.ndim != 3:
        raise ValueError("image must have shape [bands, rows, cols]")
    bands = image.shape[0]
    if feature_band is not None:
        if not 1 <= feature_band <= bands:
            raise ValueError(f"feature_band must be between 1 and {bands}")
        gray = image[feature_band - 1].astype(np.float32)
    elif bands == 1:
        gray = image[0].astype(np.float32)
    elif bands >= 3:
        gray = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).astype(np.float32)
    else:
        gray = np.mean(image, axis=0, dtype=np.float32)
    usable = gray[valid_mask & np.isfinite(gray)]
    if usable.size == 0:
        raise ValueError("reference image contains no finite pixels inside the AOI")
    low, high = np.percentile(usable, (2, 98))
    if high <= low:
        return np.zeros_like(gray, dtype=np.uint8)
    scaled = np.clip((gray - low) * (255.0 / (high - low)), 0, 255)
    scaled[~np.isfinite(scaled)] = 0
    return scaled.astype(np.uint8)


def extract_orb_tile(feature_image, valid_mask, core, halo_px, max_features):
    """Extract core-owned ORB features from a tile plus context halo.

    ``core`` is ``(row_start, row_end, col_start, col_end)`` in full-image pixels.
    A feature is retained only if its centre lies in the core and AOI mask.
    """
    if max_features < 1:
        raise ValueError("max_features must be positive")
    cv2 = _cv2()
    rows, cols = feature_image.shape
    r0, r1, c0, c1 = core
    hr0, hr1 = max(0, r0 - halo_px), min(rows, r1 + halo_px)
    hc0, hc1 = max(0, c0 - halo_px), min(cols, c1 + halo_px)
    patch = feature_image[hr0:hr1, hc0:hc1]
    patch_mask = (valid_mask[hr0:hr1, hc0:hc1].astype(np.uint8) * 255)
    # Detect extra candidates before core filtering, then retain the strongest core set.
    # The OpenCV default (31 px) suppresses all candidates in small planner
    # tiles despite the halo. Keep enough context for the descriptor while
    # scaling the exclusion border to the available patch size.
    edge_threshold = min(31, max(5, min(patch.shape) // 4))
    orb = cv2.ORB_create(nfeatures=max_features * 3, edgeThreshold=edge_threshold)
    keypoints, descriptors = orb.detectAndCompute(patch, patch_mask)
    if descriptors is None:
        return []
    features = []
    for keypoint, descriptor in zip(keypoints, descriptors):
        col, row = keypoint.pt[0] + hc0, keypoint.pt[1] + hr0
        ir, ic = int(round(row)), int(round(col))
        if not (r0 <= row < r1 and c0 <= col < c1 and 0 <= ir < rows and 0 <= ic < cols):
            continue
        if not valid_mask[ir, ic]:
            continue
        features.append(ExtractedFeature(
            col=float(col), row=float(row), size=float(keypoint.size), angle=float(keypoint.angle),
            response=float(keypoint.response), octave=int(keypoint.octave), descriptor=bytes(descriptor),
        ))
    return sorted(features, key=lambda item: item.response, reverse=True)[:max_features]
