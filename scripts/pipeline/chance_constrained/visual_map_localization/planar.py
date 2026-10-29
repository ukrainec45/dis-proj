"""Planar virtual-nadir geometry. DSM never enters a pose solver."""
import cv2
import numpy as np


def covariance_extent(covariance, probability_mass):
    covariance = np.asarray(covariance, float)
    if (covariance.shape != (2, 2) or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T)
            or np.linalg.eigvalsh(covariance).min() < 0):
        raise ValueError("Expected symmetric PSD XY covariance")
    if not 0 < probability_mass < 1:
        raise ValueError("search_probability_mass must be in (0,1)")
    return np.sqrt(-2 * np.log1p(-probability_mass) * np.diag(covariance))


def ground_offsets(pixels, K, dist, rotation, agl):
    if not np.isfinite(agl) or agl <= 0:
        raise ValueError("invalid_agl")
    rays = cv2.undistortPoints(np.asarray(pixels, np.float64).reshape(-1, 1, 2),
                               K, dist).reshape(-1, 2)
    rays = np.column_stack([rays, np.ones(len(rays))]) @ rotation.T
    if np.any(rays[:, 2] >= -1e-8):
        raise ValueError("projection_crosses_horizon")
    return rays[:, :2] * (-agl / rays[:, 2, None])


def rectify(image, K, dist, rotation, agl, gsd, max_pixels, max_condition):
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], float)
    offsets = ground_offsets(corners, K, dist, rotation, agl)
    lower, upper = offsets.min(axis=0), offsets.max(axis=0)
    shape = np.ceil((upper-lower)/gsd).astype(int) + 1
    if np.any(shape < 2) or np.prod(shape.astype(float)) > max_pixels:
        raise ValueError("rectified_size_unsupported")
    target = np.column_stack([(offsets[:, 0]-lower[0])/gsd,
                              (upper[1]-offsets[:, 1])/gsd]).astype(np.float32)
    undistorted = cv2.undistort(image, K, dist, None, K)
    source = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, dist,
                                  P=K).reshape(-1, 2).astype(np.float32)
    H = cv2.getPerspectiveTransform(source, target)
    p = np.array([w/2, h/2, 1.])
    q = H @ p
    jac = (H[:2, :2]*q[2] - np.outer(q[:2], H[2, :2])) / q[2]**2
    condition = float(np.linalg.cond(jac))
    if not np.isfinite(condition) or condition > max_condition:
        raise ValueError("ill_conditioned_projection")
    query = cv2.warpPerspective(undistorted, H, tuple(map(int, shape)))
    source_mask = cv2.undistort(np.full((h, w), 255, np.uint8), K, dist, None, K)
    valid = cv2.warpPerspective(source_mask, H, tuple(map(int, shape)),
                                flags=cv2.INTER_NEAREST)
    camera_pixel = np.array([-lower[0]/gsd, upper[1]/gsd])
    return query, valid, offsets, camera_pixel, condition


def search_bounds(reference, footprint, prior_xy, covariance, mass):
    """Native pixel bounds and coverage, without allocating a crop."""
    extent = covariance_extent(covariance, mass)
    lower = np.min(footprint, axis=0) + prior_xy - extent
    upper = np.max(footprint, axis=0) + prior_xy + extent
    inverse = ~reference.transform
    pixels = np.array([inverse * tuple(lower), inverse * tuple(upper)])
    start = np.floor(pixels.min(axis=0)).astype(int)
    end = np.ceil(pixels.max(axis=0)).astype(int) + 1
    requested = np.prod((end-start).astype(float))
    h, w = reference.image.shape[:2]
    a, b = np.maximum(start, 0), np.minimum(end, [w, h])
    coverage = float(np.prod(np.maximum(b-a, 0))/requested)
    return start, end, a, b, coverage


def search_crop(reference, footprint, prior_xy, covariance, mass, max_pixels,
                *, matching_gsd_m=None):
    start, end, a, b, coverage = search_bounds(
        reference, footprint, prior_xy, covariance, mass)
    if np.any(b-a < 2):
        raise ValueError("map_outside")
    native_shape = b-a
    output_shape = native_shape.copy()
    if matching_gsd_m is not None:
        if not np.isfinite(matching_gsd_m) or matching_gsd_m <= 0:
            raise ValueError("matching_gsd_m must be finite and positive")
        resolution = np.array([abs(reference.transform.a), abs(reference.transform.e)])
        output_shape = np.maximum(2, np.ceil(native_shape*resolution/matching_gsd_m)).astype(int)
    # Limit the raster actually sent to the matcher, not the native map view.
    if np.prod(output_shape.astype(float)) > max_pixels:
        raise ValueError("search_size_unsupported")
    transform = reference.transform * type(reference.transform).translation(*a)
    crop = reference.image[a[1]:b[1], a[0]:b[0]]  # NumPy view; no native crop copy.
    if not np.array_equal(output_shape, native_shape):
        scale = native_shape/output_shape
        interpolation = cv2.INTER_AREA if np.all(scale >= 1) else cv2.INTER_LINEAR
        crop = cv2.resize(crop, tuple(map(int, output_shape)), interpolation=interpolation)
        # OpenCV resize aligns pixel centres: src=(dst+.5)*scale-.5.
        transform = (transform * type(transform).translation(*((scale-1)/2))
                     * type(transform).scale(*scale))
    return crop, transform, coverage

