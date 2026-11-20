"""Map-only features. No sample/GT object is accepted by this module."""
import cv2
import numpy as np

from scripts.visibility_localization.core import footprint, nadir_angle

from .contracts import CONDITIONS, CorrectionQuery

GEOMETRY = ('sigma_major_m', 'sigma_minor_m', 'yaw_sigma_deg', 'tilt_sigma_deg',
            'z_sigma_m', 'agl_m', 'nadir_deg', 'fov_x_deg', 'fov_y_deg',
            'map_gsd_x_m', 'map_gsd_y_m', 'search_width_m', 'search_height_m',
            'map_coverage', 'fog_extreme', 'smoke_extreme')
MAP_FEATURES = ('gray_std', 'entropy_bits', 'gradient_mean', 'corner_density', 'repeat_correlation')
ALL_FEATURES = GEOMETRY + MAP_FEATURES
FEATURE_VERSION = 'map-prior-camp-v2'


def extract_features(query: CorrectionQuery, reference):
    """Reference implements pixels_to_enu, plane_at and bgr/resolution/origin.

    Reference arrays use the existing AnyVisLoc map adapter; metric EPSG rasters
    are adapted in planning.py. Fixed 256px sampling bounds CPU/memory use.
    """
    from pyproj import CRS
    if hasattr(reference, 'crs'):
        if query.crs == 'LOCAL_ENU' or CRS.from_user_input(query.crs) != CRS.from_user_input(reference.crs):
            raise ValueError('query_map_crs_mismatch')
    elif query.crs != 'LOCAL_ENU':
        raise ValueError('query_map_crs_mismatch')
    if query.scenario not in CONDITIONS:
        raise ValueError('unsupported_scenario')
    z = reference.plane_at(query.position[:2])
    visible, _ = footprint(query.position, query.camera_to_enu, query.K, query.dist,
                           query.image_shape, z, center_crop=True)
    sigma = np.sqrt(np.linalg.eigvalsh(query.covariance_xy))
    radius = 3 * sigma[-1]
    x0, y0, x1, y1 = visible.bounds
    bounds = np.array([x0-radius, y0-radius, x1+radius, y1+radius])
    corners = reference.pixels_to_enu([[0, 0], [reference.bgr.shape[1]-1, reference.bgr.shape[0]-1]])
    low, high = corners.min(axis=0), corners.max(axis=0)
    area = np.maximum(0, np.minimum(bounds[2:], high)-np.maximum(bounds[:2], low)).prod()
    coverage = float(area / np.prod(bounds[2:]-bounds[:2]))
    if coverage <= 0:
        raise ValueError('no_map_coverage')
    xs = np.linspace(bounds[0], bounds[2], 256)
    ys = np.linspace(bounds[1], bounds[3], 256)
    xx, yy = np.meshgrid(xs, ys)
    # Invert the actual adapter mapping (also supports planning raster affine).
    if hasattr(reference, 'enu_to_pixels'):
        uv = reference.enu_to_pixels(np.column_stack([xx.ravel(), yy.ravel()]))
    else:
        uv = (np.column_stack([xx.ravel(), -yy.ravel()])-reference.origin)/reference.resolution
    u, v = uv[:, 0].reshape(256, 256), uv[:, 1].reshape(256, 256)
    valid = (u >= 0) & (v >= 0) & (u <= reference.bgr.shape[1]-1) & (v <= reference.bgr.shape[0]-1)
    patch = cv2.remap(reference.bgr, u.astype('float32'), v.astype('float32'), cv2.INTER_LINEAR)
    if hasattr(reference, 'valid_pixels'):
        valid &= cv2.remap(reference.valid_pixels.astype('uint8'), u.astype('float32'),
                           v.astype('float32'), cv2.INTER_NEAREST).astype(bool)
    if valid.sum() < 16:
        raise ValueError('insufficient_map_pixels')
    coverage = min(coverage, float(valid.mean()))
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(float) / 255
    hist = np.histogram(gray[valid], bins=32, range=(0, 1))[0].astype(float)
    hist = hist[hist > 0] / hist.sum()
    gy, gx = np.gradient(gray)
    # Require neighbouring valid pixels; avoid counting padding edges as texture.
    interior = cv2.erode(valid.astype('uint8'), np.ones((5, 5), 'uint8')).astype(bool)
    if interior.sum() < 16:
        raise ValueError('insufficient_map_pixels')
    corners_response = cv2.cornerMinEigenVal(gray.astype('float32'), 3)
    threshold = max(float(corners_response[interior].max()) * .01, 1e-8)
    correlations = []
    for axis in (0, 1):
        for shift in (16, 32, 64):
            rolled = np.roll(gray, shift, axis)
            mask = interior & np.roll(interior, shift, axis)
            if axis == 0:
                mask[:shift] = False
            else:
                mask[:, :shift] = False
            if mask.sum() > 16 and gray[mask].std() > 1e-8 and rolled[mask].std() > 1e-8:
                correlations.append(float(np.corrcoef(gray[mask], rolled[mask])[0, 1]))
    features = dict(sigma_major_m=float(sigma[-1]), sigma_minor_m=float(sigma[0]),
                    yaw_sigma_deg=float(query.yaw_sigma_deg), tilt_sigma_deg=float(query.tilt_sigma_deg),
                    z_sigma_m=float(query.z_sigma_m), agl_m=float(query.position[2]-z),
                    nadir_deg=nadir_angle(query.camera_to_enu),
                    fov_x_deg=float(np.degrees(2*np.arctan(query.image_shape[1]/(2*query.K[0, 0])))),
                    fov_y_deg=float(np.degrees(2*np.arctan(query.image_shape[0]/(2*query.K[1, 1])))),
                    map_gsd_x_m=float(getattr(reference, 'source_resolution', reference.resolution)[0]),
                    map_gsd_y_m=float(getattr(reference, 'source_resolution', reference.resolution)[1]),
                    search_width_m=float(bounds[2]-bounds[0]), search_height_m=float(bounds[3]-bounds[1]),
                    map_coverage=coverage, fog_extreme=float(query.scenario == 'fog_extreme'),
                    smoke_extreme=float(query.scenario == 'smoke_extreme'),
                    gray_std=float(gray[valid].std()), entropy_bits=float(-(hist*np.log2(hist)).sum()),
                    gradient_mean=float(np.hypot(gx, gy)[interior].mean()),
                    corner_density=float((corners_response[interior] > threshold).mean()),
                    repeat_correlation=max(correlations, default=0.))
    # Return the same prior-centred map context to the image candidate.
    return features, cv2.cvtColor(patch, cv2.COLOR_BGR2RGB), dict(
        feature_version=FEATURE_VERSION, bounds_xy_m=bounds.tolist(),
        approximation='local DSM plane, 3-sigma major-axis search, 256px map context')
