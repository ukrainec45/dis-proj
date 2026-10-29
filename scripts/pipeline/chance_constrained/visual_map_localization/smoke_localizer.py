"""SIFT smoke localizer with an explicitly uncalibrated geometric XY covariance.

A fitted post-match R model and a validated yaw gate belong to the pilot stage.
No ground truth is accepted by this interface.
"""
import time

import cv2
import numpy as np

from .planar import rectify, search_crop


GEOMETRY_REJECTIONS = {
    "invalid_agl", "projection_crosses_horizon", "rectified_size_unsupported",
    "ill_conditioned_projection", "map_outside", "search_size_unsupported",
}


def localize(image, K, dist, rotation, agl, prior_mean, prior_covariance,
             search_probability_mass, reference, config, *, coverage_policy="strict"):
    if coverage_policy not in ("strict", "audit_clipped", "available_crop"):
        raise ValueError("Unknown coverage policy")
    start = time.perf_counter()
    diagnostics = {"R_calibrated": False, "yaw_available": False,
                   "coverage_policy": coverage_policy}

    def result(reason, z=None, R=None):
        return dict(status="accepted" if z is not None else "reject", reason=reason,
                    z=z, component_mask=[True, True, False] if z is not None else [False]*3,
                    R=R, diagnostics=diagnostics,
                    runtime_s=time.perf_counter()-start)

    gsd = config["rectified_gsd_m"]
    try:
        query, mask, footprint, anchor, condition = rectify(
            image, K, dist, rotation, agl, gsd, config["max_image_pixels"],
            config["maximum_projection_condition"])
        crop, transform, coverage = search_crop(
            reference, footprint, np.asarray(prior_mean), prior_covariance,
            search_probability_mass, config["max_image_pixels"], matching_gsd_m=gsd)
    except ValueError as error:
        if str(error) not in GEOMETRY_REJECTIONS:
            raise
        return result(str(error))
    diagnostics.update(projection_condition=condition, map_coverage=coverage,
                       query_shape_px=list(query.shape[:2]), reference_shape_px=list(crop.shape[:2]),
                       reference_gsd_m=[abs(transform.a), abs(transform.e)],
                       footprint_area_m2=float(abs(cv2.contourArea(footprint.astype(np.float32)))),
                       search_area_m2=float(crop.shape[0]*crop.shape[1]*abs(transform.determinant)))
    if coverage < config["minimum_map_coverage"] and coverage_policy == "strict":
        return result("insufficient_map_coverage")
    policy = config["sift"]
    sift = cv2.SIFT_create(nfeatures=policy["nfeatures"])
    gray = lambda rgb: cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Erosion keeps keypoint support away from warped black borders.
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    qk, qd = sift.detectAndCompute(gray(query), mask)
    rk, rd = sift.detectAndCompute(gray(crop), None)
    diagnostics.update(query_keypoints=len(qk), reference_keypoints=len(rk))
    if qd is None or rd is None or len(rd) < 2:
        return result("no_descriptors")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(qd, rd, k=2)
    good = [a for pair in pairs if len(pair) == 2 for a, b in [pair]
            if a.distance < policy["ratio_threshold"]*b.distance]
    diagnostics["good_matches"] = len(good)
    if len(good) < policy["minimum_good_matches"]:
        return result("insufficient_matches")
    source = np.float32([qk[m.queryIdx].pt for m in good])
    target = np.float32([rk[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(0)
    H, inliers = cv2.findHomography(source, target, cv2.RANSAC,
                                   policy["ransac_threshold_px"])
    if H is None or inliers is None or not np.isfinite(H).all():
        return result("homography_failed")
    valid = inliers.ravel().astype(bool)
    count = int(valid.sum())
    residual = cv2.perspectiveTransform(source[:, None], H)[:, 0] - target
    median = float(np.median(np.linalg.norm(residual[valid], axis=1))) if count else None
    diagnostics.update(inliers=count, inlier_ratio=count/len(good),
                       median_reprojection_error_px=median)
    if count < policy["minimum_inliers"] or count/len(good) < policy["minimum_inlier_ratio"]:
        return result("insufficient_inliers")
    if median > policy["maximum_median_reprojection_error_px"]:
        return result("high_reprojection_error")
    point = H @ np.r_[anchor, 1.]
    if abs(point[2]) < 1e-9 or not np.isfinite(point).all():
        return result("invalid_position_projection")
    xy = transform * tuple(point[:2]/point[2])
    # Translation residual approximation only; not a calibrated pose covariance.
    pixel_cov = np.cov(residual[valid].T)/count
    scale = np.diag([transform.a, transform.e])
    R = scale @ pixel_cov @ scale.T + np.eye(2)*config["geometric_R_floor_m"]**2
    diagnostics["R_method"] = "uncalibrated_translation_residual_approximation"
    return result("accepted_xy_uncalibrated", list(xy), R.tolist())
