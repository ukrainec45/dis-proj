"""SIFT image-to-map matcher with an explicit reject policy."""

from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchResult:
    accepted: bool
    reason: str
    estimated_xy: tuple[float, float] | None
    estimated_yaw_deg: float | None
    query_keypoints: int
    reference_keypoints: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    median_reprojection_error_px: float
    runtime_ms: float


def circular_error_deg(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


class SiftMapMatcher:
    def __init__(self, policy):
        self.policy = policy
        self.sift = cv2.SIFT_create(nfeatures=policy.nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    @staticmethod
    def _reject(reason, q_count, r_count, good, start, *, inliers=0,
                ratio=0.0, reprojection=np.inf):
        return MatchResult(False, reason, None, None, q_count, r_count, good,
                           inliers, ratio, reprojection,
                           (time.perf_counter() - start) * 1000.0)

    def localize(self, query_rgb, reference_rgb, reference_transform):
        start = time.perf_counter()
        query = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY)
        reference = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
        q_keypoints, q_descriptors = self.sift.detectAndCompute(query, None)
        r_keypoints, r_descriptors = self.sift.detectAndCompute(reference, None)
        q_count, r_count = len(q_keypoints), len(r_keypoints)
        if q_descriptors is None or r_descriptors is None:
            return self._reject("no_descriptors", q_count, r_count, 0, start)
        pairs = self.matcher.knnMatch(q_descriptors, r_descriptors, k=2)
        good = [first for pair in pairs if len(pair) == 2
                for first, second in [pair]
                if first.distance < self.policy.ratio_threshold * second.distance]
        if len(good) < self.policy.minimum_good_matches:
            return self._reject("insufficient_good_matches", q_count, r_count,
                                len(good), start)
        query_points = np.float32([q_keypoints[item.queryIdx].pt for item in good])
        reference_points = np.float32([r_keypoints[item.trainIdx].pt for item in good])
        cv2.setRNGSeed(0)
        homography, mask = cv2.findHomography(
            query_points, reference_points, cv2.RANSAC,
            self.policy.ransac_threshold_px,
            confidence=self.policy.ransac_confidence,
            maxIters=5000)
        if homography is None or mask is None or not np.isfinite(homography).all():
            return self._reject("homography_failed", q_count, r_count, len(good), start)
        inlier_mask = mask.ravel().astype(bool)
        inliers = int(inlier_mask.sum())
        inlier_ratio = inliers / len(good)
        projected = cv2.perspectiveTransform(
            query_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(projected - reference_points, axis=1)
        median_error = (float(np.median(errors[inlier_mask]))
                        if inliers else np.inf)
        if inliers < self.policy.minimum_inliers:
            return self._reject("insufficient_inliers", q_count, r_count,
                                len(good), start, inliers=inliers,
                                ratio=inlier_ratio, reprojection=median_error)
        if inlier_ratio < self.policy.minimum_inlier_ratio:
            return self._reject("low_inlier_ratio", q_count, r_count, len(good),
                                start, inliers=inliers, ratio=inlier_ratio,
                                reprojection=median_error)
        if median_error > self.policy.maximum_median_reprojection_error_px:
            return self._reject("high_reprojection_error", q_count, r_count,
                                len(good), start, inliers=inliers,
                                ratio=inlier_ratio, reprojection=median_error)

        height, width = query.shape
        diagnostic_points = np.float32([
            [width / 2.0, height / 2.0],
            [width / 2.0, height / 2.0 - max(10.0, height / 4.0)],
        ]).reshape(-1, 1, 2)
        mapped_pixels = cv2.perspectiveTransform(
            diagnostic_points, homography).reshape(-1, 2)
        if not np.isfinite(mapped_pixels).all():
            return self._reject("invalid_projection", q_count, r_count,
                                len(good), start, inliers=inliers,
                                ratio=inlier_ratio, reprojection=median_error)
        mapped_xy = np.asarray([
            reference_transform * tuple(point) for point in mapped_pixels
        ], dtype=float)
        direction = mapped_xy[1] - mapped_xy[0]
        if np.linalg.norm(direction) <= 1e-9:
            return self._reject("degenerate_yaw", q_count, r_count, len(good),
                                start, inliers=inliers, ratio=inlier_ratio,
                                reprojection=median_error)
        yaw = (np.degrees(np.arctan2(direction[0], direction[1])) + 360.0) % 360.0
        return MatchResult(
            True, "accepted", tuple(map(float, mapped_xy[0])), float(yaw),
            q_count, r_count, len(good), inliers, float(inlier_ratio),
            median_error, (time.perf_counter() - start) * 1000.0)
