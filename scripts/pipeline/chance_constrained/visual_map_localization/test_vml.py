import json

import cv2
import numpy as np
import pytest
from affine import Affine

from .camera_model import PinholeCamera
from .config import SiftPolicy, load_vml_config
from .experiment import aggregate_trials, analyse_rows
from .matcher import SiftMapMatcher, circular_error_deg
from .raster_sampler import nadir_transform


def _policy():
    return SiftPolicy(
        nfeatures=2000, ratio_threshold=.8, ransac_threshold_px=2,
        ransac_confidence=.995, minimum_good_matches=8,
        minimum_inliers=6, minimum_inlier_ratio=.4,
        maximum_median_reprojection_error_px=2)


def test_raspberry_pi_camera_geometry():
    camera = PinholeCamera("IMX296 + 6 mm", 1456, 1088, 3.45, 6.0, 30)
    assert camera.focal_length_px == pytest.approx(1739.1304)
    assert camera.horizontal_fov_deg == pytest.approx(45.43, abs=.02)
    assert camera.vertical_fov_deg == pytest.approx(34.73, abs=.02)
    assert camera.footprint_m(60) == pytest.approx((50.232, 37.536))
    np.testing.assert_allclose(camera.intrinsics[0:2, 2], (728, 544))


def test_nadir_transform_centre_and_yaw():
    transform = nadir_transform((100, 200), 0, .5, 40, 20)
    assert transform * (20, 10) == pytest.approx((100, 200))
    centre = np.asarray(transform * (20, 10))
    top = np.asarray(transform * (20, 0))
    direction = top - centre
    yaw = np.degrees(np.arctan2(direction[0], direction[1])) % 360
    assert yaw == pytest.approx(0)


def test_sift_localizes_known_crop_and_rejects_uniform_image():
    rng = np.random.default_rng(7)
    reference = rng.integers(0, 256, (420, 420, 3), dtype=np.uint8)
    for index in range(25):
        centre = tuple(map(int, rng.integers(20, 400, 2)))
        cv2.circle(reference, centre, int(rng.integers(3, 12)),
                   tuple(map(int, rng.integers(0, 256, 3))), -1)
    query = reference[120:300, 140:320].copy()
    transform = Affine(1, 0, 0, 0, -1, 420)
    result = SiftMapMatcher(_policy()).localize(query, reference, transform)
    assert result.accepted
    assert result.estimated_xy == pytest.approx((230, 210), abs=1.0)
    assert circular_error_deg(result.estimated_yaw_deg, 0) < 1

    uniform = np.full((180, 180, 3), 127, dtype=np.uint8)
    rejected = SiftMapMatcher(_policy()).localize(uniform, reference, transform)
    assert not rejected.accepted
    assert rejected.reason == "no_descriptors"


def test_aggregate_trials_keeps_conditional_axes():
    base = {"edge_index": 3, "scenario_id": "nominal",
            "altitude_agl_m": 60.0, "search_radius_m": 25.0,
            "runtime_ms": 10.0}
    rows = [dict(base, outcome="correct"), dict(base, outcome="reject")]
    artifact = aggregate_trials(rows, [3], ["nominal"], [60.0], [25.0])
    assert artifact["trial_count"].shape == (1, 1, 1, 1)
    assert artifact["p_correct"].item() == .5
    assert artifact["p_reject"].item() == .5
    assert artifact["p_false_accept"].item() == 0

    analysis_rows = [dict(base, outcome="correct", reject_reason="",
                          position_error_m=.5, yaw_error_deg=1),
                     dict(base, outcome="reject", reject_reason="no_descriptors",
                          position_error_m=np.nan, yaw_error_deg=np.nan)]
    analysis = analyse_rows(analysis_rows)
    assert analysis["rates"]["correct"] == .5
    assert analysis["reject_reasons"] == {"no_descriptors": 1}


def test_config_rejects_non_nadir_camera(tmp_path):
    artifact = tmp_path / "scene.npz"
    manifest = tmp_path / "manifest.json"
    artifact.write_bytes(b"placeholder")
    manifest.write_text("{}", encoding="utf-8")
    data = {
        "schema_version": 1,
        "scene": {"artifact": artifact.name, "manifest": manifest.name},
        "camera": {"model": "test", "resolution_px": [100, 100],
                   "pixel_size_um": 3, "focal_length_mm": 4,
                   "frame_rate_hz": 30, "mounting": "forward"},
        "experiment": {"sample_edges": 1, "altitudes_agl_m": [10],
                       "yaw_offsets_deg": [0], "search_radii_m": [5],
                       "repetitions": 1, "matching_gsd_m": .5,
                       "scenarios": [{"id": "nominal", "contrast": 1,
                                      "blur_sigma_px": 0, "haze_fraction": 0,
                                      "jpeg_quality": 100}]},
        "matcher": {"nfeatures": 100, "ratio_threshold": .75,
                    "ransac_threshold_px": 3, "ransac_confidence": .99,
                    "minimum_good_matches": 4, "minimum_inliers": 4,
                    "minimum_inlier_ratio": .2,
                    "maximum_median_reprojection_error_px": 3},
        "outcome": {"maximum_position_error_m": 5,
                    "maximum_yaw_error_deg": 5},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="mounting=nadir"):
        load_vml_config(path)
