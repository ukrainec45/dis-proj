"""Array-only observation models and geometry; no neural-network dependency."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon, Point

UPSTREAM_COMMIT = "6c8f7980f35defd0676d82e71b19fcd214da4033"
CONDITIONS = ("clear", "fog_weak", "fog_moderate", "fog_strong", "smoke_weak", "smoke_strong")


def defaults():
    return dict(seed=42, reference_mode="satellite", matching_method="SP_LG", map_alignment="none", scenes=[],
        nadir_max_deg=15., preferred_nadir_deg=10., sample_count=24,
        pose_convention="ned_attitude", sigma_xy_m=[20., 50., 100.], trials=3,
        profiles={"small": dict(yaw_deg=2., tilt_deg=1., z_m=2.),
                  "main": dict(yaw_deg=5., tilt_deg=2., z_m=5.),
                  "large": dict(yaw_deg=15., tilt_deg=5., z_m=10.)},
        active_profiles=["main"], retrieval_modes=["prior"], top_k=5,
        search_sigma=3., gallery_overlap=.5, patch_scale=1.5,
        retrieval_size=384, retrieval_batch=8, max_gallery_tiles=12000,
        max_patch_pixels=4000000, max_image_side=1600, match_keypoints=3000,
        fallback_patch_sizes_m=[64., 128., 256.],
        gallery_cache_entries=2, match_resize_ratio=.4, pnp_reprojection_px=8., pnp_iterations=2000,
        pnp_confidence=.999, min_matches=12, min_inliers=8,
        min_inlier_ratio=.25, max_median_reprojection_px=4.,
        min_positive_depth_fraction=.95, success_threshold_m=10.,
        recall_overlap_thresholds=[.25, .5, .75],
        fog={"fog_weak": .8, "fog_moderate": .55, "fog_strong": .3},
        airlight_rgb=[.92, .94, .96],
        smoke={"smoke_weak": dict(coverage=.35, transmission=.55, softness=.08,
                                  color_rgb=[.65, .63, .60]),
               "smoke_strong": dict(coverage=.7, transmission=.2, softness=.08,
                                    color_rgb=[.50, .48, .45])})


def validate_config(config):
    if config.get('map_alignment', 'none') not in ('none', 'prior_yaw'):
        raise ValueError('Unknown map alignment; choose none or prior_yaw')
    if config["reference_mode"] not in ("satellite","aerial"): raise ValueError("Unknown reference mode")
    if config.get('matching_method', 'SP_LG') not in ('SP_LG', 'Roma'):
        raise ValueError('Unknown matching method; choose SP_LG or Roma')
    if not 0 < config["nadir_max_deg"] <= 15: raise ValueError("Nadir threshold must be in (0,15]")
    if not 0 <= config["preferred_nadir_deg"] <= config["nadir_max_deg"]: raise ValueError("Invalid preferred angle")
    if not config["sigma_xy_m"] or min(config["sigma_xy_m"]) <= 0: raise ValueError("Positive XY sigmas required")
    if not config["retrieval_modes"] or not set(config["retrieval_modes"]) <= {"prior","global"}: raise ValueError("Invalid retrieval modes")
    for key in ["trials","sample_count","top_k","retrieval_batch","max_gallery_tiles","max_patch_pixels", "max_image_side","gallery_cache_entries"]:
        if not isinstance(config[key],int) or config[key] < 1: raise ValueError(f"Invalid {key}")
    if not 0 <= config["gallery_overlap"] < 1: raise ValueError("Invalid tile overlap")
    if config['max_patch_pixels'] < 4: raise ValueError('max_patch_pixels must be at least 4')
    sizes = config['fallback_patch_sizes_m']
    if not sizes or any(not np.isfinite(s) or s <= 0 for s in sizes):
        raise ValueError('Positive fallback patch sizes required')
    for key in ["search_sigma","patch_scale","pnp_reprojection_px","max_median_reprojection_px","success_threshold_m"]:
        if config[key] <= 0 or not np.isfinite(config[key]): raise ValueError(f"Invalid {key}")
    for name in config["active_profiles"]:
        if min(config["profiles"][name].values()) < 0: raise ValueError("Negative profile sigma")
    if not config["active_profiles"]: raise ValueError("No active profiles")
    for t in config["fog"].values():
        if not 0 <= t <= 1: raise ValueError("Invalid fog transmission")
    for p in config["smoke"].values():
        if not 0 < p["coverage"] < 1 or not 0 <= p["transmission"] <= 1 or p["softness"] <= 0:
            raise ValueError("Invalid smoke configuration")
    return config


def canonical(value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=convert, allow_nan=False)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical(parts).encode()).digest()[:4], "little") % (2**31-1)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024**2), b""): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array)
    # Preserve the exact digest without materializing two map-sized byte strings.
    h = hashlib.sha256(str((a.shape, a.dtype)).encode())
    if a.size:
        h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def degrade(rgb, condition, config, seed):
    """RGB uint8 -> RGB uint8 and transmission; empirical sRGB-domain mixing."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected uint8 HWC RGB")
    h, w = rgb.shape[:2]
    t = np.ones((h, w), np.float32)
    airlight = np.asarray(config["airlight_rgb"], float)
    coverage = 0.
    if condition in config["fog"]:
        t.fill(config["fog"][condition])
    elif condition in config["smoke"]:
        p = config["smoke"][condition]
        rng = np.random.default_rng(seed)
        # Fixed low-resolution random field: same morphology across smoke levels.
        field = gaussian_filter(rng.normal(size=(64, 64)), 3.)
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
        field = (field-field.mean()) / max(float(field.std()), 1e-8)
        threshold = np.quantile(field, 1-p["coverage"])
        alpha = 1/(1+np.exp(np.clip(-(field-threshold)/p["softness"], -40, 40)))
        t = (1-alpha*(1-p["transmission"])).astype(np.float32)
        coverage = float(np.mean(alpha >= .5))
        airlight = np.asarray(p["color_rgb"], float)
    elif condition != "clear":
        raise ValueError(condition)
    if not (np.isfinite(t).all() and np.all((t >= 0) & (t <= 1))
            and airlight.shape == (3,) and np.all((airlight >= 0) & (airlight <= 1))):
        raise ValueError("Invalid transmission/airlight")
    image = rgb.copy() if condition == "clear" else np.rint(np.clip(
        rgb.astype(float)*t[..., None] + 255*airlight*(1-t[..., None]), 0, 255)).astype(np.uint8)
    return image, t, dict(transmission_mean=float(t.mean()), transmission_min=float(t.min()),
                         smoke_coverage=coverage)


@dataclass(frozen=True)
class Prior:
    """Only simulated measurements, in right-handed East/North/Up coordinates."""
    xyz: np.ndarray
    camera_to_enu: np.ndarray
    sigma_xy_m: float
    yaw_sigma_deg: float
    tilt_sigma_deg: float
    z_sigma_m: float


def make_prior(xyz, rotation, sigma_xy, profile, seed):
    rng = np.random.default_rng(seed)
    error = rng.normal(size=3)*[sigma_xy, sigma_xy, profile["z_m"]]
    # World-frame perturbations, independent of the Euler singularity at nadir.
    angles = rng.normal(size=3)*[profile["tilt_deg"], profile["tilt_deg"], profile["yaw_deg"]]
    tilt = Rotation.from_rotvec(np.radians([angles[0], angles[1], 0.])).as_matrix()
    yaw = Rotation.from_rotvec(np.radians([0., 0., angles[2]])).as_matrix()
    prior = Prior(np.asarray(xyz)+error, yaw @ tilt @ rotation,
                  float(sigma_xy), profile["yaw_deg"], profile["tilt_deg"], profile["z_m"])
    return prior, dict(prior_error_enu_m=error.tolist(), perturbation_xyz_deg=angles.tolist(),
                      prior_xyz_enu=prior.xyz.tolist(), prior_rotation=prior.camera_to_enu.tolist())


def nadir_angle(rotation):
    axis = np.asarray(rotation)[:, 2]
    return float(np.degrees(np.arccos(np.clip(axis @ [0, 0, -1], -1, 1))))


def footprint(xyz, rotation, K, dist, image_shape, plane_z, *, center_crop=False):
    """Intersect boundary rays with an explicitly approximate scene-median plane."""
    h, w = image_shape[:2]
    x0, y0, x1, y1 = -.5, -.5, w-.5, h-.5
    if center_crop:
        side = min(h, w)
        left, top = w//2-side//2, h//2-side//2
        x0, y0, x1, y1 = left-.5, top-.5, left+2*(side//2)-.5, top+2*(side//2)-.5
    pixels = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1],[(x0+x1)/2,(y0+y1)/2]], float)
    normalized = cv2.undistortPoints(pixels[:,None,:], np.asarray(K,float), np.asarray(dist,float)).reshape(-1,2)
    rays = np.column_stack([normalized, np.ones(5)]) @ np.asarray(rotation).T
    if float(xyz[2]) <= plane_z or np.any(rays[:,2] >= -1e-5):
        raise ValueError("unsupported_prior_geometry")
    points = np.asarray(xyz) + rays*((plane_z-xyz[2])/rays[:,2])[:,None]
    polygon = Polygon(points[:4,:2])
    if not polygon.is_valid or polygon.area < 1e-6:
        raise ValueError("unsupported_prior_geometry")
    return polygon, points[4,:2]


def retrieval_truth(candidates, visible, center, thresholds=(.25,.5,.75)):
    fractions = [float(visible.intersection(Polygon(c["polygon_enu"])).area/visible.area)
                 for c in candidates]
    return dict(retrieval_overlap=fractions,
        retrieval_center_hit=any(Polygon(c["polygon_enu"]).covers(Point(center)) for c in candidates),
        best_candidate_error_m=(float(np.linalg.norm(np.mean(candidates[0]["polygon_enu"], axis=0)-center))
                                if candidates else None),
        **{f"recall_overlap_{int(t*100)}": any(q >= t for q in fractions) for t in thresholds})


def score_result(prediction, gt_xyz, threshold=10.):
    """Evaluation only. Altering gt cannot affect inference."""
    row = dict(prediction)
    pred = row.get("pred_xyz_enu")
    error = float(np.linalg.norm(np.asarray(pred)[:2]-np.asarray(gt_xyz)[:2])) if pred is not None else None
    row.update(gt_xyz_enu=np.asarray(gt_xyz).tolist(), error_xy_m=error)
    if row.get("status") != "technical_error":
        row["status"] = ("correct" if error is not None and error <= threshold else "false_accept") if row["accepted"] else "rejected"
    return row
