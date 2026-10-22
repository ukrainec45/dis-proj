"""Array-only adapter for the public JSON/NPZ AnyVisLoc release.

Dataset XY is East/South; public outputs use East/North/Up.
Pose frame is declared and checked against xyz, never inferred from Euler angles.
"""
from dataclasses import dataclass
import json
from pathlib import Path

from affine import Affine
import cv2
import numpy as np


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def relative_file(parent, value):
    path = (Path(parent) / value).resolve()
    if not path.is_relative_to(Path(parent).resolve()):
        raise ValueError("Dataset paths must stay inside the scene directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def dataset_to_enu(xyz):
    return np.asarray(xyz, dtype=float) * [1, -1, 1]


def map_transform(origin, resolution):
    origin, resolution = np.asarray(origin, float), np.asarray(resolution, float)
    if origin.shape != (2,) or resolution.shape != (2,):
        raise ValueError("Map origin and resolution must be XY pairs")
    if not np.isfinite([origin, resolution]).all() or np.any(resolution <= 0):
        raise ValueError("Invalid map origin/resolution")
    return Affine(resolution[0], 0, origin[0], 0, -resolution[1], -origin[1])


@dataclass
class Reference:
    image: np.ndarray
    dsm: np.ndarray
    transform: Affine
    dsm_transform: Affine
    files: tuple[Path, ...]

    def elevation(self, xy):
        col, row = np.rint(~self.dsm_transform * tuple(xy)).astype(int)
        if not (0 <= row < self.dsm.shape[0] and 0 <= col < self.dsm.shape[1]):
            raise ValueError("dsm_outside")
        height = float(self.dsm[row, col])
        if not np.isfinite(height):
            raise ValueError("dsm_nodata")
        return height


def load_reference(path, mode):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))["modes"][mode]
    image_path = relative_file(path.parent, raw["map_path"])
    dsm_path = relative_file(path.parent, raw["dsm_path"])
    origin = np.asarray(raw.get("map_origin_local", [0, 0]), float)
    dsm_origin = origin + np.asarray(raw.get("dsm_origin_local", [0, 0]), float)
    dsm = np.load(dsm_path, mmap_mode="r", allow_pickle=False)
    if dsm.ndim != 2:
        raise ValueError("Expected a two-dimensional DSM")
    return Reference(read_rgb(image_path), dsm,
                     map_transform(origin, raw["map_resolution"]),
                     map_transform(dsm_origin, raw["dsm_resolution"]),
                     (path, image_path, dsm_path))


def anyvisloc_attitude_to_enu(rotation, euler_deg):
    """Convert the release's ZYX/NED attitude to OpenCV-camera -> ENU.

    Translation in this release is separately serialized as dataset-local xyz.
    Camera at zero attitude is nadir, with image top pointing North.
    This mounting interpretation follows the upstream pitch/yaw footprint
    convention; real-image smoke results must still verify projection alignment.
    """
    roll, pitch, yaw = np.radians(np.asarray(euler_deg, float).reshape(3))
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    expected = np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr],
    ])
    if not np.allclose(rotation, expected, atol=1e-5, rtol=0):
        raise ValueError("pose rotation disagrees with released ZYX Euler attitude")
    ned_to_enu = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    camera_to_frd = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    return ned_to_enu @ rotation @ camera_to_frd


def load_sample(path, *, pose_world_frame):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        image = data["image"]
        image_path = None
        if image.size == 0:
            image_path = relative_file(path.parent, str(data["image_relpath"].item()))
            image = read_rgb(image_path)
        K = np.asarray(data["K"], float)
        dist = np.asarray(data["dist"], float).reshape(-1)
        pose = np.asarray(data["pose_c2w"], float)
        xyz_raw = np.asarray(data["xyz"], float)
        xyz = dataset_to_enu(xyz_raw)
        euler = (np.asarray(data["euler_deg"], float)
                 if pose_world_frame == "anyvisloc_ned_attitude" else None)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected uint8 HWC RGB image")
    if K.shape != (3, 3) or pose.shape != (4, 4) or xyz.shape != (3,):
        raise ValueError("Invalid camera/pose dimensions")
    if not all(np.isfinite(a).all() for a in (K, dist, pose, xyz)):
        raise ValueError("Non-finite camera/pose")
    if pose_world_frame == "anyvisloc_ned_attitude":
        if not np.allclose(pose[:3, 3], xyz_raw, atol=.05, rtol=0):
            raise ValueError("pose_c2w translation disagrees with dataset xyz")
        pose = pose.copy()
        pose[:3, :3] = anyvisloc_attitude_to_enu(pose[:3, :3], euler)
        pose[:3, 3] = xyz
    elif pose_world_frame == "dataset":
        pose = np.diag([1., -1., 1., 1.]) @ pose
    elif pose_world_frame != "enu":
        raise ValueError("pose_world_frame must be enu, dataset or anyvisloc_ned_attitude")
    if not np.allclose(pose[:3, 3], xyz, atol=.05, rtol=0):
        raise ValueError("pose_c2w translation disagrees with declared frame and xyz")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("Invalid camera attitude")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("Invalid camera intrinsics")
    return dict(image=image, K=K, dist=dist, rotation=rotation, xyz=xyz,
                files=(path,) if image_path is None else (path, image_path))

