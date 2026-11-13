"""Strict NPZ adapter. GT is consumed by selection/simulation/evaluation only."""
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .core import file_hash, nadir_angle, stable_seed


def relative_file(root, name):
    p = (Path(root)/str(name)).resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError("Path outside scene")
    if not p.is_file(): raise FileNotFoundError(p)
    return p


@dataclass
class Reference:
    scene_id: str
    bgr: np.ndarray
    dsm: np.ndarray
    resolution: np.ndarray
    origin: np.ndarray
    dsm_resolution: np.ndarray
    dsm_origin: np.ndarray
    plane_z: float
    identity: dict

    def pixels_to_enu(self, pixels):
        xy = np.asarray(pixels)*self.resolution+self.origin
        return xy * [1., -1.]

    def heights(self, enu_xy):
        local = np.asarray(enu_xy)*[1.,-1.]
        cr = np.rint((local-self.dsm_origin)/self.dsm_resolution).astype(int)
        valid = ((cr[:,0] >= 0) & (cr[:,0] < self.dsm.shape[1]) &
                 (cr[:,1] >= 0) & (cr[:,1] < self.dsm.shape[0]))
        z = np.full(len(cr), np.nan)
        z[valid] = self.dsm[cr[valid,1], cr[valid,0]]
        return z, valid & np.isfinite(z)

    def plane_at(self, enu_xy):
        heights, valid = self.heights(np.asarray(enu_xy).reshape(1, 2))
        return float(heights[0]) if valid[0] else self.plane_z


def reference_identity(json_path, mode="satellite"):
    """Hash source files without loading full map/DSM arrays for the manifest."""
    path = Path(json_path)
    meta = json.loads(path.read_text(encoding="utf-8-sig"))["modes"][mode]
    files = [path, relative_file(path.parent, meta['map_path']), relative_file(path.parent, meta['dsm_path'])]
    return {p.name:file_hash(p) for p in files}


def load_reference(json_path, mode="satellite"):
    path = Path(json_path)
    meta = json.loads(path.read_text(encoding="utf-8-sig"))["modes"][mode]
    image_path = relative_file(path.parent, meta["map_path"])
    dsm_path = relative_file(path.parent, meta["dsm_path"])
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError(f"Unreadable map: {image_path}")
    dsm = np.load(dsm_path, mmap_mode="r", allow_pickle=False)
    if dsm.ndim == 3 and dsm.shape[2] == 1: dsm = dsm[:,:,0]
    if dsm.ndim != 2: raise ValueError("DSM must be 2D")
    resolution = np.asarray(meta["map_resolution"], float)
    origin = np.asarray(meta.get("map_origin_local", [0,0]), float)
    dsm_res = np.asarray(meta["dsm_resolution"], float)
    dsm_origin = origin+np.asarray(meta.get("dsm_origin_local", [0,0]), float)
    if resolution.shape != (2,) or dsm_res.shape != (2,) or np.any(resolution <= 0) or np.any(dsm_res <= 0):
        raise ValueError("Invalid reference pixel size")
    # Strip memmap/subclass semantics before boolean indexing. NumPy versions
    # differ in whether indexing a read-only memmap retains read-only flags.
    # Indexing a plain ndarray creates a private, writable selection.
    valid = np.asarray(dsm)[np.isfinite(dsm)]
    if not valid.size: raise ValueError("No valid DSM heights")
    # valid is already a private copy. In-place selection avoids another full DSM copy.
    plane_z = float(np.median(valid, overwrite_input=True))
    del valid
    bgr.flags.writeable = False
    identity = {p.name: file_hash(p) for p in (path, image_path, dsm_path)}
    return Reference(path.parent.name, bgr, dsm, resolution, origin, dsm_res, dsm_origin,
                     plane_z, identity)


def read_sample(path, convention="ned_attitude", max_side=1600):
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        xyz_dataset = np.asarray(f["xyz"], float).reshape(3)
        pose = np.asarray(f["pose_c2w"], float)
        if pose.shape != (4,4) or not np.allclose(pose[:3,3], xyz_dataset, atol=.05):
            raise ValueError("Pose/XYZ translation disagreement")
        raw_rotation = pose[:3,:3]
        if convention == "ned_attitude":
            # Verified public-release serialization: ZYX attitude in NED;
            # identity attitude is nadir. Validate against Euler, never use pitch as nadir.
            euler = np.asarray(f["euler_deg"], float).reshape(3)
            expected = Rotation.from_euler("ZYX", euler[::-1], degrees=True).as_matrix()
            if not np.allclose(raw_rotation, expected, atol=2e-5):
                raise ValueError("Unverified pose convention: matrix disagrees with ZYX/NED attitude")
            ned_to_enu = np.array([[0,1,0],[1,0,0],[0,0,-1.]])
            camera_to_frd = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
            rotation = ned_to_enu @ raw_rotation @ camera_to_frd
        elif convention == "opencv_enu":
            rotation = raw_rotation
        else:
            raise ValueError("Declare ned_attitude or opencv_enu pose convention")
        if not (np.allclose(rotation.T@rotation, np.eye(3), atol=1e-4) and np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
            raise ValueError("Invalid camera rotation")
        rgb = np.asarray(f["image"]).copy()
        source_files = [path]
        if rgb.size == 0:
            image_path = relative_file(path.parent, str(f["image_relpath"].item()))
            bgr = cv2.imread(str(image_path))
            if bgr is None: raise ValueError("Unreadable query image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            source_files.append(image_path)
        K = np.asarray(f["K"], float).copy()
        dist = np.asarray(f["dist"], float).reshape(-1)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected uint8 HWC RGB query")
    if K.shape != (3,3) or K[0,0] <= 0 or K[1,1] <= 0:
        raise ValueError("Invalid intrinsics")
    if not all(np.isfinite(v).all() for v in (xyz_dataset, rotation, K, dist)):
        raise ValueError("Nonfinite metadata")
    h,w = rgb.shape[:2]
    scale = min(1., max_side/max(h,w))
    if scale < 1:
        nh,nw = max(2,round(h*scale)), max(2,round(w*scale))
        rgb = cv2.resize(rgb, (nw,nh), interpolation=cv2.INTER_AREA)
        K[0] *= nw/w
        K[1] *= nh/h
    return dict(scene_id=path.parent.name, sample_id=path.stem, path=str(path),
                rgb=rgb, K=K, dist=dist, xyz_enu=xyz_dataset*[1,-1,1], rotation=rotation,
                nadir_deg=nadir_angle(rotation), source_identity={p.name:file_hash(p) for p in source_files})


def select_diagnostic_samples(selection, identities):
    """Keep the exact comparison frames; never replace missing frames silently."""
    requested = [tuple(identity) for identity in identities]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError('Diagnostic frame identities must be nonempty and unique')
    by_id = {(row['scene_id'], row['sample_id']): row for row in selection}
    if len(by_id) != len(selection):
        raise ValueError('Duplicate frames in selection')
    missing = [identity for identity in requested if identity not in by_id]
    if missing:
        raise ValueError(f'Fixed diagnostic frames missing from selection: {missing}. Check scenes and selection audit.')
    return [by_id[identity] for identity in requested]


def validate_balanced_selection(selection, scenes, sample_count):
    """Require the advertised number of unique frames and equal scene coverage."""
    if not scenes or len(scenes) != len(set(scenes)) or sample_count % len(scenes):
        raise ValueError('Balanced selection requires unique scenes and an evenly divisible frame count')
    identities = [(row['scene_id'], row['sample_id']) for row in selection]
    if len(identities) != sample_count or len(set(identities)) != sample_count:
        raise ValueError(f'Expected {sample_count} unique frames; obtained {len(identities)}')
    per_scene = sample_count // len(scenes)
    counts = pd.Series([scene for scene, _ in identities]).value_counts()
    if set(counts.index) != set(scenes) or not counts.eq(per_scene).all():
        raise ValueError(f'Expected {per_scene} frames per scene; obtained {counts.to_dict()}')
    return pd.DataFrame({'scene_id': list(scenes), 'selected_frames': [per_scene]*len(scenes)})


def select_samples(root, config, output):
    """Deterministic scene-balanced selection, independent of localization outcomes."""
    rows, eligible = [], {}
    for scene in config["scenes"]:
        matches = list(Path(root).rglob(f"{scene}/L??_reference.json"))
        if len(matches) != 1: raise ValueError(f"Expected exactly one reference JSON for {scene}")
        ref = matches[0]
        files = sorted(ref.parent.glob("L??_????.npz"))
        if not files: raise FileNotFoundError(f"No samples in {ref.parent}")
        eligible[scene] = []
        for path in files:
            try:
                s = read_sample(path, config["pose_convention"], config["max_image_side"])
                row = dict(scene_id=scene, sample_id=path.stem, path=str(path),
                           reference_json=str(ref), nadir_deg=s["nadir_deg"],
                           eligible=s["nadir_deg"] <= config["nadir_max_deg"], error="")
                if row["eligible"]: eligible[scene].append(row)
            except Exception as exc:
                row = dict(scene_id=scene,sample_id=path.stem,path=str(path),eligible=False,error=str(exc))
            rows.append(row)
    for scene, candidates in eligible.items():
        candidates.sort(key=lambda r:(r["nadir_deg"] > config["preferred_nadir_deg"],
                                     stable_seed(config["seed"],scene,r["sample_id"])))
    selected = []
    while len(selected) < config["sample_count"]:
        before = len(selected)
        for candidates in eligible.values():
            if candidates and len(selected) < config["sample_count"]: selected.append(candidates.pop(0))
        if len(selected) == before: break
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/"selection_audit.csv", index=False)
    pd.DataFrame(selected).to_csv(output/"selected_samples.csv", index=False)
    if not selected:
        raise ValueError("No verified nadir samples; inspect selection_audit.csv and pose convention")
    return selected, pd.DataFrame(rows)
