"""Paired, resumable experiment records; evaluation lives outside inference."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .backend import localize
from .core import (CONDITIONS,canonical,digest,file_hash,array_hash,footprint,
                   make_prior,retrieval_truth,score_result,stable_seed,validate_config)
from .data import read_sample,load_reference,reference_identity


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(canonical(value),encoding="utf-8")
    os.replace(temporary,path)


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def log_resources(output, stage, backend, **context):
    """Persist breadcrumbs even when the OS kills Python without a traceback."""
    row = dict(time_unix=time.time(),stage=stage,**context)
    for filename, fields in [('/proc/self/status', {'VmRSS':'rss_kib','VmHWM':'peak_rss_kib'}),
                             ('/proc/meminfo', {'MemAvailable':'system_available_kib'})]:
        try:
            for line in Path(filename).read_text().splitlines():
                key, _, value = line.partition(':')
                if key in fields:
                    row[fields[key]] = int(value.strip().split()[0])
        except OSError:
            pass
    try:
        torch = backend.torch
        row.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                   gpu_reserved_bytes=torch.cuda.memory_reserved())
    except (AttributeError, RuntimeError):
        pass
    with (Path(output)/'resource_log.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(canonical(row)+'\n')


def environment():
    versions = {}
    for name in ["numpy","scipy","opencv-python-headless","opencv-python","torch","torchvision",
                 "timm","pandas","shapely","Pillow","kornia"]:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    info = dict(python=platform.python_version(),platform=platform.platform(),versions=versions)
    try:
        import torch
        info.update(cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    cudnn=torch.backends.cudnn.version())
    except ImportError: pass
    return info


def export_records(output):
    rows = []
    manifest_path = Path(output)/"experiment_config.json"
    experiment_id = read_json(manifest_path)["experiment_id"] if manifest_path.exists() else None
    for path in sorted((Path(output)/"records").glob("*.json")):
        envelope = read_json(path)
        if digest(envelope["result"]) != envelope["sha256"]: raise ValueError(f"Corrupt record {path}")
        row = dict(envelope["result"])
        if row["attempt_id"] != path.stem or (experiment_id is not None and row["experiment_id"] != experiment_id):
            raise ValueError(f"Foreign attempt record {path}")
        for key,value in row.items():
            if isinstance(value,(list,dict)): row[key] = canonical(value)
        rows.append(row)
    frame = pd.DataFrame(rows)
    temp = Path(output)/"per_sample_results.csv.tmp"
    frame.to_csv(temp,index=False)
    os.replace(temp,Path(output)/"per_sample_results.csv")
    return frame


def run_experiment(backend, selection, config, output, *, retry_technical=False):
    from .core import degrade
    validate_config(config)
    matching_method = config.get('matching_method', 'SP_LG')
    if backend.identity.get('matching_method', matching_method) != matching_method:
        raise ValueError('Backend matcher/config mismatch')
    output = Path(output).resolve()
    if not selection: raise ValueError("Empty selection")
    for row in selection:
        scene = Path(row["path"]).resolve().parent
        if output.is_relative_to(scene) or scene.is_relative_to(output):
            raise ValueError("Results and source scene must be separate")
    orchestration = {"sample_count","scenes","sigma_xy_m","trials","active_profiles","retrieval_modes"}
    if digest({k:v for k,v in config.items() if k not in orchestration}) != digest(
            {k:v for k,v in backend.config.items() if k not in orchestration}):
        raise ValueError("Backend/config mismatch")
    output.mkdir(parents=True,exist_ok=True)
    if hasattr(backend, 'release_reference'):
        backend.release_reference()
    log_resources(output,'prepare_begin',backend)
    # Hash sources before inference, not on the basis of its success.
    prepared, reference_ids = [], {}
    reference = None
    for row in sorted(selection,key=lambda x:(x["scene_id"],x["sample_id"])):
        sample = read_sample(row["path"],config["pose_convention"],config["max_image_side"])
        if row["scene_id"] not in reference_ids:
            reference = None
            log_resources(output,'manifest_reference_hash',backend,scene_id=row['scene_id'])
            reference_ids[row["scene_id"]] = reference_identity(row["reference_json"],config["reference_mode"])
        prepared.append(dict(scene_id=row["scene_id"],sample_id=row["sample_id"],
            source=sample["source_identity"],K=sample["K"].tolist(),dist=sample["dist"].tolist(),
            xyz_enu=sample["xyz_enu"].tolist(),rotation=sample["rotation"].tolist(),
            nadir_deg=sample["nadir_deg"]))
    reference = None
    manifest = dict(schema_version=1,config=config,backend=backend.identity,environment=environment(),
        selection=prepared,references=reference_ids,source_sha256={p.name:file_hash(p)
            for p in Path(__file__).parent.glob("*.py") if not p.name.startswith("test_")},
        footprint_model="query_center_crop_on_local_DSM_plane_or_scene_median",
        conditions=list(CONDITIONS),ground_truth_used_in_inference=False,
        synthetic_visibility=True,synthetic_prior=True)
    identity = digest(manifest)
    manifest["experiment_id"] = identity
    manifest_path = output/"experiment_config.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError("Experiment identity changed: use a new output directory")
    else: write_json(manifest_path,manifest)
    total = (len(selection)*len(config["sigma_xy_m"])*len(config["active_profiles"])*
             config["trials"]*len(config["retrieval_modes"])*len(CONDITIONS))
    write_json(output/"progress.json",dict(expected_trials=total,experiment_id=identity,complete=False))
    completed = 0
    current_scene,reference = None,None
    try:
        with tqdm(total=total,desc="Повна локалізація") as bar:
            for selected in sorted(selection,key=lambda r:(r["scene_id"],r["sample_id"])):
                sample = read_sample(selected["path"],config["pose_convention"],config["max_image_side"])
                if selected["scene_id"] != current_scene:
                    if hasattr(backend, 'release_reference'):
                        backend.release_reference()
                    reference = None
                    log_resources(output,'scene_reference_load',backend,scene_id=selected['scene_id'])
                    reference = load_reference(selected["reference_json"],config["reference_mode"])
                    current_scene = selected["scene_id"]
                    reference_before = array_hash(reference.bgr)
                try:
                    gt_visible,gt_center = footprint(sample["xyz_enu"],sample["rotation"],sample["K"],sample["dist"],
                        sample["rgb"].shape,reference.plane_at(sample['xyz_enu'][:2]),center_crop=True)
                    evaluation_error = None
                except ValueError as exc:
                    gt_visible,gt_center,evaluation_error = None,None,str(exc)
                for profile_name in config["active_profiles"]:
                    profile = config["profiles"][profile_name]
                    for sigma in config["sigma_xy_m"]:
                        for trial in range(config["trials"]):
                            pair_id = digest([sample["scene_id"],sample["sample_id"],profile_name,sigma,trial])[:24]
                            prior_seed = stable_seed(config["seed"],pair_id,"prior")
                            prior,prior_meta = make_prior(sample["xyz_enu"],sample["rotation"],sigma,profile,prior_seed)
                            for mode in config["retrieval_modes"]:
                                for condition in CONDITIONS:
                                    key = digest([identity,pair_id,mode,condition])
                                    path = output/"records"/(key+".json")
                                    if path.exists():
                                        saved = read_json(path)
                                        if digest(saved["result"]) != saved["sha256"]:
                                            raise ValueError(f"Corrupt completed attempt: {key}")
                                        old = saved["result"]
                                        if old["experiment_id"] != identity or old["attempt_id"] != key:
                                            raise ValueError("Attempt identity mismatch")
                                        if not (retry_technical and old["status"] == "technical_error"):
                                            completed += 1
                                            bar.update(1)
                                            continue
                                        # Preserve technical failures before retrying.
                                        write_json(output/"retry_history"/f"{key}-{time.time_ns()}.json",saved)
                                    smoke_seed = stable_seed(config["seed"],sample["scene_id"],sample["sample_id"],"smoke")
                                    inference_seed = stable_seed(config["seed"],pair_id,"inference")
                                    image,transmission,visibility_meta = degrade(sample["rgb"],condition,config,smoke_seed)
                                    row = dict(experiment_id=identity,attempt_id=key,pair_id=pair_id,
                                        scene_id=sample["scene_id"],sample_id=sample["sample_id"],condition=condition,
                                        severity=condition.split("_",1)[-1] if condition != "clear" else "none",
                                        profile=profile_name,sigma_xy_m=sigma,trial_id=trial,retrieval_mode=mode,
                                        reference_mode=config["reference_mode"],retrieval_method="CAMP",matching_method=matching_method,
                                        map_alignment=config.get('map_alignment', 'none'),
                                        seed=config["seed"],prior_seed=prior_seed,smoke_seed=smoke_seed,inference_seed=inference_seed,
                                        nadir_deg=sample["nadir_deg"],camera_z_enu_m=float(sample["xyz_enu"][2]),
                                        height_above_scene_plane_m=float(sample["xyz_enu"][2]-reference.plane_z),
                                        height_above_local_plane_m=float(sample['xyz_enu'][2]-reference.plane_at(sample['xyz_enu'][:2])),
                                        profile_sigmas=profile,evaluation_geometry_error=evaluation_error,
                                        inliers=0,raw_matches=0,matches=0,
                                        **prior_meta,**visibility_meta)
                                    try:
                                        log_resources(output,'inference_begin',backend,scene_id=sample['scene_id'],
                                                      sample_id=sample['sample_id'],condition=condition)
                                        prediction = localize(backend,image,sample["K"],sample["dist"],reference,prior,mode,inference_seed)
                                    except Exception as exc:
                                        if isinstance(exc,ValueError) and str(exc) == "unsupported_prior_geometry":
                                            prediction = dict(accepted=False,status="inference_complete",valid_pnp=False,
                                                rejection_reason=str(exc),pred_xyz_enu=None,candidates=[],candidate_results=[])
                                        else:
                                            prediction = dict(accepted=False,status="technical_error",pred_xyz_enu=None,candidates=[],
                                                rejection_reason="technical_error",exception_type=type(exc).__name__,
                                                exception=str(exc),traceback=traceback.format_exc())
                                    scored = score_result(prediction,sample["xyz_enu"],config["success_threshold_m"])
                                    if gt_visible is not None:
                                        scored.update(retrieval_truth(prediction["candidates"],gt_visible,gt_center,
                                            config["recall_overlap_thresholds"]))
                                    row.update(scored)
                                    write_json(path,dict(sha256=digest(row),result=row))
                                    log_resources(output,'record_saved',backend,scene_id=sample['scene_id'],
                                                  sample_id=sample['sample_id'],condition=condition,status=row['status'])
                                    completed += 1
                                    bar.update(1)
                                    if completed % 12 == 0: export_records(output)
                if array_hash(reference.bgr) != reference_before:
                    raise AssertionError("Reference map mutated during inference")
    finally:
        if hasattr(backend, 'release_reference'):
            backend.release_reference()
        frame = export_records(output)
        write_json(output/"progress.json",dict(expected_trials=total,completed_trials=completed,
                    experiment_id=identity,complete=completed == total,
                    technical_errors=int((frame.get("status",pd.Series(dtype=str)) == "technical_error").sum())))
    return frame
