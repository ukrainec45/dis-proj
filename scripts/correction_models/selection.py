"""Metadata-only inventory, geographic audit and reproducible farthest-point selection."""
from pathlib import Path
from io import BytesIO
import zipfile

import numpy as np
from shapely.geometry import shape

from scripts.visibility_localization.core import stable_seed, file_hash
from scripts.visibility_localization.data import read_sample, reference_identity
from .artifacts import digest, read, save_verified, load_verified, write_once
from .config import REVIEWED, SPLIT_SCENES, FRAMES_PER_SCENE, SEED, main_config
from .contracts import PROTOCOL_ID

SCENE_POLICY = 'official_scene_groups_v1'
COORDINATE_SOURCE = 'https://github.com/UAV-AVL/Benchmark/blob/main/docs/GROUND_TRUTH.md'
_REFERENCE_IDENTITY = reference_identity


def scene_policy(scene_ids, previous=None):
    """Document scene grouping without claiming verified geographic separation.

    Retain known shared groups and polygon evidence from older files. A blank
    legacy geographic template is input, never rewritten as a completed audit.
    """
    entries = (previous or {}).get('scenes', {})
    return dict(mode=SCENE_POLICY, reviewed=False, geographic_overlap_verified=False,
        method='official AnyVisLoc scene IDs; retain known shared groups',
        evidence=COORDINATE_SOURCE, evaluation_scope='held_out_dataset_scenes',
        limitation='Global coordinates are sanitized; scene separation does not prove geographic non-overlap.',
        scenes={s: dict(spatial_group=entries.get(s, {}).get('spatial_group') or s,
            evidence=entries.get(s, {}).get('evidence') or 'Official dataset scene identifier only',
            footprint_wgs84=entries.get(s, {}).get('footprint_wgs84')) for s in sorted(scene_ids)})


def selection_readiness(data):
    """Cheap preflight over the existing inventory; never reopen NPZs."""
    counts = {}
    positions = {}
    for row in data['rows']:
        scene = row['scene_id']
        counts.setdefault(scene, 0)
        positions.setdefault(scene, set())
        if row['eligible']:
            counts[scene] += 1
            positions[scene].add(tuple(row['xyz_enu'][:2]))
    eligible = sorted(s for s in counts if counts[s] >= FRAMES_PER_SCENE
                      and len(positions[s]) >= FRAMES_PER_SCENE)
    return dict(eligible_scenes=eligible, eligible_scene_count=len(eligible),
        required_scenes=16, minimum_additional_scenes=max(0, 16-len(eligible)),
        insufficient_scenes={s: dict(eligible_frames=counts[s], distinct_positions=len(positions[s]),
                                    required_frames=FRAMES_PER_SCENE)
                             for s in sorted(counts) if s not in eligible})


def _metadata_array(archive, name, shape, trace=None):
    """Bound both decompressed member size and NPY shape before NumPy allocates."""
    member = archive.getinfo(name+'.npy')
    if member.file_size > 64*1024:
        raise ValueError(f'Oversized metadata member: {name}')
    if trace is not None:
        trace(f'read {name}.npy: {member.file_size} uncompressed bytes')
    stream = BytesIO(archive.read(member))
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        actual, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        actual, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f'Unsupported metadata NPY version: {version}')
    size = 1
    for dim in actual:
        size *= dim
    if (dtype.kind not in 'fiu' or size != int(np.prod(shape))
            or (shape == (4, 4) and actual != shape)):
        raise ValueError(f'Invalid metadata shape/dtype: {name}')
    stream.seek(0)
    return np.asarray(np.load(stream, allow_pickle=False), float).reshape(shape)


def read_metadata(path, convention='ned_attitude', max_side=1600, *, trace=None):
    """Read only tiny NPZ metadata members, not the image or full-file checksum."""
    from scipy.spatial.transform import Rotation
    from scripts.visibility_localization.core import nadir_angle
    path = Path(path)
    if trace is not None:
        trace('open NPZ directory')
    with zipfile.ZipFile(path) as data:
        xyz = _metadata_array(data, 'xyz', (3,), trace)
        pose = _metadata_array(data, 'pose_c2w', (4, 4), trace)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[:3, 3], xyz, atol=.05):
            raise ValueError('Invalid pose metadata')
        if convention == 'ned_attitude':
            euler = _metadata_array(data, 'euler_deg', (3,), trace)
            expected = Rotation.from_euler('ZYX', euler[::-1], degrees=True).as_matrix()
            if not np.allclose(pose[:3, :3], expected, atol=2e-5):
                raise ValueError('Unverified NED pose convention')
            rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.]])@expected@np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
        elif convention == 'opencv_enu':
            rotation = pose[:3, :3]
        else:
            raise ValueError('Unknown pose convention')
        if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1):
            raise ValueError('Invalid rotation')
    return dict(scene_id=path.parent.name, sample_id=path.stem, xyz_enu=xyz*[1., -1., 1.],
                nadir_deg=nadir_angle(rotation))


def inventory(root, output, *, reader=read_metadata, progress=print, min_available_mb=512):
    """Metadata only, with scene checkpoints and a preemptive available-RAM guard.

    Resume trusts source path/size/mtime, as extraction receipts do. Selected
    frames are subsequently validated and hashed during prepare_selection.
    """
    import psutil
    root, output = Path(root).resolve(), Path(output)
    config = main_config()
    process = psutil.Process()
    def status(message):
        if progress is not None:
            progress(f'{message}; RSS={process.memory_info().rss/1024**2:.0f} MiB; '
                     f'available={psutil.virtual_memory().available/1024**2:.0f} MiB')
    checkpoints = output.parent/(output.stem+'.scenes')
    rows, errors = [], []
    for ref in sorted(root.rglob('L??_reference.json')):
        scene = ref.parent.name
        paths = sorted(ref.parent.glob('L??_????.npz'))
        sources = []
        for path in [ref, *paths]:
            stat = path.stat()
            sources.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        identity = dict(root=str(root), sources=sources, pose_convention=config['pose_convention'],
                        nadir_max_deg=config['nadir_max_deg'], metadata_reader='bounded-npy-v1')
        checkpoint = checkpoints/(digest(identity)+'.json')
        if checkpoint.exists():
            cached = load_verified(checkpoint)
            if cached['identity'] != identity:
                raise ValueError('Inventory checkpoint identity mismatch')
            rows.extend(cached['rows'])
            errors.extend(cached['errors'])
            status(f'{scene}: reused metadata checkpoint ({len(paths)} frames)')
            continue
        scene_rows, scene_errors = [], []
        for index, path in enumerate(paths, 1):
            if index == 1 or index % 25 == 0:
                status(f'{scene} [{index}/{len(paths)}] {path.name}')
            if psutil.virtual_memory().available < min_available_mb*1024**2:
                raise MemoryError(f'Inventory stopped before {path}: less than {min_available_mb} MiB '
                                  'RAM available. Restart the session and resume with the same paths; '
                                  'completed scene checkpoints are retained.')
            try:
                s = reader(path, config['pose_convention'], config['max_image_side'])
                if s['scene_id'] != scene:
                    raise ValueError('scene identity mismatch')
                scene_rows.append(dict(scene_id=scene, sample_id=s['sample_id'], path=str(path),
                    reference_json=str(ref.resolve()), xyz_enu=s['xyz_enu'].tolist(),
                    nadir_deg=float(s['nadir_deg']),
                    eligible=bool(s['nadir_deg'] <= config['nadir_max_deg'])))
            except Exception as exc:
                if isinstance(exc, MemoryError):
                    raise
                scene_errors.append(dict(path=str(path), error=f'{type(exc).__name__}: {exc}'))
        save_verified(checkpoint, dict(identity=identity, rows=scene_rows, errors=scene_errors))
        rows.extend(scene_rows)
        errors.extend(scene_errors)
        status(f'{scene}: metadata saved ({len(scene_rows)} frames, {len(scene_errors)} errors)')
    if not rows:
        raise ValueError(f'No readable AnyVisLoc frames under {root}')
    save_verified(output, dict(root=str(root), rows=rows, errors=errors))
    return dict(frames=len(rows), eligible=sum(r['eligible'] for r in rows), errors=len(errors),
                scenes=sorted({r['scene_id'] for r in rows}))


def audit_template(inventory_path, output):
    """Create scene-group policy with explicit unverified geography, not a false audit."""
    data = load_verified(inventory_path)
    scenes = sorted({r['scene_id'] for r in data['rows']})
    write_once(output, scene_policy(scenes))


def audit_groups(audit, scene_ids):
    scene_mode = audit.get('mode') == SCENE_POLICY
    if scene_mode and (audit.get('geographic_overlap_verified') is not False
                       or audit.get('evaluation_scope') != 'held_out_dataset_scenes'):
        raise ValueError('Scene-ID grouping must explicitly mark geographic overlap as unverified')
    if not scene_mode and (audit.get('reviewed') is not True or not audit.get('method') or not audit.get('evidence')):
        raise ValueError('Geographic overlap audit needs reviewed=true, method and evidence; local XYZ alone is insufficient')
    entries = audit.get('scenes', {})
    if set(scene_ids)-set(entries):
        raise ValueError('Geographic audit is missing scenes')
    groups, polygons = {}, {}
    for scene in scene_ids:
        entry = entries[scene]
        if not entry.get('spatial_group') or not entry.get('evidence'):
            raise ValueError(f'Missing geographic group/evidence: {scene}')
        groups[scene] = entry['spatial_group']
        if entry.get('footprint_wgs84') is not None:
            polygon = shape(entry['footprint_wgs84'])
            if polygon.geom_type not in ('Polygon', 'MultiPolygon') or not polygon.is_valid or polygon.is_empty:
                raise ValueError(f'Invalid footprint: {scene}')
            if not (-180 <= polygon.bounds[0] <= polygon.bounds[2] <= 180 and
                    -90 <= polygon.bounds[1] <= polygon.bounds[3] <= 90):
                raise ValueError(f'Footprint must use WGS84 longitude/latitude: {scene}')
            polygons[scene] = polygon
    keys = sorted(polygons)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            if polygons[a].intersection(polygons[b]).area > 0 and groups[a] != groups[b]:
                raise ValueError(f'Overlapping territories need one spatial group: {a}, {b}')
    return groups


def farthest_frames(rows, count=FRAMES_PER_SCENE):
    rows = sorted(rows, key=lambda r: stable_seed(SEED, r['scene_id'], r['sample_id']))
    if len(rows) < count:
        raise ValueError('Insufficient eligible frames')
    xy = np.asarray([r['xyz_enu'][:2] for r in rows], float)
    chosen = [0]
    distance = np.full(len(rows), np.inf)
    while len(chosen) < count:
        distance = np.minimum(distance, np.linalg.norm(xy-xy[chosen[-1]], axis=1))
        distance[chosen] = -1
        nxt = int(np.argmax(distance))
        if distance[nxt] <= 0:
            raise ValueError('Insufficient distinct horizontal frame positions')
        chosen.append(nxt)
    return [rows[i] for i in chosen]


def prepare_selection(inventory_path, audit_path, output, *, sample_reader=read_sample):
    data = load_verified(inventory_path)
    readiness = selection_readiness(data)
    if readiness['eligible_scene_count'] < 16:
        raise ValueError(f'Need 16 eligible scenes with at least 10 distinct frame positions; '
                         f'available={readiness["eligible_scene_count"]}; '
                         f'add at least {readiness["minimum_additional_scenes"]} eligible scene(s). '
                         f'Insufficient scenes: {readiness["insufficient_scenes"]}. '
                         'Existing inventory is reusable; no geographic reviewed=true flag is required.')
    previous = read(audit_path) if audit_path is not None and Path(audit_path).exists() else None
    audit = scene_policy({r['scene_id'] for r in data['rows']}, previous)
    by_scene = {}
    for row in data['rows']:
        if row['eligible']:
            by_scene.setdefault(row['scene_id'], []).append(row)
    eligible = readiness['eligible_scenes']
    groups = audit_groups(audit, eligible)
    # Exact duplicate aerial maps must not cross splits. Reuse hashes required
    # by the existing source manifest; no image decoding or localization here.
    references_all, map_owners, duplicate_maps = {}, {}, []
    fingerprint = reference_identity
    if fingerprint is _REFERENCE_IDENTITY:
        from .fingerprints import cached_reference_identity
        fingerprint = lambda path, mode: cached_reference_identity(
            path, mode, Path(output).parent/'reference_fingerprints')
    for scene in eligible:
        ref = by_scene[scene][0]['reference_json']
        hashes = fingerprint(ref, 'aerial')
        references_all[scene] = hashes
        meta = read(ref).get('modes', {}).get('aerial', {})
        map_key = Path(meta.get('map_path', '')).name
        map_hash = hashes.get(map_key)
        if map_hash is not None:
            if map_hash in map_owners:
                other = map_owners[map_hash]
                duplicate_maps.append([other, scene])
                old_group, new_group = groups[scene], groups[other]
                groups = {s: new_group if group == old_group else group for s, group in groups.items()}
            else:
                map_owners[map_hash] = scene
    for scene, group in groups.items():
        audit['scenes'][scene]['spatial_group'] = group
    audit['map_overlap_check'] = dict(exact_duplicate_pairs=duplicate_maps,
        exact_map_hashes_checked=len(map_owners) > 0, partial_visual_overlap='not_verified')
    # One scene per known group. Prefer previously reviewed scenes for train.
    ordered = sorted(eligible, key=lambda s: (s not in REVIEWED, stable_seed(SEED, s, 'scene')))
    unique, seen = [], set()
    for scene in ordered:
        if groups[scene] not in seen:
            unique.append(scene)
            seen.add(groups[scene])
    fresh = [s for s in unique if s not in REVIEWED]
    if len(unique) < 16 or len(fresh) < 4:
        raise ValueError(f'Need 16 eligible scene groups and 4 previously unreviewed test scenes; '
                         f'available={len(unique)}, unreviewed={len(fresh)}, scenes={unique}')
    test = sorted(fresh, key=lambda s: stable_seed(SEED, s, 'test'))[:4]
    remaining = [s for s in unique if s not in test]
    split_scenes = {'test': test, 'train': remaining[:8],
                    'validation': remaining[8:10], 'calibration': remaining[10:12]}
    selected = []
    references = {}
    for split in SPLIT_SCENES:
        for scene in split_scenes[split]:
            chosen = farthest_frames(by_scene[scene])
            for row in chosen:
                for field in ('path', 'reference_json'):
                    if not Path(row[field]).is_file():
                        raise FileNotFoundError(row[field])
                s = sample_reader(row['path'], main_config()['pose_convention'], main_config()['max_image_side'])
                if (s['scene_id'], s['sample_id']) != (scene, row['sample_id']) or not np.allclose(s['xyz_enu'], row['xyz_enu']):
                    raise ValueError('Frame metadata changed after inventory')
                selected.append(dict(row, split=split, spatial_group=groups[scene], source_identity=s['source_identity']))
            references[scene] = references_all[scene]
    result = dict(protocol_id=PROTOCOL_ID, rows=selected, seed=SEED, split_scenes=split_scenes,
                  inventory_sha256=digest(data), audit_sha256=digest(audit),
                  source_audit_sha256=file_hash(audit_path) if previous is not None else None,
                  spatial_overlap_audit=audit, references=references, expected_attempts=1920)
    validate_selection(result)
    save_verified(output, result)
    return dict(frames=len(selected), expected_attempts=1920, split_scenes=split_scenes)


def validate_selection(selection):
    if selection.get('protocol_id') != PROTOCOL_ID:
        raise ValueError('Selection protocol changed; prepare a new selection using the existing inventory')
    rows = selection['rows']
    if len(rows) != 160 or len({(r['scene_id'], r['sample_id']) for r in rows}) != 160:
        raise ValueError('Main selection requires 160 unique frames')
    membership = {}
    scene_split = {}
    for row in rows:
        split = row['split']
        if split not in SPLIT_SCENES:
            raise ValueError('Invalid split')
        for key in (('scene', row['scene_id']), ('group', row['spatial_group'])):
            if key in membership and membership[key] != split:
                raise ValueError('Scene/spatial leakage')
            membership[key] = split
        scene_split.setdefault(row['scene_id'], []).append(row)
        if split == 'test' and row['scene_id'] in REVIEWED:
            raise ValueError('Previously reviewed territory cannot be final test')
    if len(scene_split) != 16 or any(len(v) != 10 for v in scene_split.values()):
        raise ValueError('Exactly 10 frames per scene required')
    for split, count in SPLIT_SCENES.items():
        if len({r['scene_id'] for r in rows if r['split'] == split}) != count:
            raise ValueError('Incorrect scene split counts')
    groups = audit_groups(selection['spatial_overlap_audit'], list(scene_split))
    if any(groups[r['scene_id']] != r['spatial_group'] for r in rows):
        raise ValueError('Selection differs from audited spatial groups')
