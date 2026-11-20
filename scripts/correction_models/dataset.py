"""Validate paired main data and load checksum-verified map descriptors lazily."""
from pathlib import Path
import numpy as np

from .artifacts import digest, load_verified
from .config import SPLIT_SCENES, REVIEWED
from .contracts import PROTOCOL_ID, EDGES, CONDITIONS
from .features import ALL_FEATURES


def validate_rows(rows, *, require_complete=False):
    ids, membership, cases, counts = set(), {}, {}, {}
    if not rows:
        raise ValueError('Empty dataset')
    for row in rows:
        if row['attempt_id'] in ids:
            raise ValueError('Duplicate attempt')
        ids.add(row['attempt_id'])
        if row['split'] not in SPLIT_SCENES or row['condition'] not in CONDITIONS:
            raise ValueError('Invalid split/condition')
        if row['split'] == 'test' and row['scene_id'] in REVIEWED:
            raise ValueError('Previously reviewed test scene')
        for key in (('scene', row['scene_id']), ('spatial', row['spatial_group']), ('frame', row['scene_id'], row['sample_id'])):
            if key in membership and membership[key] != row['split']:
                raise ValueError('Scene/spatial/frame leakage')
            membership[key] = row['split']
        if row['sigma_xy_m'] not in (5., 20., 50., 100.):
            raise ValueError('Unapproved prior uncertainty')
        case = (row['scene_id'], row['sample_id'], row['sigma_xy_m'])
        previous = cases.setdefault(case, {})
        if row['condition'] in previous:
            raise ValueError('Duplicate paired condition')
        previous[row['condition']] = row
        counts.setdefault(row['split'], set()).add(row['scene_id'])
        features, label = row['features'], row['label']
        if features is not None:
            if set(features) != set(ALL_FEATURES) or not np.isfinite(list(features.values())).all():
                raise ValueError('Feature allowlist/finite check failed')
            if not row.get('embedding'):
                raise ValueError('Supported row needs CAMP descriptor')
            if (not np.isclose(features['sigma_major_m'], row['sigma_xy_m']) or
                    not np.isclose(features['sigma_minor_m'], row['sigma_xy_m'])):
                raise ValueError('Main campaign requires the declared isotropic prior uncertainty')
            if (features['fog_extreme'] != float(row['condition'] == 'fog_extreme') or
                    features['smoke_extreme'] != float(row['condition'] == 'smoke_extreme')):
                raise ValueError('Visibility features disagree with scenario')
        if label['technical_error']:
            if require_complete:
                raise ValueError('Technical errors cannot enter a complete main dataset')
            continue
        before, after = np.asarray(label['error_before_xy_m']), np.asarray(label['error_after_xy_m'])
        if type(label['accepted']) is not bool or type(label['class_index']) is not int:
            raise ValueError('Boolean acceptance and integer class required')
        if before.shape != (2,) or after.shape != (2,) or not np.isfinite([before, after]).all():
            raise ValueError('Invalid error vectors')
        if not np.isclose(np.linalg.norm(before), label['error_before_m']) or not np.isclose(np.linalg.norm(after), label['error_after_m']):
            raise ValueError('Error vector/norm disagreement')
        delta = label['error_before_m']-label['error_after_m']
        expected = int(np.searchsorted(EDGES, delta, side='right')+1) if label['accepted'] else 0
        if not np.isclose(delta, label['delta_m']) or label['class_index'] != expected:
            raise ValueError('Class/delta disagreement')
        if not label['accepted'] and not np.array_equal(before, after):
            raise ValueError('Rejection must preserve incoming position error')
    for group in cases.values():
        if require_complete and set(group) != set(CONDITIONS):
            raise ValueError('Incomplete paired visibility group')
        values = list(group.values())
        for row in values[1:]:
            if row['prior_seed'] != values[0]['prior_seed'] or row['prior_metadata'] != values[0]['prior_metadata']:
                raise ValueError('Conditions must share the same prior')
            if not row['label']['technical_error'] and not values[0]['label']['technical_error']:
                if row['label']['error_before_xy_m'] != values[0]['label']['error_before_xy_m']:
                    raise ValueError('Paired conditions must share the actual incoming error')
            if row['features'] is not None and values[0]['features'] is not None:
                invariant = set(ALL_FEATURES)-{'fog_extreme', 'smoke_extreme'}
                if any(row['features'][k] != values[0]['features'][k] for k in invariant):
                    raise ValueError('Map/geometry features must be unchanged across paired visibility conditions')
    if require_complete:
        if len(rows) != 1920 or {k: len(v) for k, v in counts.items()} != SPLIT_SCENES:
            raise ValueError('Complete main dataset requires 1920 attempts and 8/2/2/4 scenes')
        frames = {(r['scene_id'], r['sample_id']) for r in rows}
        if len(frames) != 160 or any(sum(r['scene_id'] == s for r in rows) != 120 for s in {r['scene_id'] for r in rows}):
            raise ValueError('Need 160 frames and 120 attempts per scene')


def load_dataset(path):
    data = load_verified(path)
    if data['protocol_id'] != PROTOCOL_ID:
        raise ValueError('Dataset protocol mismatch')
    validate_rows(data['rows'], require_complete=True)
    return data


def usable(rows, split):
    return [r for r in rows if r['split'] == split and r['features'] is not None and not r['label']['technical_error']]


def descriptor(row, root, cache):
    from .campaign import verify_assets
    verify_assets(row, root)
    key = row['embedding']['path']
    if key not in cache:
        saved = load_verified(Path(root)/key)
        vector = np.asarray(saved['vector'], float)
        if vector.shape != (row['embedding']['dimension'],) or not np.isfinite(vector).all():
            raise ValueError('Invalid descriptor shape/value')
        cache[key] = vector.tolist()
    return cache[key]


def model_inputs(rows, root, cache=None):
    cache = {} if cache is None else cache
    return [dict(features=r['features'], camp=descriptor(r, root, cache)) for r in rows]
