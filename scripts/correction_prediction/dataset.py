"""Explicit spatial split audit and checksum-protected label/feature separation."""
from collections import defaultdict

import numpy as np

from .artifacts import digest, load_verified, save_verified
from .features import ALL_FEATURES, FEATURE_VERSION
from .contracts import PROTOCOL_ID

REVIEWED = {'Scene_05', 'Scene_07', 'Scene_08', 'Scene_11', 'Scene_21'}
SPLITS = {'pilot', 'train', 'validation', 'calibration', 'test'}


def validate_rows(rows, *, require_main=False):
    if not rows:
        raise ValueError('Empty dataset')
    ids = set()
    memberships = defaultdict(set)
    for row in rows:
        if row['attempt_id'] in ids:
            raise ValueError('Duplicate attempt identifier')
        ids.add(row['attempt_id'])
        if row['split'] not in SPLITS or not row['spatial_group']:
            raise ValueError('Explicit split and audited spatial group required')
        for key in (('scene', row['scene_id']), ('spatial', row['spatial_group']),
                    ('frame', row['scene_id'], row['sample_id'])):
            memberships[key].add(row['split'])
        if row['split'] == 'test' and row['scene_id'] in REVIEWED:
            raise ValueError('Previously reviewed scene cannot be a final test')
        f = row['features']
        if f is not None and (set(f) != set(ALL_FEATURES) or not np.isfinite(list(f.values())).all()):
            raise ValueError('Feature allowlist/finite check failed')
        label = row['label']
        if not label['technical_error']:
            cls = label['class_index']
            if type(cls) is not int or not 0 <= cls < 8 or not np.isfinite(label['delta_m']):
                raise ValueError('Invalid class/delta label')
            from .contracts import EDGES
            expected = 0 if not label['accepted'] else int(np.searchsorted(EDGES, label['delta_m'], side='right')+1)
            if cls != expected or (not label['accepted'] and label['delta_m'] != 0):
                raise ValueError('Delta/class disagreement')
            before, after = label['error_before_m'], label['error_after_m']
            if (not np.isfinite([before, after]).all() or min(before, after) < 0
                    or not np.isclose(before-after, label['delta_m'], atol=1e-7)):
                raise ValueError('Error/delta disagreement')
            for vector, norm in [('error_before_xy_m', before), ('error_after_xy_m', after)]:
                v = np.asarray(label[vector], float)
                if v.shape != (2,) or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v), norm):
                    raise ValueError('Error vector/norm disagreement')
    if any(len(v) > 1 for v in memberships.values()):
        raise ValueError('Scene/frame/spatial-group leakage across splits')
    if require_main:
        if not {'train', 'validation', 'calibration', 'test'} <= {r['split'] for r in rows}:
            raise ValueError('Main campaign needs train/validation/calibration/test groups')
        if any(r['split'] == 'pilot' for r in rows):
            raise ValueError('Pilot must not be relabelled as a main campaign')


def save_dataset(path, rows, provenance):
    validate_rows(rows)
    save_verified(path, dict(protocol_id=PROTOCOL_ID, feature_version=FEATURE_VERSION,
                             provenance=provenance, rows=rows))


def load_dataset(path):
    data = load_verified(path)
    if data['protocol_id'] != PROTOCOL_ID or data['feature_version'] != FEATURE_VERSION:
        raise ValueError('Dataset protocol mismatch')
    validate_rows(data['rows'])
    return data


def usable(rows, split):
    return [r for r in rows if r['split'] == split and r['features'] is not None
            and not r['label']['technical_error']]


def dataset_id(data):
    return digest(data)
