"""Whole-territory learning curves on working validation only."""
import numpy as np

from scripts.visibility_localization.core import stable_seed

from .artifacts import save_verified, manifest
from .dataset import load_dataset, validate_rows, usable, dataset_id
from .models import CANDIDATES, fit_model
from .metrics import evaluate


def learning_curve(dataset_path, output):
    """Diagnostic after an agreed main data collection; never looks at test scores.

    A curve over one training territory is explicitly a single point, not
    evidence that data sufficiency or cross-territory saturation was reached.
    """
    data = load_dataset(dataset_path)
    validate_rows(data['rows'], require_main=True)
    if data['provenance']['stage'] != 'main':
        raise ValueError('Learning curves require main working data, not eight pilot frames')
    train, validation = usable(data['rows'], 'train'), usable(data['rows'], 'validation')
    if not train or not validation:
        raise ValueError('Nonempty train and validation required')
    groups = sorted({r['spatial_group'] for r in train}, key=lambda g: stable_seed(42, g, 'learning_curve'))
    xval = [r['features'] for r in validation]
    yval = np.array([r['label']['class_index'] for r in validation], dtype=int)
    points = []
    for n in range(1, len(groups)+1):
        included = set(groups[:n])
        selected = [r for r in train if r['spatial_group'] in included]
        x = [r['features'] for r in selected]
        y = np.array([r['label']['class_index'] for r in selected], dtype=int)
        scores = {}
        for name in CANDIDATES:
            model = fit_model(name, x, y)
            scores[name] = evaluate(model.predict(xval, False), yval)
        points.append(dict(training_groups=n, training_attempts=len(selected),
                           unique_frames=len({(r['scene_id'], r['sample_id']) for r in selected}),
                           scores=scores))
    result = manifest(dataset_id=dataset_id(data), training_group_order=groups, points=points,
                      final_test_evaluated=False, data_sufficiency_established=False)
    save_verified(output, result)
    return result
