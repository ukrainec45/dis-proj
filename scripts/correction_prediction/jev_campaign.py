"""Explicit request preparation/execution, separated from label evaluation."""
from pathlib import Path

import cv2

from scripts.visibility_localization.core import file_hash

from .artifacts import save_verified, load_verified, digest, assert_frozen_environment
from .dataset import load_dataset, dataset_id
from .jev import JevClient, build_request, PROMPT_VERSION


def run_jev_predictions(dataset_path, output, *, with_image=False, max_requests=0,
                        model='jev-1.14.0', frozen_run=None, cache_dir=None, transport=None):
    if Path(output).exists():
        raise FileExistsError('Use a new Jev prediction output file')
    data = load_dataset(dataset_path)
    identity = dataset_id(data)
    include_test = frozen_run is not None
    if include_test:
        run = load_verified(Path(frozen_run)/'frozen_run.json')
        assert_frozen_environment(run)
        name = 'jev_map' if with_image else 'jev_numeric'
        if run['dataset_id'] != identity or name not in run['external_candidates']:
            raise ValueError('Final-test Jev candidate was not frozen on this dataset')
        a = load_verified(Path(frozen_run)/'models'/(name+'.json'))
        if digest(a) != run['model_ids'][name]:
            raise ValueError('Jev calibration changed after freeze')
        if a['model'] != model or a['prompt_version'] != PROMPT_VERSION:
            raise ValueError('Frozen Jev model/prompt changed')
    splits = {'test'} if include_test else {'pilot', 'validation', 'calibration'}
    client = JevClient(cache_dir or Path(output).parent/'jev_cache', max_requests=max_requests, transport=transport)
    predictions = []
    for row in data['rows']:
        if row['split'] not in splits or row['features'] is None or row['label']['technical_error']:
            continue
        patch = None
        if with_image:
            root = Path(dataset_path).resolve().parent
            path = (root/row['map_context']).resolve()
            if not path.is_relative_to(root) or file_hash(path) != row['map_context_sha256']:
                raise ValueError('Invalid/changed map context')
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Unreadable map context')
            patch = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        body = build_request(row['features'], data['provenance']['localizer_settings'], model, patch)
        p = client.predict(body)
        predictions.append(dict(attempt_id=row['attempt_id'], probabilities=p.tolist()))
    if not predictions:
        raise ValueError('No supported rows in requested Jev split')
    save_verified(output, dict(dataset_id=identity, model=model, prompt_version=PROMPT_VERSION,
                              with_image=with_image, include_test=include_test, predictions=predictions))
    return dict(predictions=len(predictions), output=str(output))
