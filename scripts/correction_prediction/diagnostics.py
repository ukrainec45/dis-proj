"""Read saved attempts and expose actual exceptions without rerunning a model."""
from collections import Counter
from pathlib import Path

from .artifacts import load_verified


def diagnose_campaign(output, examples=3):
    if type(examples) is not int or examples < 1:
        raise ValueError('Positive example count required')
    paths = sorted((Path(output)/'records').glob('*.json'))
    if not paths:
        raise FileNotFoundError(f'No saved attempts in {Path(output)/"records"}')
    counts, samples = Counter(), {}
    technical = 0
    for path in paths:
        row = load_verified(path)
        if not row['label']['technical_error']:
            continue
        technical += 1
        prediction = row['localization']
        key = (prediction.get('exception_type', 'unknown'), prediction.get('exception', 'message unavailable'))
        counts[key] += 1
        if key not in samples:
            samples[key] = dict(record=str(path), scene_id=row['scene_id'], sample_id=row['sample_id'],
                                condition=row['condition'], traceback=prediction.get('traceback'))
    return dict(saved_attempts=len(paths), technical_errors=technical,
        errors=[dict(exception_type=k[0], message=k[1], count=n) for k, n in counts.most_common()],
        examples=[dict(exception_type=k[0], message=k[1], **samples[k]) for k, _ in counts.most_common(examples)])
