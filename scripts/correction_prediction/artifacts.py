"""Immutable, checksum-addressed research artifacts. Never persist credentials."""
import hashlib
import json
import os
import platform
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

from .contracts import PROTOCOL_ID


def canonical(value):
    def convert(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'), default=convert)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation deliberately rejects accidental overwrite and concurrent runs.
    with path.open('x', encoding='utf-8') as f:
        f.write(canonical(value)+'\n')


def save_verified(path, value):
    write_once(path, dict(sha256=digest(value), payload=value))


def load_verified(path):
    obj = read(path)
    if digest(obj['payload']) != obj['sha256']:
        raise ValueError(f'Checksum mismatch: {path}')
    return obj['payload']


def runtime():
    result = dict(python=platform.python_version(),
                  cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    for name in ('numpy', 'scipy', 'scikit-learn', 'opencv-python', 'pyproj'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def code_identity():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def dependency_identity():
    root = Path(__file__).parent.parent/'visibility_localization'
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob('*.py')) if not p.name.startswith('test_')}


def assert_frozen_environment(frozen):
    if (frozen['protocol_id'] != PROTOCOL_ID or frozen['source_sha256'] != code_identity()
            or frozen['dependency_sha256'] != dependency_identity() or frozen['runtime'] != runtime()):
        raise ValueError('Frozen code/dependencies/runtime changed; create a new development run')


def manifest(**kwargs):
    return dict(protocol_id=PROTOCOL_ID, runtime=runtime(), source_sha256=code_identity(),
                dependency_sha256=dependency_identity(),
                research_only=True, validation_passed=False, **kwargs)
