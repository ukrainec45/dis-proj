"""Avoid re-reading gigabytes of maps in every bounded Drive session."""
from pathlib import Path

from scripts.visibility_localization.data import relative_file
from scripts.visibility_localization.core import file_hash
from .artifacts import read, digest, save_verified, load_verified


def cached_reference_identity(json_path, mode, cache):
    path = Path(json_path).resolve()
    meta = read(path)['modes'][mode]
    files = [path, relative_file(path.parent, meta['map_path']), relative_file(path.parent, meta['dsm_path'])]
    signature = [dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files]
    cache_path = Path(cache)/(digest(signature)+'.json')
    if cache_path.exists():
        saved = load_verified(cache_path)
        if saved['signature'] != signature:
            raise ValueError('Reference fingerprint cache mismatch')
        return saved['hashes']
    result = {p.name: file_hash(p) for p in files}
    save_verified(cache_path, dict(signature=signature, hashes=result))
    return result
