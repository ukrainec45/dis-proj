"""Official checkpoint acquisition shared by visibility and control notebooks."""
from pathlib import Path
import os
import shutil
import time
from urllib.error import HTTPError, URLError

from .core import file_hash


ROMA_WEIGHTS = {
    'Matching_Models/RoMa/ckpt/roma_outdoor.pth':
        'https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth',
    'Matching_Models/RoMa/ckpt/dinov2_vitl14_pretrain.pth':
        'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
}


def download_roma_weights(source, *, cache_dir=None, attempts=4):
    """Retry transient failures and optionally reuse persistent official weights.

    VISIBILITY_MODEL_CACHE may point to Drive; local source paths expected by the
    author stay unchanged. Only completed downloads are published to the cache.
    """
    import torch
    if attempts < 1:
        raise ValueError('attempts must be positive')
    cache_dir = cache_dir or os.environ.get('VISIBILITY_MODEL_CACHE')
    hashes = {}
    for relative, url in ROMA_WEIGHTS.items():
        path = Path(source)/relative
        cached = Path(cache_dir)/Path(relative).name if cache_dir else None
        path.parent.mkdir(parents=True, exist_ok=True)
        if (not path.is_file() or path.stat().st_size == 0) and cached is not None and cached.is_file() and cached.stat().st_size:
            print(f'Reusing cached model: {cached}', flush=True)
            shutil.copy2(cached, path)
        if not path.is_file() or path.stat().st_size == 0:
            partial = path.with_suffix('.download')
            for attempt in range(1, attempts+1):
                print(f'Downloading {path.name} ({attempt}/{attempts}): {url}', flush=True)
                try:
                    torch.hub.download_url_to_file(url, str(partial), progress=True)
                    if not partial.is_file() or partial.stat().st_size == 0:
                        raise OSError(f'Empty model download: {url}')
                    partial.replace(path)
                    break
                except (URLError, TimeoutError, ConnectionError, OSError) as exc:
                    partial.unlink(missing_ok=True)
                    transient = not isinstance(exc, HTTPError) or exc.code in (408,429,500,502,503,504)
                    if not transient or attempt == attempts:
                        raise RuntimeError(
                            f'Cannot download {path.name}: {exc}. Official URL: {url}. '
                            f'Rerun the model cell, or place the official file in {cached or path}.') from exc
                    delay = min(2**attempt, 16)
                    print(f'Temporary download failure: {exc}; retry in {delay}s', flush=True)
                    time.sleep(delay)
        hashes[relative] = file_hash(path)
        if cached is not None and cached.resolve() != path.resolve():
            if not cached.is_file() or file_hash(cached) != hashes[relative]:
                cached.parent.mkdir(parents=True, exist_ok=True)
                temporary = cached.with_suffix('.download')
                shutil.copy2(path, temporary)
                temporary.replace(cached)
    return hashes
