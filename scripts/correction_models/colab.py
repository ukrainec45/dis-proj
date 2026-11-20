"""GPU backend initialization using the existing pinned visibility experiment."""
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
import tempfile

from .environment import configure_cuda_determinism, verify_cuda_matmul, cuda_execution_identity


CAMP_FILENAME = 'weights_0.9446_for_U1652.pth'


def _check_checkpoint_file(path):
    """Reject missing/empty/HTML responses before publishing a download.

    This is only a container check. The complete tensor/key/shape validation
    happens in camp_from_complete_checkpoint before backbone initialization.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'CAMP checkpoint not found: {path}')
    with path.open('rb') as stream:
        signature = stream.read(4)
    if not (signature.startswith(b'PK\x03\x04') or signature.startswith(b'\x80')):
        raise ValueError(f'Invalid CAMP checkpoint (empty file or non-PyTorch response): {path}')


def ensure_camp_checkpoint(checkpoint=None, *, cache_dir=None, downloader=None):
    """Resolve an override or download official CAMP weights into this cache.

    None means automatic acquisition, not a missing required setting. Explicit
    paths are never silently replaced. No network access occurs at import time.
    """
    if checkpoint is not None:
        path = Path(checkpoint)
        _check_checkpoint_file(path)
        print(f'CAMP: using supplied checkpoint {path}', flush=True)
        return path
    cache_dir = cache_dir or os.environ.get('VISIBILITY_MODEL_CACHE')
    if cache_dir is None:
        raise ValueError('Set a persistent model cache directory for automatic CAMP download')
    cache = Path(cache_dir)
    cached = cache/CAMP_FILENAME
    if cached.exists():
        _check_checkpoint_file(cached)
        print(f'CAMP: reusing cached checkpoint {cached}', flush=True)
        return cached
    from scripts.visibility_localization.colab import CAMP_URL
    if downloader is None:
        import gdown
        downloader = gdown.download
    cache.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='camp-', suffix='.download', dir=cache)
    os.close(handle)
    partial = Path(name)
    try:
        print(f'CAMP: downloading official weights to {cached}', flush=True)
        result = downloader(CAMP_URL, str(partial), fuzzy=True, quiet=False)
        if not result:
            raise RuntimeError('CAMP download did not return a file')
        _check_checkpoint_file(partial)
        partial.replace(cached)
    except Exception as exc:
        raise RuntimeError(
            f'Could not download CAMP: {exc}. Official source: {CAMP_URL}. '
            'Retry the model cell, or set CAMP_CHECKPOINT to an existing official checkpoint.') from exc
    finally:
        partial.unlink(missing_ok=True)
    return cached


@contextmanager
def camp_from_complete_checkpoint(source, checkpoint):
    """Same checked complete-backbone initialization as the existing notebook."""
    import torch
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module('Retrieval_Models.CAMP.hand_convnext.ConvNext.make_model')
    original = module.create_model

    def factory(name, *args, **kwargs):
        if name != 'convnext_base':
            raise RuntimeError('Unexpected CAMP backbone')
        kwargs['pretrained'] = False
        model = original(name, *args, **kwargs)
        state = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
        prefix = 'model_1.convnext.'
        backbone = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        expected = model.state_dict()
        if set(backbone) != set(expected) or any(backbone[k].shape != v.shape for k, v in expected.items()):
            raise RuntimeError('CAMP checkpoint does not completely cover the backbone')
        model.load_state_dict(backbone, strict=True)
        return model

    module.create_model = factory
    try:
        yield
    finally:
        module.create_model = original


def initialize_backend(runtime_root, checkpoint=None):
    """May clone/download weights only when explicitly called by the user."""
    configure_cuda_determinism()
    verify_cuda_matmul()
    from scripts.visibility_localization.colab import checkout_upstream
    from scripts.visibility_localization.backend import AnyVisLocBackend
    from .config import main_config
    checkpoint = ensure_camp_checkpoint(checkpoint, cache_dir=(
        os.environ.get('VISIBILITY_MODEL_CACHE') or Path(runtime_root).parent/'model_cache'))
    source = checkout_upstream(Path(runtime_root)/'Benchmark')
    from .profiling import Profile
    import torch
    with Profile(torch) as initialization:
        with camp_from_complete_checkpoint(source, checkpoint):
            backend = AnyVisLocBackend(source, main_config(), checkpoint)
    backend.initialization_profile = initialization.result
    backend.identity['cuda_execution'] = cuda_execution_identity(torch)
    print('CUDA execution:', backend.identity['cuda_execution'], flush=True)
    backend.scratch = Path(runtime_root)/'localization_scratch'
    backend.scratch.mkdir(parents=True, exist_ok=True)
    return backend
