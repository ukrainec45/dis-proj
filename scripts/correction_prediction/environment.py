"""Detect pip/kernel version mixing before importing the GPU model stack."""
from importlib import metadata, import_module
import os
import sys


CUBLAS_CONFIGS = (':4096:8', ':16:8')


def configure_cuda_determinism():
    """Configure cuBLAS before CUDA initialization; retain strict determinism."""
    current = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    torch = sys.modules.get('torch')
    if current not in CUBLAS_CONFIGS and torch is not None and torch.cuda.is_initialized():
        raise RuntimeError(
            'CUDA was already initialized without a supported CUBLAS_WORKSPACE_CONFIG. '
            'Use Runtime -> Restart session, then set CUBLAS_WORKSPACE_CONFIG=:4096:8 '
            'before importing/initializing PyTorch. Do not rerun the pilot in this kernel.')
    if current is not None and current not in CUBLAS_CONFIGS:
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for this experiment')
    if current is None:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    return os.environ['CUBLAS_WORKSPACE_CONFIG']


def verify_cuda_matmul():
    """Exercise CUDA matrix and matrix-vector products before loading weights.

    The previous global determinism setting is restored; the localization
    backend subsequently enables its own strict deterministic setting.
    """
    configure_cuda_determinism()
    torch = import_module('torch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; select a GPU runtime before starting the pilot')
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        matrix = torch.ones((16, 16), device='cuda')
        vector = torch.ones(16, device='cuda')
        for result in (matrix @ matrix, matrix @ vector):
            torch.cuda.synchronize()
            if not torch.equal(result, torch.full_like(result, 16)):
                raise RuntimeError('Unexpected CUDA matrix-product preflight result')
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


MODULES = {'numpy': 'numpy', 'scipy': 'scipy', 'scikit-learn': 'sklearn',
           'pandas': 'pandas', 'matplotlib': 'matplotlib', 'Pillow': 'PIL',
           'shapely': 'shapely', 'pyproj': 'pyproj', 'rasterio': 'rasterio',
           'torch': 'torch', 'torchvision': 'torchvision', 'timm': 'timm',
           'kornia': 'kornia', 'opencv-python-headless': 'cv2'}


def loaded_snapshot():
    result = {}
    for distribution, module in MODULES.items():
        if module in sys.modules or any(name.startswith(module+'.') for name in sys.modules):
            try:
                result[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                result[distribution] = None
    return result


def _same_version(distribution, loaded, installed):
    if loaded is None or installed is None:
        return True
    loaded, installed = str(loaded).split('+')[0], str(installed).split('+')[0]
    # cv2 exposes 4.10.0, whereas the distribution exposes 4.10.0.84.
    if distribution == 'opencv-python-headless':
        return installed == loaded or installed.startswith(loaded+'.')
    return loaded == installed


def require_clean_kernel(before=None):
    after = loaded_snapshot()
    changed = {name for name, version in (before or {}).items() if after.get(name) != version}
    for distribution, installed in after.items():
        module = sys.modules.get(MODULES[distribution])
        if not _same_version(distribution, getattr(module, '__version__', None), installed):
            changed.add(distribution)
    if changed:
        raise RuntimeError(
            'Already-imported libraries changed on disk: '+', '.join(sorted(changed))+'. '
            'Use Runtime -> Restart session, then set INSTALL_DEPENDENCIES=False and rerun from the top. '
            'Do not initialize localization models in this kernel.')


def verify_scipy_imports():
    try:
        import_module('scipy.optimize._numdiff')
        import_module('scipy.sparse')
    except (ImportError, AttributeError, ValueError) as exc:
        raise RuntimeError(
            'SciPy import consistency check failed. Restart the Python session first and set '
            'INSTALL_DEPENDENCIES=False. If this also fails in a fresh Python process, '
            'repair the pinned SciPy installation, then restart again.') from exc
