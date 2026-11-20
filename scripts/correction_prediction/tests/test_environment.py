"""Kernel restart guards without pip changes or heavyweight GPU imports."""
from types import SimpleNamespace

import pytest

from scripts.correction_prediction import environment


def test_changed_loaded_distribution_requires_restart(monkeypatch):
    monkeypatch.setattr(environment, 'loaded_snapshot', lambda: {'scipy': '1.14.1'})
    monkeypatch.setattr(environment.sys, 'modules', {})
    with pytest.raises(RuntimeError, match='Restart session'):
        environment.require_clean_kernel({'scipy': '1.16.0'})


def test_stale_kernel_detected_even_when_install_cell_is_skipped(monkeypatch):
    monkeypatch.setattr(environment, 'loaded_snapshot', lambda: {'scipy': '1.14.1'})
    monkeypatch.setattr(environment.sys, 'modules', {'scipy': SimpleNamespace(__version__='1.16.0')})
    with pytest.raises(RuntimeError, match='scipy'):
        environment.require_clean_kernel()


def test_matching_versions_allow_torch_build_and_opencv_wheel_suffix(monkeypatch):
    monkeypatch.setattr(environment, 'loaded_snapshot', lambda: {'torch': '2.6.0', 'opencv-python-headless': '4.10.0.84'})
    monkeypatch.setattr(environment.sys, 'modules', {
        'torch': SimpleNamespace(__version__='2.6.0+cu126'),
        'cv2': SimpleNamespace(__version__='4.10.0')})
    environment.require_clean_kernel({'torch': '2.6.0', 'opencv-python-headless': '4.10.0.84'})


def test_scipy_smoke_check_reports_restart_with_original_error(monkeypatch):
    def broken(name):
        raise ImportError("cannot import name 'atleast_nd'")
    monkeypatch.setattr(environment, 'import_module', broken)
    with pytest.raises(RuntimeError, match='Restart') as error:
        environment.verify_scipy_imports()
    assert isinstance(error.value.__cause__, ImportError)


def test_cublas_default_is_set_before_cuda_and_valid_override_is_preserved(monkeypatch):
    monkeypatch.setattr(environment.os, 'environ', {})
    torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
    monkeypatch.setitem(environment.sys.modules, 'torch', torch)
    assert environment.configure_cuda_determinism() == ':4096:8'
    environment.os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    assert environment.configure_cuda_determinism() == ':16:8'


def test_late_cublas_configuration_requires_restart_without_mutating_environment(monkeypatch):
    monkeypatch.setattr(environment.os, 'environ', {})
    torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(environment.sys.modules, 'torch', torch)
    with pytest.raises(RuntimeError, match='Restart session'):
        environment.configure_cuda_determinism()
    assert 'CUBLAS_WORKSPACE_CONFIG' not in environment.os.environ


def test_invalid_cublas_configuration_is_not_silently_overridden(monkeypatch):
    monkeypatch.setattr(environment.os, 'environ', {'CUBLAS_WORKSPACE_CONFIG': 'invalid'})
    monkeypatch.setitem(environment.sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)))
    with pytest.raises(ValueError, match=':4096:8 or :16:8'):
        environment.configure_cuda_determinism()


def test_gpu_preflight_restores_deterministic_policy_after_failure(monkeypatch):
    monkeypatch.setattr(environment.os, 'environ', {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    policies = []
    def unavailable_operation(*args, **kwargs):
        raise RuntimeError('Synthetic CUDA failure')
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
        are_deterministic_algorithms_enabled=lambda: False,
        is_deterministic_algorithms_warn_only_enabled=lambda: True,
        use_deterministic_algorithms=lambda enabled, **kwargs: policies.append((enabled, kwargs)),
        ones=unavailable_operation)
    monkeypatch.setattr(environment, 'import_module', lambda name: torch)
    with pytest.raises(RuntimeError, match='Synthetic CUDA failure'):
        environment.verify_cuda_matmul()
    assert policies == [(True, {}), (False, {'warn_only': True})]
