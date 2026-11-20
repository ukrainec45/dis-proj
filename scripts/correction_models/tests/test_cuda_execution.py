"""CPU-only checks of CUDA configuration and error handling, not GPU validation."""
from types import SimpleNamespace

import pytest

from scripts.correction_models.environment import (
    configure_cuda_linalg, cuda_execution_identity, require_collection_cuda)
from scripts.correction_models.profiling import Profile


def fake_torch():
    state = {'library': 'Default'}
    def preferred(value=None):
        if value is not None:
            state['library'] = value.title()
        return '_LinalgBackend.'+state['library']
    return SimpleNamespace(__version__='synthetic', version=SimpleNamespace(cuda='synthetic'),
        cuda=SimpleNamespace(get_device_name=lambda: 'synthetic GPU'),
        backends=SimpleNamespace(cuda=SimpleNamespace(preferred_linalg_library=preferred)))


def test_cusolver_is_selected_and_frozen(monkeypatch):
    monkeypatch.setenv('TORCH_LINALG_PREFER_CUSOLVER', '0')
    monkeypatch.setenv('CUDA_LAUNCH_BLOCKING', '0')
    torch = fake_torch()
    assert configure_cuda_linalg(torch) == 'cusolver'
    identity = cuda_execution_identity(torch)
    assert identity['preferred_linalg_library'] == 'cusolver'
    assert identity['torch_version'] == 'synthetic' and identity['cuda_version'] == 'synthetic'
    backend = SimpleNamespace(torch=torch, identity={'cuda_execution': identity})
    require_collection_cuda(backend)
    torch.backends.cuda.preferred_linalg_library('magma')
    with pytest.raises(RuntimeError, match='requires cuSOLVER'):
        require_collection_cuda(backend)


def test_debug_execution_is_rejected_before_collection(monkeypatch):
    monkeypatch.setenv('CUDA_LAUNCH_BLOCKING', '1')
    with pytest.raises(RuntimeError, match='debugging'):
        require_collection_cuda(SimpleNamespace(torch=fake_torch(), identity={}))


def test_changed_cuda_identity_cannot_silently_resume(monkeypatch):
    monkeypatch.setenv('CUDA_LAUNCH_BLOCKING', '0')
    torch = fake_torch()
    configure_cuda_linalg(torch)
    backend = SimpleNamespace(torch=torch, identity={'cuda_execution': cuda_execution_identity(torch)})
    torch.__version__ = 'another version'
    with pytest.raises(RuntimeError, match='settings changed'):
        require_collection_cuda(backend)


@pytest.mark.parametrize('body_fails', [True, False])
def test_profiler_preserves_primary_failure_and_stops_sampler(body_fails):
    calls = []
    def synchronize():
        calls.append('sync')
        if len(calls) > 1:
            raise RuntimeError('secondary CUDA synchronization failure')
    def forbidden():
        raise AssertionError('Do not inspect GPU stats after CUDA failure')
    torch = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=synchronize, reset_peak_memory_stats=lambda: None,
        max_memory_allocated=forbidden, max_memory_reserved=forbidden))
    profile = Profile(torch)
    error = ValueError('original RoMa error')
    if body_fails:
        with pytest.raises(ValueError) as caught:
            with profile:
                raise error
        assert caught.value is error
        assert calls == ['sync']
    else:
        with pytest.raises(RuntimeError, match='secondary CUDA synchronization failure'):
            with profile:
                pass
    assert profile.result['cuda_profile_incomplete']
    assert not profile.worker.is_alive()
