"""Test notebook startup before Drive mounting or Torch imports occur."""
from pathlib import Path
from types import SimpleNamespace
import os
import sys

import nbformat
import pytest


def startup():
    notebook = nbformat.read(Path(__file__).parents[1]/'notebooks/01_collect_colab.ipynb', as_version=4)
    cell = next(c.source for c in notebook.cells if c.cell_type == 'code'
                and '# End of CUDA startup' in c.source)
    return compile(cell.split('# End of CUDA startup')[0], '<cuda-startup>', 'exec')


@pytest.mark.parametrize('loaded', [False, True])
def test_inherited_flag_is_cleared_before_cuda_initialization(monkeypatch, loaded):
    monkeypatch.setenv('CUDA_LAUNCH_BLOCKING', '1')
    monkeypatch.setenv('TORCH_LINALG_PREFER_CUSOLVER', '0')
    if loaded:
        monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)))
    else:
        monkeypatch.delitem(sys.modules, 'torch', raising=False)
    exec(startup(), {})
    assert os.environ['CUDA_LAUNCH_BLOCKING'] == '0'
    assert os.environ['TORCH_LINALG_PREFER_CUSOLVER'] == '1'


def test_initialized_cuda_requires_real_python_restart(monkeypatch):
    monkeypatch.setenv('CUDA_LAUNCH_BLOCKING', '1')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True)))
    with pytest.raises(RuntimeError, match='Reconnecting the browser'):
        exec(startup(), {})
    assert os.environ['CUDA_LAUNCH_BLOCKING'] == '1'
