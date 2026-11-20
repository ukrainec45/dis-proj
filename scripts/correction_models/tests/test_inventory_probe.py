"""Probe the metadata operation without running the main dataset inventory."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.correction_models.inventory_probe import _worker, _stop_reason, probe_inventory


@pytest.mark.parametrize('elapsed,rss,available,expected', [
    (0, 200, 2000, None),
    (0, 200, 100, 'low system RAM (possibly outside the Python child)'),
    (0, 1100, 2000, 'child RSS limit'),
    (121, 200, 2000, 'time limit'),
])
def test_probe_stop_conditions(elapsed, rss, available, expected):
    assert _stop_reason(elapsed, rss*1024**2, available*1024**2,
                        max_seconds=120, max_rss_mb=1024, min_available_mb=1536) == expected


def test_worker_traces_files_and_does_not_create_inventory(tmp_path, monkeypatch, capsys):
    limits = []
    monkeypatch.setitem(sys.modules, 'resource', SimpleNamespace(
        RLIMIT_AS=9, setrlimit=lambda *args: limits.append(args)))
    directory = tmp_path/'Scene_01'
    directory.mkdir()
    (directory/'L01_reference.json').write_text('{}')
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 80]
    for index in (1, 2):
        # No image at all: tracing must access metadata only.
        np.savez(directory/f'L01_{index:04d}.npz', xyz=pose[:3, 3], pose_c2w=pose,
                 euler_deg=np.zeros(3))
    _worker(tmp_path, 1, 3072)
    output = capsys.readouterr().out
    assert limits == [(9, (3072*1024**2, 3072*1024**2))]
    assert 'sha256=' in output and 'read xyz.npy:' in output
    assert 'BEGIN [1/1]' in output and 'END L01_0001.npz' in output
    assert 'L01_0002' not in output
    assert sorted(p.name for p in tmp_path.iterdir()) == ['Scene_01']


def test_probe_requires_linux_limits(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'win32')
    with pytest.raises(RuntimeError, match='Linux/Colab'):
        probe_inventory('unused')


def test_probe_refuses_low_memory_without_starting_worker(monkeypatch):
    import psutil
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=100*1024**2))
    with pytest.raises(MemoryError, match='restart the session'):
        probe_inventory('unused')
