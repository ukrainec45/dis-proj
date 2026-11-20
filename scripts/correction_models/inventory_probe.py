"""Bounded Colab/Linux diagnostic; inspect at most 25 frames, never run inventory.

The child has an OS address-space limit. The parent also watches child RSS,
available system RAM and wall time. External DriveFS processes are outside the
child's address-space limit; killing the child cannot guarantee their cleanup.
No source file, audit, selection or inventory checkpoint is modified.
"""
from pathlib import Path
import hashlib
import os
import subprocess
import sys
import threading
import time

import psutil


def _worker(root, max_frames, address_space_mb):
    # Colab is Linux. Fail explicitly on platforms without this OS protection.
    import resource
    limit = address_space_mb*1024**2
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    print(f'Child address-space limit: {address_space_mb} MiB', flush=True)
    print('Importing metadata reader', flush=True)
    from scripts.correction_models import selection
    reader_path = Path(selection.__file__)
    print(f'Reader: {reader_path}; sha256={hashlib.sha256(reader_path.read_bytes()).hexdigest()[:16]}',
          flush=True)
    print('Locating first scene', flush=True)
    ref = next(iter(sorted(Path(root).rglob('L??_reference.json'))), None)
    if ref is None:
        raise FileNotFoundError(f'No scene reference under {root}')
    paths = sorted(ref.parent.glob('L??_????.npz'))[:max_frames]
    if not paths:
        raise FileNotFoundError(f'No frames under {ref.parent}')
    process = psutil.Process()
    for index, path in enumerate(paths, 1):
        print(f'BEGIN [{index}/{len(paths)}] {path}; size={path.stat().st_size} bytes; '
              f'RSS={process.memory_info().rss/1024**2:.0f} MiB', flush=True)
        result = selection.read_metadata(path, trace=lambda s: print('  '+s, flush=True))
        print(f'END {path.name}; nadir={result["nadir_deg"]:.3f}; '
              f'RSS={process.memory_info().rss/1024**2:.0f} MiB', flush=True)
    print('Probe finished. Full inventory was NOT started.', flush=True)


def _stop_reason(elapsed, rss_bytes, available_bytes, *, max_seconds, max_rss_mb,
                 min_available_mb):
    if available_bytes < min_available_mb*1024**2:
        return 'low system RAM (possibly outside the Python child)'
    if rss_bytes > max_rss_mb*1024**2:
        return 'child RSS limit'
    if elapsed >= max_seconds:
        return 'time limit'
    return None


def probe_inventory(root, *, max_frames=25, max_seconds=120, max_rss_mb=1024,
                    min_available_mb=1536, address_space_mb=3072, progress=print):
    """Stream trace from an isolated, resource-limited reader; return status.

    Start in a fresh session, after mounting Drive and setting paths. A successful
    probe does not certify all other frames or identify an external DriveFS leak.
    """
    if not sys.platform.startswith('linux'):
        raise RuntimeError('This diagnostic requires Linux/Colab resource limits')
    if not 1 <= max_frames <= 25 or min(max_seconds, max_rss_mb, min_available_mb,
                                      address_space_mb) <= 0:
        raise ValueError('Use 1..25 frames and positive resource limits')
    if psutil.virtual_memory().available < min_available_mb*1024**2:
        raise MemoryError('Not enough available RAM to start probe; restart the session first')
    env = dict(os.environ)
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS'):
        env[variable] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                              str(root), str(max_frames), str(address_space_mb)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env=env)
    def relay():
        for line in child.stdout:
            progress(line.rstrip())
    thread = threading.Thread(target=relay, daemon=True)
    thread.start()
    process = psutil.Process(child.pid)
    started, reason, peak = time.monotonic(), None, 0
    try:
        while child.poll() is None:
            try:
                rss = process.memory_info().rss
            except psutil.NoSuchProcess:
                break
            peak = max(peak, rss)
            reason = _stop_reason(time.monotonic()-started, rss,
                                  psutil.virtual_memory().available,
                                  max_seconds=max_seconds, max_rss_mb=max_rss_mb,
                                  min_available_mb=min_available_mb)
            if reason is not None:
                progress('STOP: '+reason)
                child.kill()
                break
            time.sleep(.1)
        child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        thread.join(timeout=2)
        if not thread.is_alive():
            child.stdout.close()
    result = dict(returncode=child.returncode, stopped_reason=reason,
                  child_peak_rss_mb=round(peak/1024**2), elapsed_s=round(time.monotonic()-started, 1))
    progress(str(result))
    return result


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] != '--worker':
        raise SystemExit('Use probe_inventory(DATASET_ROOT) from the Colab notebook.')
    _worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
