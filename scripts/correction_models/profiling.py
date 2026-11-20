"""Inline RSS sampling and synchronized CUDA peak measurements, never a separate run."""
import threading
import time

import psutil


class Profile:
    def __init__(self, torch=None):
        self.torch = torch
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.peak = 0

    def _sample(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.stop.wait(.02)

    def __enter__(self):
        if self.torch is not None:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        self.start_rss = self.process.memory_info().rss
        self.peak = self.start_rss
        self.started = time.perf_counter()
        self.worker = threading.Thread(target=self._sample, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *exc):
        sync_error = None
        try:
            # A failed CUDA operation can poison the context. Preserve the body
            # exception and avoid making further CUDA calls in that case.
            if self.torch is not None and exc[0] is None:
                self.torch.cuda.synchronize()
        except Exception as error:
            sync_error = error
        finally:
            self.stop.set()
            self.worker.join()
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.result = dict(elapsed_s=time.perf_counter()-self.started,
                               process_rss_start_bytes=self.start_rss, sampled_peak_rss_bytes=self.peak,
                               rss_sampling_interval_s=.02)
            if self.torch is not None and exc[0] is None and sync_error is None:
                self.result.update(cuda_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=self.torch.cuda.max_memory_reserved(),
                    device=self.torch.cuda.get_device_name(),
                    device_total_memory_bytes=self.torch.cuda.get_device_properties(0).total_memory)
            elif self.torch is not None:
                self.result['cuda_profile_incomplete'] = True
        if sync_error is not None:
            raise sync_error
        return False
