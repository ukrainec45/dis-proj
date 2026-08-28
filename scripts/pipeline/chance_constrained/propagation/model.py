"""Runtime query wrapper over a validated INS/OpenVINS model artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import load_model_artifact
from .contracts import (
    ERROR_STATE_DIM,
    PropagationMode,
    PropagationOutcome,
    PropagationQuery,
    RelativeErrorDistribution,
)


class InsOpenVinsPropagationModel:
    def __init__(self, arrays, *, allow_unvalidated=False):
        self._arrays = arrays
        if not bool(np.asarray(arrays["validation_passed"]).item()) and not allow_unvalidated:
            raise ValueError("INS/OpenVINS artifact failed held-out validation")
        self.model_id = str(np.asarray(arrays["model_id"]).item())
        self._regimes = tuple(map(str, arrays["visual_regime_ids"]))
        self._modes = tuple(map(str, arrays["mode_names"]))
        self._durations = np.asarray(arrays["duration_bins_s"], dtype=float)

    @classmethod
    def load(cls, path, *, allow_unvalidated=False):
        return cls(load_model_artifact(Path(path)),
                   allow_unvalidated=allow_unvalidated)

    @staticmethod
    def _kinematic_transition(duration_s):
        transition = np.eye(ERROR_STATE_DIM)
        transition[0, 2] = duration_s
        transition[1, 3] = duration_s
        return transition

    def _duration_weights(self, duration):
        if duration < self._durations[0] - 1e-9 or duration > self._durations[-1] + 1e-9:
            raise ValueError(
                f"duration_s={duration} is outside calibrated range "
                f"[{self._durations[0]}, {self._durations[-1]}]")
        exact = np.flatnonzero(np.isclose(self._durations, duration, atol=1e-9))
        if len(exact):
            index = int(exact[0])
            return index, index, 0.0
        upper = int(np.searchsorted(self._durations, duration, side="right"))
        lower = upper - 1
        weight = float((duration - self._durations[lower]) /
                       (self._durations[upper] - self._durations[lower]))
        return lower, upper, weight

    def propagate(self, query: PropagationQuery):
        if query.visual_regime not in self._regimes:
            raise KeyError(f"unknown visual regime: {query.visual_regime}")
        regime_index = self._regimes.index(query.visual_regime)
        mode_index = self._modes.index(query.incoming_mode.value)
        lower, upper, weight = self._duration_weights(query.duration_s)
        support = np.asarray(self._arrays["supported"], dtype=bool)
        for duration_index in {lower, upper}:
            if not support[regime_index, mode_index, duration_index]:
                raise ValueError(
                    "requested propagation group has insufficient empirical samples: "
                    f"regime={query.visual_regime}, mode={query.incoming_mode.value}, "
                    f"duration={self._durations[duration_index]}")

        index_lower = (regime_index, mode_index, lower)
        index_upper = (regime_index, mode_index, upper)

        def interpolate(name):
            first = np.asarray(self._arrays[name][index_lower], dtype=float)
            second = np.asarray(self._arrays[name][index_upper], dtype=float)
            return (1.0 - weight) * first + weight * second

        mean = interpolate("drift_mean")
        process_noise = interpolate("process_noise_covariance")
        process_noise = .5 * (process_noise + process_noise.T)
        quantiles = interpolate("absolute_error_quantiles")
        probabilities = interpolate("mode_transition_probability")
        probabilities = np.maximum(probabilities, 0.0)
        probabilities /= probabilities.sum()

        transition = self._kinematic_transition(query.duration_s)
        outgoing_covariance = (
            transition @ query.incoming_covariance @ transition.T + process_noise)
        outgoing_covariance = .5 * (outgoing_covariance + outgoing_covariance.T)
        distribution = RelativeErrorDistribution(
            mean=mean,
            covariance=process_noise,
            quantile_levels=np.asarray(self._arrays["quantile_levels"], dtype=float),
            absolute_error_quantiles=quantiles,
        )
        return PropagationOutcome(
            outgoing_covariance=outgoing_covariance,
            outgoing_mode_probabilities={
                PropagationMode(name): float(value)
                for name, value in zip(self._modes, probabilities)
            },
            relative_error_distribution=distribution,
        )
