"""Public contracts for the horizontal INS/OpenVINS propagation model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


ERROR_STATE_FIELDS = ("dx", "dy", "dvx", "dvy", "dyaw")
ERROR_STATE_DIM = len(ERROR_STATE_FIELDS)


class PropagationMode(str, Enum):
    VIO_OPERATIONAL = "vio_operational"
    VIO_DEGRADED = "vio_degraded"
    INS_ONLY = "ins_only"
    REINITIALIZED = "reinitialized"


PROPAGATION_MODES = tuple(mode.value for mode in PropagationMode)


def _covariance(value, label):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (ERROR_STATE_DIM, ERROR_STATE_DIM):
        raise ValueError(f"{label} must have shape ({ERROR_STATE_DIM}, {ERROR_STATE_DIM})")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be finite")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-8):
        raise ValueError(f"{label} must be symmetric")
    if np.linalg.eigvalsh(matrix).min() < -1e-9:
        raise ValueError(f"{label} must be positive semidefinite")
    return matrix


@dataclass(frozen=True)
class PropagationQuery:
    duration_s: float
    incoming_covariance: np.ndarray
    incoming_mode: PropagationMode | str
    visual_regime: str

    def __post_init__(self):
        duration = float(self.duration_s)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration_s must be finite and positive")
        mode = (self.incoming_mode if isinstance(self.incoming_mode, PropagationMode)
                else PropagationMode(str(self.incoming_mode)))
        regime = str(self.visual_regime).strip()
        if not regime:
            raise ValueError("visual_regime must be non-empty")
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "incoming_covariance",
                           _covariance(self.incoming_covariance, "incoming_covariance"))
        object.__setattr__(self, "incoming_mode", mode)
        object.__setattr__(self, "visual_regime", regime)


@dataclass(frozen=True)
class RelativeErrorDistribution:
    """Moment and empirical-quantile description of one relative-error increment."""

    mean: np.ndarray
    covariance: np.ndarray
    quantile_levels: np.ndarray
    absolute_error_quantiles: np.ndarray

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=float)
        levels = np.asarray(self.quantile_levels, dtype=float)
        quantiles = np.asarray(self.absolute_error_quantiles, dtype=float)
        if mean.shape != (ERROR_STATE_DIM,) or not np.isfinite(mean).all():
            raise ValueError(f"mean must be a finite {ERROR_STATE_DIM}-vector")
        if levels.ndim != 1 or not len(levels) or not np.all((levels > 0) & (levels < 1)):
            raise ValueError("quantile_levels must be a non-empty vector inside (0, 1)")
        if quantiles.shape != (len(levels), 3) or not np.isfinite(quantiles).all():
            raise ValueError("absolute_error_quantiles must have shape (quantile, 3)")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance",
                           _covariance(self.covariance, "relative error covariance"))
        object.__setattr__(self, "quantile_levels", levels)
        object.__setattr__(self, "absolute_error_quantiles", quantiles)


@dataclass(frozen=True)
class PropagationOutcome:
    outgoing_covariance: np.ndarray
    outgoing_mode_probabilities: dict[PropagationMode, float]
    relative_error_distribution: RelativeErrorDistribution

    def __post_init__(self):
        covariance = _covariance(self.outgoing_covariance, "outgoing_covariance")
        probabilities = {
            (key if isinstance(key, PropagationMode) else PropagationMode(str(key))): float(value)
            for key, value in self.outgoing_mode_probabilities.items()
        }
        if set(probabilities) != set(PropagationMode):
            raise ValueError("outgoing_mode_probabilities must contain every propagation mode")
        values = np.asarray(list(probabilities.values()), dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("outgoing mode probabilities must be finite and non-negative")
        if not np.isclose(values.sum(), 1.0, atol=1e-8):
            raise ValueError("outgoing mode probabilities must sum to one")
        object.__setattr__(self, "outgoing_covariance", covariance)
        object.__setattr__(self, "outgoing_mode_probabilities", probabilities)
