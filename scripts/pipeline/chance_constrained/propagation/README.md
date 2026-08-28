# INS/OpenVINS propagation model

This module implements the standalone propagation component

```text
(incoming covariance, incoming mode, duration, visual regime)
    -> (outgoing covariance, mode probabilities, relative-error distribution)
```

It does not search for a route, execute VML/TAN, or modify `scene_edges.npz`.
An edge will later supply its travel time and visual regime through a
`PropagationQuery`.

## Input data

A single Ubuntu/OpenVINS replay is converted into a standardized `RunLog` that
contains the `[px, py, vx, vy, yaw]` state, ground truth, reported covariance,
accepted visual updates, tracked-feature count, processing time, visual regime,
and forced camera availability. `openvins_adapter.py` checks that the data are
finite and synchronizes all streams by timestamp.

The Ubuntu replay procedure and collection of the 12 measured logs are
documented in `ubuntu/README.md`. Synthetic fixtures exist only inside the unit
tests and do not constitute a research model.

## Building the artifact

```bash
export EUROC_PROPAGATION_ROOT=/absolute/path/to/euroc_propagation
python -m scripts.pipeline.chance_constrained.propagation.build_model \
  --config configs/chance_constrained/openvins/euroc_monocular_baseline.example.json \
  --output-dir "$EUROC_PROPAGATION_ROOT/model"
```

Outputs:

- `segments.csv` — measured relative errors for every 2/5/10-second window;
- `ins_openvins_model.npz` — mean increments, `Q`, empirical quantiles,
  mode-transition counts, probabilities and confidence bounds, and support
  flags;
- `summary.json` — a human-readable validation result;
- `manifest.json` — the configuration and SHA-256 fingerprints of the
  standardized logs and source manifest.

A failed validation does not discard the diagnostic outputs, but the artifact
is marked with `validation_passed=false`. The runtime class refuses to use such
an artifact by default.

## Runtime query

```python
import numpy as np
from scripts.pipeline.chance_constrained.propagation import (
    InsOpenVinsPropagationModel,
    PropagationQuery,
)

model = InsOpenVinsPropagationModel.load("/path/ins_openvins_model.npz")
outcome = model.propagate(PropagationQuery(
    duration_s=5.0,
    incoming_covariance=np.eye(5),
    incoming_mode="vio_operational",
    visual_regime="nominal_20hz",
))
```

A query is allowed only within the measured duration range and for groups with
sufficient empirical support. Parameters for intermediate durations are
interpolated between adjacent bins. Extrapolation outside 2–10 seconds is
rejected.
