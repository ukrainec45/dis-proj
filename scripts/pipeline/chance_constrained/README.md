# Chance-constrained input pipeline

This package contains two separate preprocessing increments for the
chance-constrained planner: the versioned scene/directed-edge artifact and the
empirical INS/OpenVINS propagation-model pipeline. Absolute visual map
localization (VML), TAN, the combined edge operator, and route-success outcome
models remain separate and are not fabricated here.

Run it with:

```powershell
python -m scripts.pipeline.chance_constrained.build_inputs `
  --config configs/chance_constrained/pyatka.example.json `
  --output-dir results/chance_constrained/pyatka
```

The command writes `scene_edges.npz` and `manifest.json`. Relative source paths
are resolved from the configuration file. Existing artifacts are not
overwritten unless `--force` is supplied.

The NPZ stores a flat directed-edge axis. Arrays that depend on operating
conditions use `edge, scenario, altitude, speed` axis order. A finite
`edge_goal_entry_fraction` identifies the first continuous intersection with
the circular delivery zone; `edge_goal_entry_time_s` contains the corresponding
partial traversal time.

`edge_mean_visibility` is an environmental input, not a VML success
probability. `edge_source_window_px` only identifies the high-resolution
orthophoto/DEM source window required by later sensor models. Future artifacts
must keep INS/OpenVINS propagation, VML outcomes, and TAN outcomes separate and
combine them through a conditional edge transition that depends on the incoming
navigation belief state.

The INS/OpenVINS increment lives in `propagation/`. It consumes synchronized
OpenVINS/EuRoC replay logs, creates relative 2/5/10-second error segments, fits
conditional process-noise and mode-transition models, and writes
`ins_openvins_model.npz`, `segments.csv`, `summary.json`, and `manifest.json`.
No measured EuRoC artifact is committed to the repository; creating its numeric
contents requires the Ubuntu replay procedure documented in
`propagation/ubuntu/README.md`.

The outdoor MUN-FRL preparation workflow is in
`notebooks/vins_dataset_generation.ipynb`. It runs in Google Colab, keeps only
the nadir monochrome camera, IMU, synchronization, and offline ground-truth
streams, and writes a compact ROS 2 Jazzy bag. LiDAR is excluded. The resulting
data supports normal VIO and post-initialization camera-blackout propagation;
it does not define a separate standalone INS model. The downstream replay
runner and log adapter are still EuRoC-specific and must be adapted to the
selected MUN-FRL topics and ground-truth representation before model fitting.

If the circle is smaller than the grid geometry and intersects no edge, input
preparation fails explicitly. The operator must then increase the justified
delivery radius or reduce planner-cell size rather than silently planning to a
different target.

The JSON schema version is `1` and requires these top-level objects:
`sources`, `grid`, `delivery_zone`, `vehicle`, `operating_points`, and
`scenarios`. Sensor and Monte Carlo objects are retained in the manifest as
explicit assumptions for later increments.
