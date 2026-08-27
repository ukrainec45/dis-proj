# Chance-constrained input pipeline

This package implements the first preprocessing increment for the
chance-constrained planner. It creates a versioned scene and directed-edge
artifact; it deliberately does **not** estimate OpenVINS propagation,
absolute visual map localization (VML), TAN, or route-success outcome models
yet.

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

If the circle is smaller than the grid geometry and intersects no edge, input
preparation fails explicitly. The operator must then increase the justified
delivery radius or reduce planner-cell size rather than silently planning to a
different target.

The JSON schema version is `1` and requires these top-level objects:
`sources`, `grid`, `delivery_zone`, `vehicle`, `operating_points`, and
`scenarios`. Sensor and Monte Carlo objects are retained in the manifest as
explicit assumptions for later increments.
