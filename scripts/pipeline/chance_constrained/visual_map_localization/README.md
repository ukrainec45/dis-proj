# Synthetic VML feasibility experiment

This package evaluates a concrete camera and matcher on spatially distributed
directed edges. It intentionally renders the query from the same orthophoto
that is used as the reference map. Therefore its outputs support software
verification, matcher-policy tuning, and provisional spatial ranking only;
they are not measured real-flight probabilities.

Build the sensor-neutral Pyatka scene first:

```powershell
python -m scripts.pipeline.chance_constrained.build_inputs `
  --config configs/chance_constrained/pyatka.example.json `
  --output-dir results/chance_constrained/pyatka
```

Run a two-trial smoke check:

```powershell
python -m scripts.pipeline.chance_constrained.visual_map_localization.run_experiment `
  --config configs/chance_constrained/vml/pyatka_rpi_gs_sift.json `
  --output-dir results/chance_constrained/pyatka/vml-rpi-gs-sift-smoke `
  --smoke
```

Run the configured campaign:

```powershell
python -m scripts.pipeline.chance_constrained.visual_map_localization.run_experiment `
  --config configs/chance_constrained/vml/pyatka_rpi_gs_sift.json `
  --output-dir results/chance_constrained/pyatka/vml-rpi-gs-sift-v1
```

For a bounded first pilot (96 trials), use
`configs/chance_constrained/vml/pyatka_rpi_gs_sift_pilot.json`. The larger
campaign is deliberately separate because its uncached SIFT reference
extraction is an offline calibration workload, not a planner-time operation.

The runner writes per-trial `trials.csv`, aggregated
`vml_sample_outcomes.npz`, a source-fingerprinted `manifest.json`, a summary,
and a spatial diagnostic plot. The NPZ axes are
`sampled_edge, scenario, altitude, search_radius`. Yaw and repeat trials are
pooled inside each cell.

Rebuild the descriptive breakdown from retained trials without rerunning SIFT:

```powershell
python -m scripts.pipeline.chance_constrained.visual_map_localization.analyze_results `
  --trials results/chance_constrained/pyatka/vml-rpi-gs-sift-pilot-v1/trials.csv `
  --output results/chance_constrained/pyatka/vml-rpi-gs-sift-pilot-v1/analysis.json
```

The next increment must fit and spatially validate a conditional outcome model
before producing a planner-consumable table for every edge. It must not copy a
sample probability blindly to unsampled edges.
