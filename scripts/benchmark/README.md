# Pre-flight planning benchmark

Run the complete deterministic suite and retain its versioned artifacts:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark.run_benchmark --run-id synthetic-v1
```

Run selected built-in cases and a prepared real AOI archive:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark.run_benchmark `
  --run-id zhytomyr-v1 `
  --scenario realistic `
  --scenario foggy_valley `
  --map zhytomyr=path\to\planning_layers.npz
```

Results are written to `results/benchmarks/<run-id>/`:

- `manifest.json` records methods, weights, decision profiles, platform data,
  and SHA-256 checksums for supplied NPZ archives.
- `summary.csv` has one comparable row per scenario and planner.
- `runs/<scenario>/<method>.json` retains every path, objective vector, and
  TOPSIS selection.
- `figures/` contains route-overlay and objective-space comparisons.

The compared planners use the same `CostMap` and hard constraints. EMOA* is
evaluated against time-only A* and repeated weighted A* over 15 fixed weight
vectors whose components are multiples of 0.25 and sum to one.

## Thesis campaign

Run the controlled 30-variant synthetic campaign (four conflict-rich families,
two vehicle profiles, and five weather profiles):

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark.run_campaign --run-id thesis-synthetic-v1
```

To add the exactly five real AOIs, copy and complete
`aoi_manifest.example.json`, build each NPZ with the pipeline, and run:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark.run_campaign `
  --run-id thesis-real-v1 `
  --aoi-manifest my_aois.json `
  --include-navigation-ablations
```

The campaign fails before large experiments if its tiny exhaustive-oracle suite
does not exactly reproduce the EMOA* front. It also writes aggregate tables,
bootstrap confidence intervals, and paired EMOA*/weighted-A* Wilcoxon results
to `analysis/`.
