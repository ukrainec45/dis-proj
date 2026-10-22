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

## AnyVisLoc increment: stage 1 / smoke

The separate `anyvisloc_smoke` runner implements the first stage of
`docs/vml_implementation_plan.md`. The synthetic Pyatka experiment above is
retained with its original conventions and results.

Open `notebooks/vml_anyvisloc_smoke.ipynb` in a **hosted Google Colab runtime**.
The first executable cell refuses execution outside Colab. Sign in and mount
Drive. Upload Scene_05.zip to your own Drive and set SCENE_ARCHIVE to its path;
the example is /content/drive/MyDrive/AnyVisLoc/Scene_05.zip. Existing repository
and result paths are retained. Copy the current repository code/config to Drive.

After repeated OneDrive download failures, the user chose Drive archive storage.
The notebook reads the ZIP directly from mounted Drive, checks SHA-256 and free
runtime disk space, then extracts into /content/AnyVisLoc. It does not create a
second ZIP copy in the runtime or delete the Drive archive. Only extracted data
is ephemeral; results, checkpoints and dataset_staging_receipt.json stay in
Drive. After VM replacement, remount Drive and extract the same archive again.
The old URL/cURL download cells have been removed from this workflow.

Real-data campaigns and the notebook must not run locally. Local pytest tests
use generated fixtures only.

The smoke selects the first 20 distinct AGL-eligible frames deterministically
in sorted scene/sample order, then performs 20 trials for each map mode. This
is a bounded data-flow check; maximin/medoid selection belongs to the pilot.
The inventory explicitly reports altitude eligibility, not full geometric
eligibility; unsupported geometry remains visible as reject reasons.

Decisions and provisional settings:

- The user approved `synthetic_gaussian_from_gt` position priors for smoke.
  The example uses 20 m standard deviation per XY axis, zero correlation and
  seed 42. These are configurable experiment settings, not fitted flight data.
  Each frame gets the same prior in both modes, independent of execution order.
- AGL uses the aerial DSM and GT camera location for both modes so that the
  physical frame altitude is consistent. Missing/outside DSM is excluded;
  AGL must be in [20,120] m. Attitude is dataset metadata, with no noise added.
- Dataset XY is East/South. Outputs and map transforms use East/North. The
  reference DSM origin is relative to the map origin, as in upstream code.
  `pose_world_frame` is explicit. The default `anyvisloc_ned_attitude`
  addresses Scene 5 metadata observed in the saved notebook: pose translation
  equals raw dataset xyz, while rotation equals Rz(yaw) Ry(pitch) Rx(roll).
  Translation is converted East/South/Up -> ENU separately. Attitude uses
  NED->ENU on the left and OpenCV-camera->FRD on the right. Zero attitude is
  nadir, with image top North. This interpretation agrees with upstream's
  zero-roll pitch/yaw footprint equation; real-image alignment still needs the
  smoke check. Every sample must pass raw translation and Euler/matrix checks;
  there is no automatic frame guessing. Explicit legacy `enu` and `dataset`
  modes remain available for fixtures/other known camera pose formats.
  Translation tolerance is 5 cm with zero relative tolerance.
- The saved notebook's metadata output can be read directly in the shared
  workspace. Read it before asking the user to paste diagnostic output.
  The pose fix changes code/config fingerprints; notebook output defaults to
  anyvisloc-smoke-v2 to preserve the earlier failed run. Rerun path and import
  cells before the smoke cell; extraction need not be repeated. The import cell
  explicitly reloads the adapter/runner so cached Python modules are updated.
- Virtual-nadir rectification uses real K, distortion and camera-to-world
  attitude. The camera's ground projection, rather than image centre, is the
  localization anchor. Crops bound footprint plus the configured covariance
  ellipse; no fixed tilt cutoff is imposed. Numerical conditioning, image
  allocation and map coverage limits are explicit config settings. Distorted
  footprint bounds currently use four undistorted corner rays.
- The smoke always returns partial XY observations; yaw is unavailable until a
  reliability gate is implemented and validated. Its PSD geometric R uses a
  translation residual approximation plus a configured 0.5 m floor, and is
  explicitly `R_calibrated=false`. It is not the final post-match R model.
  Accepted residuals are retained without threshold-based local/gross labels.
- SIFT thresholds, rectification GSD and resource limits are provisional smoke
  settings. No thresholds are tuned on smoke, and no empirical probabilities
  or planner tables are exported.

Each recovery unit is `scene-reference × mode × sift × smoke`. Atomic per-trial
JSON envelopes contain a SHA-256 checksum and identity over code, config,
NumPy/OpenCV versions and source data checksums. Reconnect and rerun to reuse
verified trials and rebuild each unit's `trials.json` and `status.json`. A crash
after saving a trial does not require rerunning it. Corruption or changed inputs
raises an error; changed code/config/library versions require a new output dir.
Only one runtime should write a given output directory.

Drive outputs: `run.json`, bounded `inventory.json`, per-unit trial files,
checksummed shards/status, `summary.json` (Acc@1/3/5/10/20 with rejects in the
denominator), and `manifest.json`. Full file hashing includes selected samples,
external query images, reference JSON, maps and DSMs. Validation is false for
both modes. There is no trained artifact in smoke, so its status is
`not_fitted_smoke_only`; the plan's `production_enabled` decision for the best
**trained** artifact is unchanged. No target platform timing claim is made.

Next stages remain: full inventory and relief/parallax eligibility; pilot
selection; training-fold threshold tuning; conditional outcome and post-match R
models; JSON/NPZ export/loader; independent cross-scene validation; fallback
SP-LightGlue-GIM if needed; full campaign; planner opportunity scheduling.

Upstream format and coordinate references (checked 2026-10-22):
https://github.com/UAV-AVL/Benchmark/blob/main/avl_data.py
https://github.com/UAV-AVL/Benchmark/blob/main/avl_utils.py
https://github.com/UAV-AVL/Benchmark/blob/main/docs/GROUND_TRUTH.md

Software checks (synthetic fixtures; does not execute the notebook):

```powershell
.venv/Scripts/python.exe -m pytest scripts/pipeline/chance_constrained/visual_map_localization/test_anyvisloc_smoke.py scripts/pipeline/chance_constrained/visual_map_localization/test_vml.py -q -p no:cacheprovider --basetemp=misc/pytest_anyvisloc_review
```

Matching-grid correction after Scene 5 smoke: aerial had 20/20
search_size_unsupported rejects before SIFT; satellite had 16 coverage rejects,
2 insufficient-match rejects and 2 inlier rejects. The reference crop now uses
the configured rectified_gsd_m before applying the matcher pixel budget.
Resampling preserves OpenCV pixel-centre coordinates in the output transform.
The crop's physical extent, search mass and coverage/acceptance thresholds are
unchanged. The notebook reloads planar and smoke_localizer as well as the runner;
new results go to anyvisloc-smoke-v3. Files still require manual copying to Drive.
