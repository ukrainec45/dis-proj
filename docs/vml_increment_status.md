# VML increment status

Updated: 2026-10-22. Scope is governed by vml_implementation_plan.md.

## Completed milestone: real-data smoke data flow

Source: saved outputs of notebooks/vml_anyvisloc_smoke.ipynb, execution counts
17 (summary) and 19 (reject analysis), output directory anyvisloc-smoke-v3.
The user executed the notebook in hosted Colab; the agent read saved outputs.

Dataset: AnyVisLoc Scene_05; first 20 altitude-eligible frames in sorted order.
Prior: user-approved synthetic Gaussian XY prior, not recorded VIO estimates.
Each of the same 20 frames was evaluated against each reference map mode.

| Mode | Trials | Accepted | Rejects | Accepted position error |
| --- | ---: | ---: | ---: | --- |
| aerial | 20 | 4 | 16 | All 4 within 1 m according to retained summary |
| satellite | 20 | 0 | 20 | No accepted measurements |

Reasons retained by the result-only analysis:

- aerial: 4 accepted_xy_uncalibrated; 16 insufficient_map_coverage.
- satellite: 16 insufficient_map_coverage; 3 insufficient_inliers;
  1 insufficient_matches.

The native-resolution crop allocation failure is resolved. Reference images
now use the configured matching GSD before the matcher pixel-budget check,
with pixel-centre-aware affine transforms. Coverage and matcher thresholds
were not relaxed. Synthetic software verification currently passes 14 tests.

The smoke exercises the accepted XY branch on real aerial-reference data and
the rejection branch in both modes, source fingerprinting, Drive checkpoints,
manifest/summary export and result inspection. It does not prove satellite
feasibility or estimate flight probabilities. These 20 consecutive frames are
not a representative cross-scene sample. Sixteen trials per mode did not reach
matching because the existing 95% crop coverage policy rejected them.

## Not completed

- Full inventory and geometry/relief eligibility manifest.
- Deterministic maximin/medoid pilot scene selection (three scenes).
- Training-fold matcher threshold tuning and held-out matcher comparison.
- Conditional outcome model, local/gross residual mixture, calibrated post-match
  R and yaw reliability gate. The smoke's R remains an uncalibrated XY
  approximation; yaw is unavailable.
- JSON/NPZ model export/loader, supported-domain and extrapolation handling.
- Five-fold GroupKFold validation and bootstrap/calibration/coverage reports.
- SP-LightGlue-GIM evaluation if SIFT fails the validation gate. Zero accepted
  satellite smoke trials is not itself that held-out validation gate.
- Full campaign, stress sensitivity report, planner opportunity generator and
  conservative bound integration.

validation_passed=false for both modes. There is no trained model; current
artifacts remain not_fitted_smoke_only. The plan's production_enabled policy
for the best trained artifact is unchanged. target_platform_calibrated=false.

## Next work and data dependency

Build full inventory/eligibility before pilot scene selection and do not tune
thresholds to make this smoke succeed. Only Scene_05 is currently known to be
available. Pilot needs three scenes, while the planned validation requires at
least five eligible scenes per map mode. Inventory all available scenes before
selecting pilot scenes by the agreed feature-based algorithm. Additional scene
archives and space-efficient sequential staging require the user's data paths.

The user copies source/config files to Drive manually. Do not reintroduce an
automatic code upload/bootstrap. Read available saved notebook outputs before
requesting copied results. Never execute the notebook or real-data campaign
locally.
