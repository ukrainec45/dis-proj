# Working-scene experiment and modelling dataset

Source: retained results of anyvisloc-working-available-crop-v1. No real-data matching or model training was executed locally.

## Verified outcomes

| Scene | Attempts | Accepted within 3 m | Accepted over 3 m | Rejected |
| --- | ---: | ---: | ---: | ---: |
| Scene_05 | 787 | 183 | 48 | 556 |
| Scene_07 | 209 | 159 | 5 | 45 |
| Total | 996 | 342 | 53 | 601 |

Both scenes completed, with exactly the frozen 787 + 209 candidates and no Scene_21 matching. Frozen artifact hashes, all 996 trial envelopes, priors, XY errors, CSV exports and aggregate metrics were verified. Raw source images were not available locally for independent rehashing.

## Effect of removing the coverage gate

| Policy | Accepted | Within 3 m | Accepted over 3 m | Within-3-m fraction of all attempts | Over-3-m fraction of accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy 95% gate counterfactual | 241 | 234 | 7 | 23.49% | 2.90% |
| Available crop | 395 | 342 | 53 | 34.34% | 13.42% |

Removing the gate adds 154 accepted fixes: 108 within 3 m and 46 over 3 m. This is a measured trade-off, not an unconditional improvement. The legacy column is a counterfactual over the same trial results, not an independent second run.
The maximum accepted error is 432.31 m. There are 16 accepted errors over 20 m (versus 1 under the legacy counterfactual). The good outcomes on the original 16 coverage rejects did not represent all 644 below-threshold working frames.
An accepted-over-3-m label is an evaluation category, not a fitted gross-error mixture label. R remains uncalibrated. These results do not support using acceptance or small geometric residual covariance as a guarantee of correct navigation.

## Ready-to-fit dataset

features.csv contains 996 finite rows of predictors available before correspondence search. labels.csv contains matched outcomes and the fixed 3 m target. They join on sample. Training is Scene_05 (183 positive / 604 negative); validation is Scene_07 (159 positive / 50 negative). Both classes exist in both scenes.
The outcome breakdown preserves rejects separately from accepted inaccurate fixes: 342 accepted within 3 m, 53 accepted over 3 m, 601 rejects. Binary correct-fix prediction has 342 positives and 654 negatives, but evaluation must retain the three-way breakdown.
feature_spec.json defines explicit input allowlists for the constant baseline, geometry model, and geometry-plus-map-feature model, plus a separately identified online query-feature variant. Scene IDs, GT positions, actual prior error, GT-centred/actual-footprint coverage, correspondence counts, inliers and residuals are excluded as predictors.
Separate map and query keypoint-density descriptors are available from feature extraction before correspondence search, so no additional image processing or matcher rerun is needed to recover them. The map-only descriptor is suitable for a planner-oriented comparison at specified map crops; query descriptors require the actual camera frame and are only available online. They must not be presented as pre-flight inputs. Both are proxies affected by geometry, scale, padding and the fixed feature budget, not independent sharpness measures. Corresponding crop sizes are included in the geometry baseline to reduce denominator confounding.
The large base-rate difference between scenes (23.25% versus 76.08% success within 3 m) makes cross-scene validation demanding. Do not randomly split neighbouring frames and report that as cross-scene transfer.

## Next scientific step

Fit the predeclared constant baseline and regularized logistic models on Scene_05, compare probabilistic prediction on Scene_07, and record calibration and accepted-error strata. Do not use Scene_21 outcomes to select features, regularization or acceptance thresholds.
The current dataset supports a first model at synthetic prior sigma=20 m only. A dependence on changing prior uncertainty cannot be identified without a designed multi-level experiment.
A universal flight-probability claim, a calibrated R, full outcome mixture, and five-scene validation remain outside these data. Ground-truth accuracy and recommended 3/5/10 m criteria are documented by the AnyVisLoc authors: https://github.com/UAV-AVL/Benchmark/blob/main/docs/GROUND_TRUTH.md

## Outputs

- features.csv / labels.csv: numeric modelling data, with train/validation scene assignments.
- feature_spec.json: predictors, target, exclusions and limitations; training_status=not_fitted.
- legacy_gate_rejected_cases.csv: all 644 gate-rejected candidates with actual outcomes.
- largest_accepted_errors.csv: the 20 largest accepted errors with diagnostic values.
- repeated_control_comparison.csv: re-evaluation of the original 16 control frames.
- verification.json: retained-artifact validation and provenance.
