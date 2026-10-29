# AnyVisLoc data audit and coverage-control review

Date: 2026-10-28. Sources: downloaded audit and coverage-control artifacts. No real-data campaign was executed locally.

## Main finding

All 16 original Scene_05 aerial coverage rejects were accepted by unchanged SIFT thresholds when only the early 95% coverage gate was bypassed. Errors against retained dataset XY ground truth range from 0.0401 to 0.6444 m (median 0.1440 m; mean 0.1959 m; RMSE 0.2535 m).
This establishes that the coverage gate suppressed correct-to-within-1-m fixes for these 16 selected frames. It does not establish a flight success probability, calibrated measurement covariance, or unconditional safety of clipped search.

## Verified data and frame lists

| Scene | Role | Frames read | Geometry candidates | Strict policy passed | Coverage gate rejected |
| --- | --- | ---: | ---: | ---: | ---: |
| Scene_05 | working | 1118 | 787 | 147 | 640 |
| Scene_07 | working | 1219 | 209 | 205 | 4 |
| Scene_21 | holdout | 1161 | 1082 | 978 | 104 |

All 3,498 frame records are readable according to the retained audit. There are 2,078 provisional geometry candidates, of which 996 belong to working scenes (787 Scene_05 + 209 Scene_07). The current strict policy permits 352 working frames (147 + 205). Scene_21 contributes 1,082 candidates but remains held out from matching.
Frame identifiers and coverage values are exported in candidate_frames.csv; working_candidate_frames.csv excludes Scene_21; strict_working_frames.csv additionally applies the unchanged 95% policy. These are capability/data lists, not lists selected for SIFT success. Except for the initial 20 Scene_05 frames, matching performance is unmeasured.

## First blocking reasons

| Scene | AGL below 20 m | AGL above 120 m | Outside DSM | Rectified image budget | Projection crosses horizon |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scene_05 | 0 | 248 | 78 | 5 | 0 |
| Scene_07 | 0 | 1006 | 3 | 0 | 1 |
| Scene_21 | 1 | 67 | 2 | 4 | 5 |

- AGL outside 20-120 m is a protocol/domain exclusion, not evidence of poor texture or an unreadable image. All 1,006 altitude exclusions in Scene_07 are above 120 m; the scene is not generally unusable. Do not widen the altitude domain after observing outcomes without a separate protocol decision.
- Outside DSM means the required local surface elevation cannot be sampled at GT XY, so AGL is unavailable under the current method. It does not establish missing camera imagery or a coordinate error.
- Rectified image budget means the projected image exceeds the configured pixel limit. This is a geometry/resource restriction, not a SIFT failure.
- Projection crosses horizon means at least one corner ray does not intersect the assumed ground plane in the supported forward direction. It is an unsupported planar projection.
- These counts are first blocking reasons: later processing was not performed on excluded frames. Factor counts for supported frames can overlap and must not be added as independent failures.

## Why the original 16 coverage rejects occurred

- 8 frames have a fully contained footprint within numerical tolerance; 8 have a partly clipped footprint. Even the latter produced sub-metre errors in this selected sample.
- 15 frames lack 95% search-rectangle coverage even when the prior mean is moved to GT with unchanged covariance. The map margin/search extent is a contributor independent of the sampled prior displacement.
- L05_0019 has full GT-centred search coverage, but its displaced prior reduces coverage to 93.16%. This is the sole prior-displacement-only coverage rejection in these 16 frames.
- Prior displacement reduces coverage in 8 frames; this overlaps the preceding categories. It can also improve coverage in other frames.
- Actual search-rectangle coverage ranges from 60.36% to 94.92%, while actual footprint coverage ranges from 82.61% to effectively 100%. Thus the demonstrated issue is not confined to tiny rounding differences around 95%.
- SIFT accepted 33-108 inliers per control trial, with inlier ratios 0.589-0.956. These diagnostics and XY errors are retained in control_cases.csv and original comparison.csv.

## Important map-content limitation

Visual inspection of all three map-inspection.png previews shows large uniform black regions around irregular aerial-image boundaries. Their shape is consistent with padding outside the imaged area; a source-provided valid-data mask is not available in these artifacts.
The current coverage calculation measures intersection with the rectangular raster extent. It does not exclude internal black padding. Sampled near-black fractions are approximately 17.29%, 26.85%, and 37.63% for Scene_05, Scene_07, and Scene_21 respectively. These global thumbnail fractions are not per-frame valid-image coverage and must not be subtracted from the reported coverage values.
Consequently, geometry_supported and usable_frames mean provisional support under current raster-boundary checks, not verified overlap with all usable map pixels. A future map-validity rule needs source metadata or a separately inspected mask; do not infer general nodata from dark pixels alone.

## Integrity and provenance

All checks listed in verification.json passed, including frozen scene/frame hashes, 16 per-trial envelopes and identities, CSV membership, saved priors, and XY errors recomputed from retained z and GT. Audit and control configuration fingerprints agree.
Original ZIP/image/DSM/NPZ files were not downloaded and were not independently rehashed here. Their validation is evidence retained from Colab, not a second local source-data verification. Original baseline acceptance for four other frames is not a new matcher rerun.
The old report_uk.md is a retained legacy input. This English report and all derived review artifacts are separate; downloaded evidence and frozen scene roles were not modified.

## Next decision

The 95% rectangle gate is not a justified proxy for matchability on the tested frames. Agree on a revised coverage/search policy before any broader matcher run; evaluate it on a predefined selection from working Scene_05/07 candidates, retaining all rejects and accepted XY errors.
Keep SIFT thresholds, synthetic-prior protocol, AGL range and raster scale fixed for a policy comparison. Keep Scene_21 matching results unobserved until the policy is fixed. No planner changes, probability-model fitting, or threshold tuning are supported by this selected 16-frame experiment alone.
No additional scene download is required to complete the present data-audit objective.
