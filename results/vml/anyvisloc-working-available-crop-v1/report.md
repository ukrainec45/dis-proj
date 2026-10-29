# Working-scene matching without a coverage threshold

Only the early 95% raster-extent coverage gate is disabled. Map coverage remains diagnostic.
SIFT thresholds, priors, AGL domain, scale and geometry/resource limits are unchanged.
The legacy policy column is a counterfactual applied to these same outcomes, not a second matcher run.
Accepted errors are relative to retained dataset GT; acceptance is not proof of correctness.

Campaign status: completed; completed 996/996 trials.

| Scene | Status | Completed/expected | Accepted | Accepted with error >1 m | Accepted with error >5 m |
| --- | --- | ---: | ---: | ---: | ---: |
| Scene_05 | completed | 787/787 | 231 | 85 | 34 |
| Scene_07 | completed | 209/209 | 164 | 17 | 3 |

## Aggregate results (completed trials only)

Available crop: {'trial_count': 996, 'accepted_count': 395, 'reject_count': 601, 'acc_at_m': {'1': 0.29417670682730923, '3': 0.3433734939759036, '5': 0.35943775100401604, '10': 0.3755020080321285, '20': 0.38052208835341367}, 'accepted_error_m': {'min': 0.010949173425593216, 'median': 0.3185984465543652, 'mean': 6.058419644206704, 'max': 432.3123990139752, 'rmse': 37.835572112201845}, 'accepted_error_over_m': {'1': 102, '3': 53, '5': 37, '10': 21, '20': 16}}

Legacy 95% gate counterfactual: {'trial_count': 996, 'accepted_count': 241, 'reject_count': 755, 'acc_at_m': {'1': 0.21987951807228914, '3': 0.23493975903614459, '5': 0.23795180722891565, '10': 0.24096385542168675, '20': 0.24096385542168675}, 'accepted_error_m': {'min': 0.010949173425593216, 'median': 0.2315335835328543, 'mean': 0.5368133324056343, 'max': 20.339394912772597, 'rmse': 1.611867125143084}, 'accepted_error_over_m': {'1': 22, '3': 7, '5': 4, '10': 1, '20': 1}}

Error cutoffs are descriptive accuracy levels, not fitted local/gross labels.
Both denominators include all completed candidate attempts, including rejects. Incomplete results are provisional.
Frame selection is fixed by the audit before matching. Excluded-domain frames are documented in the audit.
Black padding inside the raster is not removed or automatically treated as nodata. This limitation is unchanged.
Scene roles and original audit/control results are unchanged. No holdout matching or model fitting is performed.
