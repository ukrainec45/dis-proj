# Reverse working-scene diagnostic

Training: Scene_07. Evaluation: Scene_05. Target: accepted fix with horizontal error <=3 m.
The original source split labels are unchanged; the run identity records the reversed direction.
Same predictors, L2 strength, target and optimizer as the forward experiment. Scaling uses Scene_07 only.
This diagnostic was proposed after viewing forward results; neither working scene is an untouched final test.
No candidate selection, pooled refit, planner integration or Scene_21 evaluation is performed.

| Model | Mean probability | Observed success | Log loss | Brier | AUC | Outside training ranges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 0.760766 | 0.232529 | 1.161303 | 0.457493 | 0.500000 | 0 |
| geometry | 0.642653 | 0.232529 | 1.111946 | 0.332447 | 0.798158 | 537 |
| geometry_and_map_features | 0.930306 | 0.232529 | 2.755488 | 0.660689 | 0.671588 | 559 |

Lower log loss and Brier are better. AUC measures ordering, not probability calibration.
Compare both directions separately; nearby frames are correlated and scene sizes differ.
A direction-dependent result can diagnose poor transfer but does not identify a single causal mechanism.
The constant has no features, so zero range warnings do not demonstrate support.
