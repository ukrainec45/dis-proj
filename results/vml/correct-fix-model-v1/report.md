# Correct-fix probability model training

Target: matcher accepts a horizontal fix with dataset-relative error <=3 m.
Training: Scene_05. Validation: Scene_07. Scene_21 has not been read or matched.
Fixed L2 strength 1.0, summed logistic loss, unpenalized intercept, training-only standardization.
Lower log loss and Brier score are better. AUC measures ranking, not calibration.

| Model | Validation log loss | Brier score | ROC AUC | Out-of-range validation rows |
| --- | ---: | ---: | ---: | ---: |
| constant | 1.173075 | 0.461036 | 0.500000 | 0 |
| geometry | 1.739872 | 0.564702 | 0.816352 | 55 |
| geometry_and_map_features | 2.334588 | 0.598107 | 0.872956 | 153 |

Selected by validation log loss: constant. All three fixed candidates are refitted on both working scenes and frozen for final comparison.
Observed map-feature log-loss gain over geometry: -0.594716. This is a point comparison on one validation scene, not a statistical significance claim.

Inspect calibration.png and calibration.csv: predicted probabilities should agree with observed frequencies. No automatic calibration-success certificate is inferred from small bins.
Feature extraction uses supplied camera geometry and the reference map; actual query feature counts and all post-match diagnostics are excluded.
Covariance is fixed at sigma=20 m per XY axis. Predicting a growing uncertainty along a route requires additional training conditions.
This model can become an input to correction-opportunity evaluation after independent testing and state-transition integration. It is not connected to the planner and does not certify route safety.
Final Scene_21 evaluation remains pending. Do not tune candidates using that scene.
Model exports use JSON coefficients and standardization parameters, without pickle or a scikit-learn dependency.
