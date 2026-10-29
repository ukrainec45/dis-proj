# First correct-fix model review

## Outcome

The hosted training run completed and its numerical exports are internally consistent. Probability transfer from Scene_05 to Scene_07 is poor for all three candidates. The map-feature hypothesis is not supported for the primary probability metric on this validation scene. No trained candidate is approved for planner integration.

Target: matcher accepted AND horizontal error <=3 m. Scene_05 training: 183/787 successes (23.25%). Scene_07 validation: 159/209 (76.08%). Scene_21 has not been used for matching or model evaluation by this review.

| Candidate | Mean predicted probability | Validation log loss (lower better) | Brier (lower better) | AUC (higher better) |
|---|---:|---:|---:|---:|
| Constant | 0.2325 | 1.1731 | 0.4610 | 0.5000 |
| Geometry | 0.1254 | 1.7399 | 0.5647 | 0.8164 |
| Geometry + map feature | 0.1034 | 2.3346 | 0.5981 | 0.8730 |

The constant is selected by the predeclared lowest validation log loss. This means least poor among these candidates, not sufficient calibration. After the working-scene refit the exported constant is 342/996=34.34%; that refit has no independent evaluation yet. Zero range warnings for a featureless constant are vacuous, not evidence of domain support.

AUC measures ordering, not accuracy of numerical probabilities. Adding the map feature improves ordering but worsens log loss by 0.5947. In the lowest-probability bin, the geometry+map model predicts 3.26% on average for 168 frames, while 70.24% actually succeed. This is substantial underprediction.

## Evidence and limits

Model and dataset hashes match. All 627 saved validation predictions (three candidates x 209 frames), labels, range warnings and primary metrics reproduce without fitting. Training-only standardization means and penalized objectives also reproduce. See verification.json. No real-data training, source-image matching or holdout evaluation was performed locally.

Features differ between working scenes. Geometry range warnings affect 55/209 validation frames; geometry+map warnings affect 153/209. The map-density feature alone is outside its training min/max for 114/209 frames. Range checks are coarse and do not prove that frames inside ranges are represented by the training distribution.

The map keypoint density is a limited proxy: detected features are capped at approximately 6000 and divided by crop area. Retained trials reach or exceed 6000 reference keypoints in 391/787 Scene_05 cases and 13/209 Scene_07 cases. Median reference counts are 5963 and 2193, and median crop areas are 625170 and 333450 pixels, respectively. Thus the feature mixes image content, crop geometry and detector limits. This is a plausible source of poor transfer, not a demonstrated sole cause. The geometry-only model also fails probability calibration. Different success rates are established; their causal decomposition is unresolved.

Only one validation scene is available, and nearby frames are correlated. No scene-level generalization significance or field success probability is established. Dataset ground truth and ideal attitude/AGL inputs further limit deployment claims.

## Next experiment recommendation

Preserve this result as the first fixed protocol, including its negative probability finding. Keep Scene_21 unused during model development. Prepare a hosted-Colab reverse-direction diagnostic (Scene_07 training, Scene_05 evaluation) for the same fixed candidates, with both directions reported and no claim that the reverse evaluation is a new untouched test.

Before choosing a more complex model, assess the existing feature limitations and specify an additional map descriptor measured at fixed scale on fixed-area tiles, with valid-image coverage handled explicitly. Measure its sensitivity to the detector cap. Such a descriptor needs source map imagery and a new declared extraction protocol; it cannot be manufactured from the current capped counts. If development continues, freeze the revised protocol before final Scene_21 evaluation. Further independent scenes are needed for stronger article claims.

The eventual planning role remains a conditional estimate of a correct visual correction at a candidate location. Current probabilities cannot yet support that role, do not separate rejection from inaccurate acceptance, and do not model position uncertainty after correction or arrival risk along a route.

## Why the training scene has few successes

The 23.25% is the retained SIFT experiment's outcome rate, not the probability model's training accuracy. Scene_05 has 272 insufficient-match rejects, 284 insufficient-inlier rejects, 48 accepted errors over 3 m, and 183 accepted fixes within 3 m. An inlier is a correspondence compatible with the estimated geometric transformation; rejection can follow too few inliers or too low their proportion.

At the ground-truth position, 558/787 camera footprints extend outside the map raster rectangle, versus 4/209 in Scene_07. Scene_05 success is 111/229 (48.47%) with a footprint inside the rectangle and 72/558 (12.90%) otherwise. Scene_07 has 158/205 (77.07%) and 1/4 (25.00%), respectively. These are descriptive associations, not causal effects. Ground-truth footprint coverage is used only for diagnosis and is not a model predictor. Internal padding is not measured by this rectangle test.

Even for footprints inside the rectangle, Scene_05 succeeds less often. Missing overlap therefore cannot explain the whole gap. Image content, viewpoint, map padding and planar approximation remain candidate mechanisms requiring source-image inspection or controlled experiments. No current reject is caused by the disabled 95% gate. The geometry-candidate sets also select different portions of each scene under the AGL constraints, so these are not full-scene representative success rates.
