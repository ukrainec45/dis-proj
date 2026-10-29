# Both working-scene directions reviewed

The reverse hosted run completed. All model and dataset hashes match; 2361 predictions, labels, range warnings and metrics reproduce without fitting. Scaling uses Scene_07 only. Both runs use identical input-table hashes.

| Direction | Candidate | Mean predicted probability | Observed success | Log loss | Brier | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 05 to 07 | constant | 0.232529 | 0.760766 | 1.173075 | 0.461036 | 0.500000 |
| 05 to 07 | geometry | 0.125413 | 0.760766 | 1.739872 | 0.564702 | 0.816352 |
| 05 to 07 | geometry_and_map_features | 0.103449 | 0.760766 | 2.334588 | 0.598107 | 0.872956 |
| 07 to 05 | constant | 0.760766 | 0.232529 | 1.161303 | 0.457493 | 0.500000 |
| 07 to 05 | geometry | 0.642653 | 0.232529 | 1.111946 | 0.332447 | 0.798158 |
| 07 to 05 | geometry_and_map_features | 0.930306 | 0.232529 | 2.755488 | 0.660689 | 0.671588 |

Reverse geometry improves log loss over the constant by 0.04936 and Brier by 0.12505, but predicts 64.27% against 23.25% observed. It remains substantially overconfident. Geometry+map predicts 93.03%, has worse log loss by 1.64354 than geometry, and lower AUC. The current added map descriptor worsens probability quality in both directions; a general claim that map imagery is uninformative is not justified.

For 714 reverse geometry+map predictions above or equal to 0.8, mean probability is 96.28% and observed success is 25.07%. Geometry range warnings affect 537/787 frames; adding the map feature raises that count to 559/787. Range checks do not certify support for the remaining frames.

This is additional development evidence from the same two correlated-frame scenes, proposed after forward results were seen. It is not an independent replication or final test. Do not pool directional metrics into a scene-general performance estimate. No local real-data fitting, new matching, holdout evaluation or planner integration was performed.

Recommended next development step: a fixed-scale, fixed-area map-descriptor protocol with explicit valid-image handling and detector-cap sensitivity checks, evaluated in both working-scene directions. More independent development scenes would be needed to assess broad transfer and fit reliable probabilities. Keep Scene_21 unused while designing this revision.
