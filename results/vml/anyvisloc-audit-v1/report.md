# AnyVisLoc audit: SIFT and aerial reference maps

The prior is a synthetic XY position estimate, not recorded VIO output.
95% is an engineering threshold on search rectangle area, not localization success probability.
Footprint coverage uses polygon intersection; search rectangles retain the original pixel rounding.
Near-black/white regions are inspection indicators, not established missing data.

## Scenes

- Scene_05: partially_usable; role working; frames 1118, geometry supported 787, strict policy passed 147. Shared errors: []; factors: {'agl_outside_20_120_m': 248, 'dsm_outside': 78, 'footprint_outside_map': 559, 'insufficient_map_margin_at_gt': 636, 'prior_shift_reduces_coverage': 347, 'rectified_size_unsupported': 5}.
- Scene_07: partially_usable; role working; frames 1219, geometry supported 209, strict policy passed 205. Shared errors: []; factors: {'agl_outside_20_120_m': 1006, 'dsm_outside': 3, 'footprint_outside_map': 4, 'insufficient_map_margin_at_gt': 6, 'prior_shift_reduces_coverage': 2, 'projection_crosses_horizon': 1}.
- Scene_21: partially_usable; role holdout; frames 1161, geometry supported 1082, strict policy passed 978. Shared errors: []; factors: {'agl_outside_20_120_m': 68, 'dsm_outside': 2, 'footprint_outside_map': 51, 'insufficient_map_margin_at_gt': 109, 'prior_shift_reduces_coverage': 64, 'projection_crosses_horizon': 5, 'rectified_size_unsupported': 4}.

## Baseline frames: explanations or unresolved evidence

- Scene_05/L05_0001.npz: accepted_xy_uncalibrated; coverage {'footprint_coverage': 1.0000000000000002, 'gt_search_coverage': 0.926233150644349, 'prior_search_coverage': 0.9539395734597157}; factors ['insufficient_map_margin_at_gt']; error None; baseline comparability verified: True.
- Scene_05/L05_0002.npz: insufficient_map_coverage; coverage {'footprint_coverage': 0.9999999999999999, 'gt_search_coverage': 0.886122569239835, 'prior_search_coverage': 0.9491750147318798}; factors ['insufficient_map_margin_at_gt']; error None; baseline comparability verified: True.
- Scene_05/L05_0003.npz: insufficient_map_coverage; coverage {'footprint_coverage': 1.0, 'gt_search_coverage': 0.8175085172567027, 'prior_search_coverage': 0.723300251814546}; factors ['insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; error None; baseline comparability verified: True.
- Scene_05/L05_0004.npz: insufficient_map_coverage; coverage {'footprint_coverage': 0.9769224215612684, 'gt_search_coverage': 0.7458481613285883, 'prior_search_coverage': 0.7648279952550415}; factors ['footprint_outside_map', 'insufficient_map_margin_at_gt']; error None; baseline comparability verified: True.
- Scene_05/L05_0005.npz: insufficient_map_coverage; coverage {'footprint_coverage': 0.9323757069550103, 'gt_search_coverage': 0.6902191171800571, 'prior_search_coverage': 0.6341695776436964}; factors ['footprint_outside_map', 'insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; error None; baseline comparability verified: True.
- Scene_05/L05_0006.npz: insufficient_map_coverage; coverage {'footprint_coverage': 0.826075989280576, 'gt_search_coverage': 0.5619986746189529, 'prior_search_coverage': 0.6035632223751508}; factors ['footprint_outside_map', 'insufficient_map_margin_at_gt']; error None; baseline comparability verified: True.
- Scene_05/L05_0007.npz: insufficient_map_coverage; coverage {'footprint_coverage': 0.9329449119012222, 'gt_search_coverage': 0.6235836907394308, 'prior_search_coverage': 0.6384160575235563}; factors ['footprint_outside_map', 'insufficient_map_margin_at_gt']; error None; baseline comparability verified: True.
- Scene_05/L05_0008.npz: insufficient_map_coverage; coverage {'footprint_coverage': 0.9318897133139924, 'gt_search_coverage': 0.6743568009747902, 'prior_search_coverage': 0.6641467651384617}; factors ['footprint_outside_map', 'insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; error None; baseline comparability verified: True.
- Scene_05/L05_0009.npz: insufficient_map_coverage; coverage {'footprint_coverage': 0.9819911108205231, 'gt_search_coverage': 0.7741263225392754, 'prior_search_coverage': 0.714972747675537}; factors ['footprint_outside_map', 'insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; error None; baseline comparability verified: True.
- Scene_05/L05_0010.npz: insufficient_map_coverage; coverage {'footprint_coverage': 0.9999923203146052, 'gt_search_coverage': 0.7887323943661971, 'prior_search_coverage': 0.7491300745650373}; factors ['footprint_outside_map', 'insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; error None; baseline comparability verified: True.
- Scene_05/L05_0011.npz: insufficient_map_coverage; coverage {'footprint_coverage': 0.9763269313178631, 'gt_search_coverage': 0.7782891604507222, 'prior_search_coverage': 0.7827328995397556}; factors ['footprint_outside_map', 'insufficient_map_margin_at_gt']; error None; baseline comparability verified: True.
- Scene_05/L05_0012.npz: insufficient_map_coverage; coverage {'footprint_coverage': 1.0, 'gt_search_coverage': 0.8018363939899833, 'prior_search_coverage': 0.7667779632721202}; factors ['insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; error None; baseline comparability verified: True.
- Scene_05/L05_0013.npz: insufficient_map_coverage; coverage {'footprint_coverage': 1.0, 'gt_search_coverage': 0.7955132343447385, 'prior_search_coverage': 0.8000322788896062}; factors ['insufficient_map_margin_at_gt']; error None; baseline comparability verified: True.
- Scene_05/L05_0014.npz: insufficient_map_coverage; coverage {'footprint_coverage': 1.0, 'gt_search_coverage': 0.7977239942298445, 'prior_search_coverage': 0.8052572527648661}; factors ['insufficient_map_margin_at_gt']; error None; baseline comparability verified: True.
- Scene_05/L05_0015.npz: insufficient_map_coverage; coverage {'footprint_coverage': 0.9999999999999998, 'gt_search_coverage': 0.8469652710159039, 'prior_search_coverage': 0.7693930542031808}; factors ['insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; error None; baseline comparability verified: True.
- Scene_05/L05_0016.npz: insufficient_map_coverage; coverage {'footprint_coverage': 1.0000000000000002, 'gt_search_coverage': 0.9186526850969177, 'prior_search_coverage': 0.9407372100413092}; factors ['insufficient_map_margin_at_gt']; error None; baseline comparability verified: True.
- Scene_05/L05_0017.npz: accepted_xy_uncalibrated; coverage {'footprint_coverage': 1.0, 'gt_search_coverage': 1.0, 'prior_search_coverage': 1.0}; factors []; error None; baseline comparability verified: True.
- Scene_05/L05_0018.npz: accepted_xy_uncalibrated; coverage {'footprint_coverage': 1.0, 'gt_search_coverage': 1.0, 'prior_search_coverage': 0.9996683800364782}; factors ['prior_shift_reduces_coverage']; error None; baseline comparability verified: True.
- Scene_05/L05_0019.npz: insufficient_map_coverage; coverage {'footprint_coverage': 0.9999999999999998, 'gt_search_coverage': 1.0, 'prior_search_coverage': 0.9315687540348612}; factors ['prior_shift_reduces_coverage']; error None; baseline comparability verified: True.
- Scene_05/L05_0020.npz: accepted_xy_uncalibrated; coverage {'footprint_coverage': 0.9999999999999999, 'gt_search_coverage': 0.9987152722017023, 'prior_search_coverage': 1.0}; factors []; error None; baseline comparability verified: True.

## Interpretation

Footprint outside the map means part of the actual visible ground is unavailable. Insufficient margin at GT means the map cannot support the configured search uncertainty. Reduced coverage at the prior shows the contribution of its particular displacement. Factors may coexist.
Read/coordinate errors retain their exception details and require inspection; they do not establish poor texture as the cause.
Eligibility refers to the current protocol; it does not guarantee SIFT success or validity of the planar approximation.
Per-frame details: scenes/<scene>/frames.csv; geometry figures: geometry-*.png.
The control experiment is stored separately and does not change the production localization policy.
