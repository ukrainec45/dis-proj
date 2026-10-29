# Аудит AnyVisLoc: SIFT + аерофотокарта

Prior — попередня оцінка XY з синтетичною похибкою; це не запис VIO.
95% — чинний інженерний поріг площі прямокутника, не ймовірність правильної прив’язки.
Footprint перевіряється як полігон; прямокутники — за чинним округленням до пікселів.
Чорні/білі ділянки — індикатори для огляду, не доведена відсутність даних.

## Сцени

- Scene_05: partially_usable; роль working; кадрів 1118, геометрично підтримуваних 787, проходять чинну політику 147. Спільні помилки: []; чинники: {'agl_outside_20_120_m': 248, 'dsm_outside': 78, 'footprint_outside_map': 559, 'insufficient_map_margin_at_gt': 636, 'prior_shift_reduces_coverage': 347, 'rectified_size_unsupported': 5}.
- Scene_07: partially_usable; роль working; кадрів 1219, геометрично підтримуваних 209, проходять чинну політику 205. Спільні помилки: []; чинники: {'agl_outside_20_120_m': 1006, 'dsm_outside': 3, 'footprint_outside_map': 4, 'insufficient_map_margin_at_gt': 6, 'prior_shift_reduces_coverage': 2, 'projection_crosses_horizon': 1}.
- Scene_21: partially_usable; роль holdout; кадрів 1161, геометрично підтримуваних 1082, проходять чинну політику 978. Спільні помилки: []; чинники: {'agl_outside_20_120_m': 68, 'dsm_outside': 2, 'footprint_outside_map': 51, 'insufficient_map_margin_at_gt': 109, 'prior_shift_reduces_coverage': 64, 'projection_crosses_horizon': 5, 'rectified_size_unsupported': 4}.

## Початкові кадри: пояснення або нерозв’язана причина

- Scene_05/L05_0001.npz: accepted_xy_uncalibrated; покриття {'footprint_coverage': 1.0000000000000002, 'gt_search_coverage': 0.926233150644349, 'prior_search_coverage': 0.9539395734597157}; чинники ['insufficient_map_margin_at_gt']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0002.npz: insufficient_map_coverage; покриття {'footprint_coverage': 0.9999999999999999, 'gt_search_coverage': 0.886122569239835, 'prior_search_coverage': 0.9491750147318798}; чинники ['insufficient_map_margin_at_gt']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0003.npz: insufficient_map_coverage; покриття {'footprint_coverage': 1.0, 'gt_search_coverage': 0.8175085172567027, 'prior_search_coverage': 0.723300251814546}; чинники ['insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0004.npz: insufficient_map_coverage; покриття {'footprint_coverage': 0.9769224215612684, 'gt_search_coverage': 0.7458481613285883, 'prior_search_coverage': 0.7648279952550415}; чинники ['footprint_outside_map', 'insufficient_map_margin_at_gt']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0005.npz: insufficient_map_coverage; покриття {'footprint_coverage': 0.9323757069550103, 'gt_search_coverage': 0.6902191171800571, 'prior_search_coverage': 0.6341695776436964}; чинники ['footprint_outside_map', 'insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0006.npz: insufficient_map_coverage; покриття {'footprint_coverage': 0.826075989280576, 'gt_search_coverage': 0.5619986746189529, 'prior_search_coverage': 0.6035632223751508}; чинники ['footprint_outside_map', 'insufficient_map_margin_at_gt']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0007.npz: insufficient_map_coverage; покриття {'footprint_coverage': 0.9329449119012222, 'gt_search_coverage': 0.6235836907394308, 'prior_search_coverage': 0.6384160575235563}; чинники ['footprint_outside_map', 'insufficient_map_margin_at_gt']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0008.npz: insufficient_map_coverage; покриття {'footprint_coverage': 0.9318897133139924, 'gt_search_coverage': 0.6743568009747902, 'prior_search_coverage': 0.6641467651384617}; чинники ['footprint_outside_map', 'insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0009.npz: insufficient_map_coverage; покриття {'footprint_coverage': 0.9819911108205231, 'gt_search_coverage': 0.7741263225392754, 'prior_search_coverage': 0.714972747675537}; чинники ['footprint_outside_map', 'insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0010.npz: insufficient_map_coverage; покриття {'footprint_coverage': 0.9999923203146052, 'gt_search_coverage': 0.7887323943661971, 'prior_search_coverage': 0.7491300745650373}; чинники ['footprint_outside_map', 'insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0011.npz: insufficient_map_coverage; покриття {'footprint_coverage': 0.9763269313178631, 'gt_search_coverage': 0.7782891604507222, 'prior_search_coverage': 0.7827328995397556}; чинники ['footprint_outside_map', 'insufficient_map_margin_at_gt']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0012.npz: insufficient_map_coverage; покриття {'footprint_coverage': 1.0, 'gt_search_coverage': 0.8018363939899833, 'prior_search_coverage': 0.7667779632721202}; чинники ['insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0013.npz: insufficient_map_coverage; покриття {'footprint_coverage': 1.0, 'gt_search_coverage': 0.7955132343447385, 'prior_search_coverage': 0.8000322788896062}; чинники ['insufficient_map_margin_at_gt']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0014.npz: insufficient_map_coverage; покриття {'footprint_coverage': 1.0, 'gt_search_coverage': 0.7977239942298445, 'prior_search_coverage': 0.8052572527648661}; чинники ['insufficient_map_margin_at_gt']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0015.npz: insufficient_map_coverage; покриття {'footprint_coverage': 0.9999999999999998, 'gt_search_coverage': 0.8469652710159039, 'prior_search_coverage': 0.7693930542031808}; чинники ['insufficient_map_margin_at_gt', 'prior_shift_reduces_coverage']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0016.npz: insufficient_map_coverage; покриття {'footprint_coverage': 1.0000000000000002, 'gt_search_coverage': 0.9186526850969177, 'prior_search_coverage': 0.9407372100413092}; чинники ['insufficient_map_margin_at_gt']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0017.npz: accepted_xy_uncalibrated; покриття {'footprint_coverage': 1.0, 'gt_search_coverage': 1.0, 'prior_search_coverage': 1.0}; чинники []; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0018.npz: accepted_xy_uncalibrated; покриття {'footprint_coverage': 1.0, 'gt_search_coverage': 1.0, 'prior_search_coverage': 0.9996683800364782}; чинники ['prior_shift_reduces_coverage']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0019.npz: insufficient_map_coverage; покриття {'footprint_coverage': 0.9999999999999998, 'gt_search_coverage': 1.0, 'prior_search_coverage': 0.9315687540348612}; чинники ['prior_shift_reduces_coverage']; помилка None; зіставність підтверджено: True.
- Scene_05/L05_0020.npz: accepted_xy_uncalibrated; покриття {'footprint_coverage': 0.9999999999999999, 'gt_search_coverage': 0.9987152722017023, 'prior_search_coverage': 1.0}; чинники []; помилка None; зіставність підтверджено: True.

## Як читати висновки

Вихід footprint означає відсутність частини фактичної видимої ділянки. Недостатній запас при GT означає обмеження карти для заданої невизначеності. Зменшення покриття через prior показує внесок конкретного зміщення. Чинники можуть поєднуватися.
Помилки читання/координат записані дослівно: конкретний дефект потребує огляду, винуватість текстури з цих помилок не випливає.
Статус придатності стосується поточного протоколу, не гарантує успіху SIFT чи плоского наближення.
Деталі кожного кадру — scenes/<scene>/frames.csv; рисунки — geometry-*.png.
Контрольний дослід зберігається окремо та не змінює робочу політику локалізатора.
