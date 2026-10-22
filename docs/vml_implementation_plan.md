# Завершення інкременту VML на AnyVisLoc

## 1. Результат інкременту

Після інкременту VML має бути не синтетичним експериментом, а завершеним дослідним компонентом із двома інтерфейсами:

- локалізатор: кадр + camera model + attitude/altitude + prior mean/covariance + карта → `reject` або `(z, R, diagnostics)`;
- умовна модель для planner: умови майбутньої VML-спроби → `p_reject`, `p_local`, `p_gross`, розподіли похибок і часу.

Валідація виконується на реальних зовнішніх кадрах AnyVisLoc окремо для аерофото і супутникових карт. Датасет надає `K`, distortion, pose та обидва типи карт; оцінюється лише горизонтальна локалізація, оскільки ground truth не призначений для абсолютної висоти. [AnyVisLoc benchmark](https://github.com/UAV-AVL/Benchmark), [опис ground truth](https://github.com/UAV-AVL/Benchmark/blob/main/docs/GROUND_TRUTH.md).

EuRoC не використовується для підтвердження VML над відкритою місцевістю; він може залишитися лише технічним indoor-тестом VIO. :codex-annotation{index="1"}

## 2. Реалізація та публічні контракти

- Додати адаптер AnyVisLoc, який:
  - читає NPZ і reference JSON без копіювання датасету в Git;
  - обчислює AGL відносно DSM у межах сцени;
  - залишає висоти 20–120 м;
  - використовує реальні `K`, distortion та attitude;
  - переводить результати у ROS ENU: `x=East`, `y=North`, yaw у радіанах проти годинникової стрілки від `+x`.

- Реалізувати planar pipeline:
  - undistortion і rectification до virtual-nadir;
  - пошуковий crop як сума footprint та covariance-ellipse;
  - `search_probability_mass` надходить із planner/config, а не зашивається у VML;
  - DSM використовується лише для оцінки relief/паралаксу і геометричної придатності, не для PnP;
  - невиродженість проєкції, покриття карти й conditioning визначають підтримуваність кадру без фіксованої межі кута.

- Контракт локалізатора:
  - вхід: зображення, `K/dist`, attitude, AGL, prior `mean + covariance`, `search_probability_mass`, map source;
  - вихід: статус, `z=[x,y,yaw]`, маска наявних компонентів, `R`, matcher diagnostics і runtime;
  - коли yaw ненадійний, повертаються лише `x,y` та `2×2 R`;
  - координати gross false match не маскуються великою `R`, а належать окремій гілці результатів.

- Розділити статистику на:
  - pre-match модель для planner, яка використовує лише відомі наперед ознаки;
  - post-match модель `R`, яка додатково використовує matches, inliers, reprojection error, conditioning і покриття.

- Навчання виконується через окремий Colab notebook із Google Drive. Експериментальний протокол, локалізатор, fit/export, loader і генератор planner-таблиць залишаються тестованими Python-модулями в `scripts/pipeline/chance_constrained/visual_map_localization/`.

## 3. Експеримент і статистична модель

- Кампанія запускається як `smoke → pilot → full`:
  - smoke: 20 кадрів для перевірки всього data flow;
  - pilot: три сцени, детерміновано вибрані maximin/medoid-алгоритмом за altitude, tilt, GSD, texture і relief;
  - full: усі геометрично придатні кадри всіх доступних сцен.

- Одиниця відновлення після розриву Colab-сесії: `scene × reference_mode × matcher × campaign`. Кожна одиниця записує окремий per-trial shard, checksum і статус у Drive.

- Для кожного режиму карти:
  1. Запустити SIFT.
  2. Налаштовувати acceptance thresholds лише на training folds.
  3. Якщо SIFT не проходить validation gate, запустити SP-LightGlue-GIM за тим самим протоколом.
  4. Вибрати matcher з кращою held-out прогнозною якістю; час Colab лише звітується, бо це не цільове обладнання.

- Умовна модель:
  - `SplineTransformer + LogisticRegression` для reject probability та умовної ваги gross-компонента;
  - accepted residuals поділяються robust mixture-моделлю на local і gross без фіксованого порога похибки;
  - spline/ridge-регресії прогнозують масштаб локальної covariance і runtime;
  - коефіцієнти, spline knots, covariance та mixture parameters експортуються як JSON/NPZ, без pickle.

- Pre-match ознаки: AGL, tilt, GSD ratio, search-area/footprint ratio, ellipse eccentricity, yaw uncertainty, texture/distinctiveness, DSM relief, map coverage. Post-match `R` додатково використовує геометричну діагностику.

- Якщо запит виходить за навчальний домен:
  - повертається point estimate, `extrapolated=true`, warning і distance-to-domain;
  - статистичні bounds стають повним інтервалом `[0,1]`;
  - planner за замовчуванням не отримує необґрунтованої консервативної переваги від екстраполяції.

- Blur/haze/contrast виконуються лише як окремий sensitivity test. Рівні деградації прив’язуються до квантилів реальних image-quality metrics і не домішуються до емпіричних польотних імовірностей.

## 4. Валідація та інтеграція з planner

- Застосувати 5-fold GroupKFold за сценами окремо для `aerial` і `satellite`. Якщо після фільтрації доступно менше п’яти сцен, режим автоматично отримує `validation_passed=false`.

- Режим проходить валідацію, якщо:
  - out-of-fold negative log-likelihood умовної моделі кращий за intercept-only/base-rate модель із парним 95% bootstrap interval;
  - calibration tests при `α=0.05` не виявляють систематичного недооцінювання ризику;
  - predictive coverage для `R` не має статистично значущого undercoverage;
  - двокомпонентна residual-модель покращує held-out likelihood відносно однокомпонентної та не вироджується.

- Додатково опублікувати Acc@1/3/5/10/20 м, position/yaw error curves, reject rate, gross rate, reliability plots і runtime distribution. Ці метрики не замінюють перевірку калібрування умовних імовірностей.

- Статуси `aerial` і `satellite` незалежні. За погодженим рішенням найкращий артефакт завжди має `deployment_status=production_enabled`, але фактичний `validation_passed` не змінюється і loader видає warning при невдалій валідації.

- Генератор planner-таблиць приймає обов’язкові `attempt_period_s` і `search_probability_mass`, формує впорядковані VML-спроби вздовж орієнтованих ребер та переносить часову фазу між ребрами.

- Таблиці містять point estimates і bootstrap bounds. Chance-constrained planner за замовчуванням використовує нижню межу `p_local` та верхні межі `p_gross/p_reject`.

## 5. Артефакти, тести й межі результату

У Git мають з’явитися:

- Colab notebook для Drive, підготовки даних і запуску навчання;
- адаптер AnyVisLoc, matcher interface, геометрія, outcome/R models, loader та генератор edge-opportunities;
- версовані схеми конфігурацій і manifest;
- unit-тести координатних перетворень, covariance crops, partial yaw, PSD `R`, probability normalization, extrapolation, serialization і edge scheduling;
- integration smoke-test на малому синтетичному fixture без включення AnyVisLoc-даних.

У приватному Google Drive мають бути:

- inventory та eligibility manifest;
- resumable per-trial shards із checksum;
- tuned matcher config для кожного режиму карти;
- `vml_outcome_model.npz`, `manifest.json`, `summary.json`;
- cross-scene validation report, calibration/error-mixture plots та окремий stress sensitivity report.

Manifest обов’язково містить dataset/matcher/config fingerprints, `validation_passed` окремо за режимами, `deployment_status`, supported domain, license marker і `target_platform_calibrated=false`.

Інкремент не включає global retrieval, DSM/PnP-локалізацію, власні польотні записи, ROS2 node, інтеграцію TAN або вимірювання продуктивності на цільовому бортовому комп’ютері. Отриманий результат означатиме: підхід VML і його умовна ймовірнісна модель перевірені між сценами на публічному outdoor-датасеті, але ще не є польовою валідацією для конкретної української місцевості, камери та апаратної платформи.
