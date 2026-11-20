"""Regenerate clean, unexecuted Colab notebooks for this experiment only."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).parent


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def write(name, cells, gpu=False):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                             language_info=dict(name='python'), colab=dict(name=name))
    if gpu:
        notebook.metadata['accelerator'] = 'GPU'
    nbf.validate(notebook)
    nbf.write(notebook, ROOT/name)


setup = '''import os, sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from google.colab import drive
drive.mount('/content/drive')
PROJECT_ROOT = Path('/content/drive/MyDrive/PhD/proj')
EXPERIMENT = PROJECT_ROOT / 'scripts/correction_models'
assert EXPERIMENT.is_dir(), f'Upload scripts/correction_models and scripts/visibility_localization: {EXPERIMENT}'
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
from scripts.correction_models.environment import configure_cuda_determinism
configure_cuda_determinism()
'''

collection_setup = '''import os, sys
_loaded_torch = sys.modules.get('torch')
if os.environ.get('CUDA_LAUNCH_BLOCKING', '0') != '0':
    if _loaded_torch is not None and _loaded_torch.cuda.is_initialized():
        raise RuntimeError('CUDA was already initialized with a diagnostic environment. '
                           'Use Runtime -> Restart session, then run this cell first. '
                           'Reconnecting the browser does not necessarily restart Python.')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    print('Disabled inherited CUDA_LAUNCH_BLOCKING before CUDA initialization.', flush=True)
os.environ['TORCH_LINALG_PREFER_CUSOLVER'] = '1'
# End of CUDA startup
'''+setup

install = '''import subprocess
from scripts.correction_models.environment import loaded_snapshot, require_clean_kernel, verify_scipy_imports
INSTALL_DEPENDENCIES = True  # After a requested session restart: False.
if INSTALL_DEPENDENCIES:
    before = loaded_snapshot()
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(EXPERIMENT / REQUIREMENTS)])
    require_clean_kernel(before)
verify_scipy_imports()
'''

write('01_collect_colab.ipynb', [
    md('# Correction models: збір даних\nGPU потрібен лише для локалізатора та описів CAMP. '
       '160 кадрів × 4 невизначеності × 3 сценарії = 1920 спроб. '
       'Це не навчання CAMP/RoMa і не перевірка доставки. Справжні кадри не є входом прогнозної моделі.'),
    code(collection_setup),
    code("REQUIREMENTS = 'requirements-colab.txt'\n"+install),
    md('Якщо комірка просить перезапустити Python: **Runtime → Restart session**, потім '
       '`INSTALL_DEPENDENCIES=False` і виконайте комірки зверху. Не запускайте моделі у змішаному середовищі.'),
    code('''ARCHIVE_ROOT = Path('/content/drive/MyDrive/PhD/AnyVisLoc')  # Folder containing existing Scene_XX ZIPs: adjust.
DATASET_ROOT = ARCHIVE_ROOT / 'extracted'  # Persistent extraction destination; may point to existing unpacked scenes.
DATA = EXPERIMENT / 'data/main-v1'
OUTPUT = EXPERIMENT / 'results/main-v3-cusolver'  # Keep the previous CUDA-error run unchanged.
RUNTIME = Path('/content/correction_models_runtime')
INVENTORY = DATA / 'inventory.json'
AUDIT = DATA / 'geographic_audit.json'
SELECTION = DATA / 'selection.json'
CAMP_CHECKPOINT = None  # Automatically use/download the official checkpoint.
os.environ['VISIBILITY_MODEL_CACHE'] = str(EXPERIMENT / 'model_cache')
assert ARCHIVE_ROOT.is_dir(), f'Point ARCHIVE_ROOT at the folder containing your AnyVisLoc ZIPs: {ARCHIVE_ROOT}'
'''),
    md('## Необов’язково: OneDrive → Colab → Google Drive\nДля цього достатньо CPU runtime. '
       'Ваш ПК не зберігає архіви: Python на сервері Colab записує їх у змонтований Drive. '
       'Увімкніть DOWNLOAD_MISSING_SCENES, щоб додати сцени до загальної кількості 16. '
       'Це технічний відбір без перегляду результатів локалізації, не підтвердження географічної незалежності. '
       'Автоматичний доступ до OneDrive може вимагати авторизації. Якщо він повертає 401/403, потрібні '
       'адреси файлів із браузерного HAR/JSON у ONEDRIVE_BROWSER_METADATA або дозволені прямі HTTPS-посилання '
       'на ZIP у DIRECT_URLS; адреса сторінки папки не є таким посиланням. '
       'Прямі downloadUrl можуть швидко спливати; не зберігаємо їх у звітах.'),
    md('**Запасний шлях після 401 — метадані публічної папки з браузера:** відкрийте офіційну папку '
       'авторів у Chrome/Edge → F12 → Network → очистіть список → перезавантажте сторінку й відкрийте '
       'каталог зі сценами. Збережіть **Export HAR (sanitized)** з вмістом відповідей. Перенесіть цей невеликий '
       'файл на свій Drive та вкажіть шлях у ONEDRIVE_BROWSER_METADATA. Архіви на ПК для цього не завантажуються. '
       'Не надсилайте HAR у чат: він може містити тимчасові адреси доступу. Якщо у відповідях немає '
       'явних downloadUrl, код також підтримує повний JSON RenderListDataAsStream: ListData.Row з '
       'FileLeafRef/.spItemUrl та ListSchema з відповідними .driveUrl/.driveAccessToken. Тимчасовий токен '
       'використовується лише для читання метаданих потрібного файлу; ZIP отримується за повернутою сервером адресою. '
       'Для такого JSON задайте ONEDRIVE_BROWSER_METADATA = EXPERIMENT / "one_drive_catalog.json". '
       'Окремий Row без ListSchema недостатній.'),
    code('''from urllib.error import HTTPError
from scripts.correction_models.downloads import download_archive
from scripts.correction_models.download_access import scene_links
from scripts.correction_models.archives import discover_archives, SCENE
DOWNLOAD_MISSING_SCENES = False  # Set True to transfer missing archives to your Drive.
TARGET_SCENE_COUNT = 16
ONEDRIVE_BROWSER_METADATA = None  # HAR/JSON path, e.g. EXPERIMENT / 'one_drive_catalog.json'
DIRECT_URLS = {}  # Optional fallback: {'Scene_01': 'https://...actual ZIP download URL...'}
if DOWNLOAD_MISSING_SCENES:
    local_archives, _ = discover_archives(ARCHIVE_ROOT)
    existing = {scene for item in local_archives for scene in item['scenes']}
    existing.update(p.parent.name for p in DATASET_ROOT.rglob('L??_reference.json'))
    missing_count = max(0, TARGET_SCENE_COUNT - len(existing))
    if missing_count:
        # Resolve ephemeral download URLs immediately before use; never persist them.
        catalog = ({s: {'url': url} for s, url in DIRECT_URLS.items()}
                   if DIRECT_URLS else scene_links(None, browser_metadata=ONEDRIVE_BROWSER_METADATA))
        if any(not SCENE.fullmatch(s) for s in catalog):
            raise ValueError('Scene identifiers must use Scene_XX format.')
        wanted = sorted(set(catalog) - existing)[:missing_count]
        print('Already present:', sorted(existing))
        print('Archives to transfer:', wanted)
        if not wanted:
            raise RuntimeError('No additional scenes found. Check OneDrive access or DIRECT_URLS.')
        for scene in wanted:
            print('Transferring', scene, 'to', ARCHIVE_ROOT)
            def fresh_download_url(scene=scene):
                return (DIRECT_URLS[scene] if DIRECT_URLS else
                        scene_links([scene], browser_metadata=ONEDRIVE_BROWSER_METADATA)[scene]['url'])
            try:
                download_archive(None, ARCHIVE_ROOT / (scene + '.zip'), url_provider=fresh_download_url)
            except HTTPError as exc:
                if exc.code in (401, 403):
                    raise RuntimeError('OneDrive refused the ZIP URL. It may have expired or require additional access. '
                                       'Obtain fresh browser metadata or a fresh direct ZIP URL; '
                                       'do not copy account cookies or passwords.') from exc
                raise
        if len(wanted) < missing_count:
            print('More scene links are needed:', missing_count - len(wanted))
    else:
        print('At least 16 scenes already present; no archives downloaded.')
else:
    print('OneDrive transfer is disabled. Enable DOWNLOAD_MISSING_SCENES if more archives are needed.')
'''),
    md('Після ConnectionResetError/timeout збережений `.zip.part` залишається. Є до 8 спроб із паузами '
       'та продовженням через HTTP Range; адреса ZIP отримується заново перед кожною спробою. '
       'Старий `.part` без службового файла також підтримується через перевірку перекриття 64 KiB. '
       'Не видаляйте `.part` після обриву. Готовий `.zip` з’являється лише після перевірки CRC. '
       'Якщо сервер ігнорує Range, частковий файл починається заново. Не запускайте два завантаження '
       'того самого архіву одночасно.'),
    md('## Автоматичне розпакування\nЗнаходимо ZIP-архіви в ARCHIVE_ROOT і підпапках, '
       'розпаковуємо в DATASET_ROOT. Архіви не видаляються, нові сцени мережею не завантажуються. '
       'Повторний запуск пропускає завершене розпакування. Підготовку даних можна виконати на CPU; '
       'GPU потрібен лише нижче, для локалізатора. Чотирьох сцен недостатньо для основного експерименту на 16 територіях.'),
    code('''from scripts.correction_models.archives import unpack_archives
archive_summary = unpack_archives(ARCHIVE_ROOT, DATASET_ROOT, receipt_root=DATA / 'extraction_receipts')
print(archive_summary)
if archive_summary['available_scene_count'] < 16:
    print('For the agreed main experiment, add at least', archive_summary['minimum_additional_scene_count'],
          'more scenes to ARCHIVE_ROOT and rerun this cell. Eligibility is checked from the inventory.')
'''),
    md('## Інвентаризація та розбиття за сценами\nГотовий inventory повторно використовуємо без читання '
       'кадрів. AnyVisLoc приховує глобальні координати: reviewed=true для географічного аудиту не потрібне. '
       'Офіційна сцена входить лише в одну частину вибірки. Географічне неперекриття не вважається доведеним. '
       'Відомі спільні групи з попереднього JSON зберігаються; точні дублікати карт перевіряються при підготовці.'),
    code('''import json
import gc
import psutil
from scripts.correction_models.selection import inventory, audit_template, prepare_selection, selection_readiness
from scripts.correction_models.artifacts import load_verified, read
from scripts.correction_models.archives import inventory_snapshot_paths
INVENTORY, AUDIT = inventory_snapshot_paths(DATASET_ROOT, DATA)
gc.collect()
print('Stage: metadata inventory (no images/maps/models)', flush=True)
print('Reader:', __import__('scripts.correction_models.selection', fromlist=['__file__']).__file__, flush=True)
print('Initial RSS MiB:', round(psutil.Process().memory_info().rss / 1024**2), flush=True)
if not INVENTORY.exists():
    print(inventory(DATASET_ROOT, INVENTORY, progress=lambda message: print(message, flush=True)))
else:
    snapshot = load_verified(INVENTORY)
    print('Cached inventory:', len(snapshot['rows']), 'frames;', len(snapshot['errors']), 'metadata errors')
    del snapshot
print('Selection readiness:', selection_readiness(load_verified(INVENTORY)), flush=True)
print('Stage: scene grouping policy', flush=True)
if not AUDIT.exists():
    audit_template(INVENTORY, AUDIT)
print('Scene grouping source:', AUDIT)
print('Global geographic non-overlap: unverified; reviewed=true is not required')
'''),
    md('Якщо інвентаризація зупинилася через RAM: перезапустіть сеанс і виконайте налаштування та цю '
       'комірку з тими самими DATASET_ROOT/DATA. Повторювати завантаження й розпакування не потрібно. '
       'Завершені сцени зберігаються в inventory-<id>.scenes і пропускаються; вивід показує файл, RSS '
       'процесу та доступну системну RAM. За залишку менш як 512 MiB буде явна помилка. '
       'Цей поріг не гарантує захисту від зовнішнього скидання Colab. На цьому етапі не запускайте '
       'ініціалізацію CAMP/RoMa чи завантаження повних карт.'),
    md('Порожній старий geographic_audit автоматично читається як джерело відомих груп; змінювати його '
       'на reviewed=true не потрібно. У selection.json зберігається ефективна політика з '
       'geographic_overlap_verified=false. Якщо є WGS84-полігони, їхні перекриття й далі перевіряються. '
       'Часткове візуальне перекриття автоматично не підтверджене й позначене як not_verified. '
       'Потрібно 16 придатних груп, по 10 різних позицій; Scene_05/07/08/11/21 не можуть бути тестовими. '
       'Scene_12 у поточному inventory має лише 4 придатні кадри: додайте щонайменше одну придатну сцену. '
       'Після додавання скануються нові сцени; завершені сцени використовують кеш. '
       'Старий selection іншого протоколу не переписується: використайте новий шлях SELECTION/OUTPUT.'),
    code('''from scripts.correction_models.artifacts import load_verified
from scripts.correction_models.selection import prepare_selection, validate_selection

if not SELECTION.exists():
    print(prepare_selection(INVENTORY, AUDIT, SELECTION))
else:
    frozen_selection = load_verified(SELECTION)
    validate_selection(frozen_selection)
    print('Using frozen selection:', frozen_selection['split_scenes'])
'''),
    md('## Локалізатор і частина збору\nВаги кешуються на Drive. Завантаження ваг не входить у час корекції. '
       'Використовується cuSOLVER; версії PyTorch/CUDA, GPU та режим виконання записуються в manifest. '
       'CUDA_LAUNCH_BLOCKING=1 вимкніть перед основним збором: це діагностичний режим. '
       'Профілювання відбувається разом із збором. За першої технічної помилки запуск зупиняється й зберігає traceback. '
       'Відхилена корекція — штатний результат, а не технічна помилка.'),
    code('''from scripts.correction_models.colab import initialize_backend
backend = initialize_backend(RUNTIME, CAMP_CHECKPOINT)
print('Initialization profile:', backend.initialization_profile)
'''),
    code('''from scripts.correction_models.campaign import run_collection
MAX_NEW_ATTEMPTS = 1  # First normal-mode attempt; after success use 120.
MAX_SECONDS = 3300  # Soft limit: the currently running attempt is allowed to finish.
summary = run_collection(SELECTION, backend, OUTPUT,
    max_new_attempts=MAX_NEW_ATTEMPTS, max_seconds=MAX_SECONDS)
print(json.dumps(summary, indent=2))
'''),
    md('Після першої успішної спроби встановіть MAX_NEW_ATTEMPTS=120 і повторюйте до `complete=true`. '
       'Попередня діагностика однієї пари не гарантує стабільність усіх сцен. '
       'Після розриву сеансу виконайте налаштування знову '
       'та продовжуйте з тими самими SELECTION/OUTPUT. Збережені спроби пропускаються. '
       'Не запускайте два процеси збору в одній OUTPUT одночасно. Після технічної помилки спочатку виправте '
       'причину; старий запис не перезаписується автоматично. `dataset.json` з’явиться після всіх 1920 спроб. '
       'Фінальні тестові результати не показуються в поточному зведенні.'),
], gpu=True)

write('02_train_evaluate_cpu.ipynb', [
    md('# Correction models: навчання й незалежний тест\nДостатньо CPU runtime. Ваги CAMP/RoMa не завантажуються: '
       'використовуємо вже збережені описи карти. Кожна модель повертає вісім імовірностей та емпіричний прогноз похибки.'),
    code(setup),
    code("REQUIREMENTS = 'requirements-cpu.txt'\n"+install),
    code('''import json
DATASET = EXPERIMENT / 'results/main-v3-cusolver/dataset.json'
RUN = EXPERIMENT / 'results/models-v3-cusolver'
TEST_OUTPUT = EXPERIMENT / 'results/test-v3-cusolver'
assert DATASET.is_file(), 'Complete the GPU collection first: dataset.json is published only after 1920 attempts.'
from scripts.correction_models.dataset import load_dataset, usable
data = load_dataset(DATASET)
print({s: len(usable(data['rows'], s)) for s in ('train', 'validation', 'calibration', 'test')})
'''),
    md('## Навчання\nМоделі навчаються лише на train. Температура калібрування — лише на calibration. '
       'Вибір кандидата — за validation log loss. Середня похибка не підміняє розподіл. '
       'Якщо певного класу не було в train, його згладжена ймовірність позначається як unresolved mass: '
       'для неї немає вигаданого вектора похибки.'),
    code('''from scripts.correction_models.training import run_training
from scripts.correction_models.artifacts import read, load_verified
if not RUN.exists():
    run_training(DATASET, RUN)
development = read(RUN / 'summary.json')
print('Selected:', development['selected'])
print(json.dumps(development['classification'][development['selected']], indent=2))
print(json.dumps(development['distribution'][development['selected']], indent=2))
'''),
    code('''from IPython.display import display, Image, Markdown
display(Markdown((RUN / 'report.md').read_text()))
display(Image(filename=str(RUN / 'calibration.png')))
display(Image(filename=str(RUN / 'territories.png')))
display(Image(filename=str(RUN / 'localizer_visibility.png')))
'''),
    md('## Остаточний тест\nПісля фіксації моделей увімкніть RUN_FINAL_TEST. Вибраний кандидат уже зафіксований. '
       'Тест не використовуємо для нових налаштувань. Повторна комірка читає готовий звіт без повторного оцінювання. '
       'Чотири території — обмежена перевірка узагальнення; bootstrap-інтервали позначені exploratory.'),
    code('''RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    from scripts.correction_models.training import evaluate_frozen
    if not (TEST_OUTPUT / 'summary.json').exists():
        evaluate_frozen(RUN, DATASET, TEST_OUTPUT)
    final = read(TEST_OUTPUT / 'summary.json')
    print('Frozen selected:', final['selected'])
    print(json.dumps(final['classification'][final['selected']], indent=2))
    print(json.dumps(final['distribution'][final['selected']], indent=2))
    display(Image(filename=str(TEST_OUTPUT / 'calibration.png')))
    display(Image(filename=str(TEST_OUTPUT / 'territories.png')))
    display(Image(filename=str(TEST_OUTPUT / 'localizer_visibility.png')))
'''),
    md('## Приклад виклику для планування\nВикористовуємо тільки збережений GT-free запит навчального кадру '
       'та карту. Еталонні координати, реальний кадр і метрики зіставлення не передаються моделі. '
       'Якщо обрано CAMP-кандидат, точний опис цього фрагмента читається з кешу без GPU. Для нового фрагмента '
       'потрібно обчислити опис незміненою CAMP; для нової карти — також перевірити домен.'),
    code('''from scripts.correction_models.contracts import CorrectionQuery
from scripts.correction_models.prediction import load_predictor
from scripts.correction_models.embeddings import CachedCampEncoder
from scripts.visibility_localization.data import load_reference
selection = load_verified(EXPERIMENT / 'data/main-v1/selection.json')
row = next(r for r in data['rows'] if r['split'] == 'train' and r['features'] is not None)
source = next(s for s in selection['rows'] if s['scene_id'] == row['scene_id'])
reference = load_reference(source['reference_json'], 'aerial')
encoder = CachedCampEncoder(DATASET.parent / 'embeddings', data['provenance']['encoder'])
predictor = load_predictor(RUN, reference, encoder=encoder)
answer = predictor.predict_correction(CorrectionQuery(**row['query']))
print({k: v for k, v in answer.items() if k != 'accepted_error_distribution'})
'''),
    md('Моделі збережені в JSON у `models/`, вибір — у `frozen_run.json`. '
       'Результат цього етапу не є підтвердженням ймовірності доставки чи швидкості на Jetson. '
       'Планувальник і маршрутний експеримент не запускаються.'),
])
