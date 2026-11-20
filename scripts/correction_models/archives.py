"""Discover and safely unpack existing AnyVisLoc ZIPs; never download or delete archives."""
from pathlib import Path, PurePosixPath
import re
import zipfile

from scripts.visibility_localization.colab import stage_archive
from .artifacts import digest, save_verified, load_verified

SCENE = re.compile(r'Scene_\d{2}')
REFERENCE = re.compile(r'L\d{2}_reference\.json')


def discover_archives(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Archive directory does not exist: {root}')
    result, ignored = [], []
    for path in sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.zip'):
        with zipfile.ZipFile(path) as archive:
            references = [PurePosixPath(item.filename) for item in archive.infolist()
                          if not item.is_dir() and REFERENCE.fullmatch(PurePosixPath(item.filename).name)
                          and '__MACOSX' not in PurePosixPath(item.filename).parts]
        if not references:
            ignored.append(str(path))
            continue
        embedded = {ref.parent.name for ref in references if SCENE.fullmatch(ref.parent.name)}
        flat = all(len(ref.parts) == 1 for ref in references)
        prefix = None
        if flat:
            match = re.search(r'Scene_\d{2}(?!\d)', path.stem)
            if match is None or len(references) != 1:
                raise ValueError(f'Flat scene ZIP requires a Scene_XX filename and one reference JSON: {path}')
            prefix = match.group()
            embedded = {prefix}
        elif len(embedded) != len(references):
            raise ValueError(f'Each reference must be inside its own Scene_XX directory: {path}')
        result.append(dict(path=str(path), scenes=sorted(embedded), prefix=prefix,
                           reference_members=[str(ref) for ref in references]))
    return result, ignored


def unpack_archives(archive_root, dataset_root, *, receipt_root=None, progress=print):
    """Reuse complete, unchanged archives; a failed extraction resumes via file CRC checks.

    Receipts use archive path/size/mtime and require expected reference JSONs to exist.
    Source members are CRC-checked during initial extraction. Later data preparation
    verifies selected source files; receipts are not an adversarial integrity guarantee.
    """
    root = Path(dataset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipts = Path(receipt_root) if receipt_root is not None else root/'.extraction_receipts'
    archives, ignored = discover_archives(archive_root)
    extracted, reused = [], []
    for index, entry in enumerate(archives, 1):
        path = Path(entry['path'])
        destination = root/entry['prefix'] if entry['prefix'] else root
        stat = path.stat()
        signature = dict(archive=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                         destination=str(destination), reference_members=entry['reference_members'])
        receipt_path = receipts/(digest(signature)+'.json')
        references = [destination/member for member in entry['reference_members']]
        if any(not p.resolve().is_relative_to(root) for p in references):
            raise ValueError(f'Unsafe reference member in ZIP: {path}')
        if receipt_path.exists() and all(p.is_file() for p in references):
            receipt = load_verified(receipt_path)
            if receipt['signature'] != signature:
                raise ValueError('Extraction receipt identity mismatch')
            reused.append(str(path))
            if progress is not None:
                progress(f'[{index}/{len(archives)}] Already extracted: {path.name}')
            continue
        if progress is not None:
            progress(f'[{index}/{len(archives)}] Extracting {path.name} -> {destination}')
        # Shared extractor validates all paths before writing, rejects symlinks,
        # checks free space and CRC, and never replaces conflicting existing files.
        result = stage_archive(path, destination)
        if not all(p.is_file() for p in references):
            raise ValueError(f'Expected scene references missing after extraction: {path}')
        if not receipt_path.exists():
            save_verified(receipt_path, dict(signature=signature, extraction=result, scenes=entry['scenes']))
        extracted.append(str(path))
    found = {}
    for ref in sorted(root.rglob('L??_reference.json')):
        if not SCENE.fullmatch(ref.parent.name):
            continue
        if ref.parent.name in found:
            raise ValueError(f'Multiple extracted reference directories for {ref.parent.name}; '
                             'keep a single source copy per scene in DATASET_ROOT')
        found[ref.parent.name] = str(ref)
    if not archives and not found:
        raise FileNotFoundError(f'No AnyVisLoc ZIPs in {archive_root} and no extracted scenes in {root}. '
                                'Supported archives: ZIP containing Scene_XX/Lxx_reference.json, '
                                'or flat Scene_XX.zip. No downloads are performed.')
    return dict(dataset_root=str(root), discovered_archives=len(archives),
        newly_extracted_archives=len(extracted), reused_archives=len(reused),
        extracted=extracted, ignored_zip_files=ignored, scenes=sorted(found),
        available_scene_count=len(found), minimum_additional_scene_count=max(0, 16-len(found)),
        note='Scene count is not geographic independence; main selection still requires an overlap audit.')


def inventory_snapshot_paths(dataset_root, data_dir):
    """Adding extracted scenes creates a new snapshot, never silently reuses a four-scene inventory."""
    sources = []
    for ref in sorted(Path(dataset_root).resolve().rglob('L??_reference.json')):
        sources.append(dict(reference=str(ref), frames=len(list(ref.parent.glob('L??_????.npz')))))
    signature = digest(sources)[:16]
    return (Path(data_dir)/('inventory-'+signature+'.json'),
            Path(data_dir)/('geographic_audit-'+signature+'.json'))
