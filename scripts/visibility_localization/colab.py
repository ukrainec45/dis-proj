"""Colab data acquisition. Only official portals or explicitly supplied URLs."""
import base64
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import urllib.request
import zipfile
import zlib

from .core import UPSTREAM_COMMIT,file_hash
from .campaign import write_json

OFFICIAL_ONEDRIVE = "https://1drv.ms/f/c/fef40436afe2e5bb/IgC0Zat90dsgTY-2lNlDliGXAfuTZOYEeKcllr7VPD_F3eY?e=LJ5fAD"
CAMP_URL = "https://drive.google.com/file/d/1qHjXr3VVQuJZ5kE5u7YrUB8id90Nv2GJ/view"


def checkout_upstream(destination):
    path = Path(destination)
    if not path.exists():
        subprocess.run(["git","clone","https://github.com/UAV-AVL/Benchmark.git",str(path)],check=True)
    # Upstream tracks old .pyc files; Python imports and downloaded .pth files
    # must not make an otherwise unchanged source checkout impossible to resume.
    status = subprocess.check_output(["git","-C",str(path),"status","--porcelain","--untracked-files=all"],text=True)
    if any(Path(line[3:]).suffix in {".py",".yaml",".yml",".txt",".md"} for line in status.splitlines()):
        raise ValueError("Upstream source has local changes; use a fresh directory")
    actual = subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
    if actual != UPSTREAM_COMMIT:
        subprocess.run(["git","-C",str(path),"checkout","--detach",UPSTREAM_COMMIT],check=True)
    return path


def fetch_json(url):
    request = urllib.request.Request(url,headers={"User-Agent":"AnyVisLoc-research/1.0"})
    with urllib.request.urlopen(request,timeout=30) as response:
        return json.load(response)


def official_scene_links(scenes):
    """Best-effort anonymous OneDrive shares API; 401/403 are explicit blockers.

    API route is an endpoint, not an invented archive URL. Download URLs must
    come from returned driveItem metadata. No account/authentication bypass.
    """
    token = "u!"+base64.urlsafe_b64encode(OFFICIAL_ONEDRIVE.encode()).decode().rstrip("=")
    root = "https://api.onedrive.com/v1.0/shares/"+token+"/root"
    pending,links,visited = [root+"/children"],{},0
    while pending:
        url = pending.pop(0)
        visited += 1
        if visited > 100: raise RuntimeError("Portal traversal limit; select archives manually")
        page = fetch_json(url)
        if page.get("@odata.nextLink"): pending.append(page["@odata.nextLink"])
        for item in page.get("value",[]):
            name = item.get("name","")
            if "folder" in item:
                # Traverse folders, never request a whole-folder ZIP download.
                drive = item.get("parentReference",{}).get("driveId")
                if drive and item.get("id"):
                    pending.append(f"https://api.onedrive.com/v1.0/drives/{drive}/items/{item['id']}/children")
            elif name.lower().endswith(".zip"):
                import re
                candidates = scenes if scenes is not None else re.findall(r"Scene_\d+",name,flags=re.IGNORECASE)
                for scene in candidates:
                    if name.lower().startswith(scene.lower()+".") or name.lower().startswith(scene.lower()+"_"):
                        link = item.get("@content.downloadUrl") or item.get("@microsoft.graph.downloadUrl")
                        if link:
                            if scene in links: raise ValueError(f"Ambiguous archive for {scene}")
                            links[scene] = dict(url=link,name=name,size=item.get("size"))
    if scenes is not None and set(links) != set(scenes):
        raise RuntimeError("Official portal did not expose an unambiguous ZIP for every requested scene")
    return links


def download_archive(url,destination,max_bytes=15_000_000_000):
    if not url.startswith("https://"): raise ValueError("HTTPS URL required")
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None: raise ValueError("Existing archive CRC mismatch")
        return destination
    partial = destination.with_suffix(destination.suffix+".part")
    with urllib.request.urlopen(url,timeout=60) as response:
        length = int(response.headers.get("Content-Length",0))
        if length > max_bytes: raise RuntimeError("Download exceeds configured archive budget")
        if shutil.disk_usage(destination.parent).free < (length or max_bytes)+512*1024**2:
            raise RuntimeError("Insufficient space for archive download")
        size = 0
        with partial.open("wb") as stream:
            while True:
                block = response.read(4*1024**2)
                if not block: break
                size += len(block)
                if size > max_bytes: raise RuntimeError("Archive budget exceeded")
                stream.write(block)
    with zipfile.ZipFile(partial) as archive:
        bad = archive.testzip()
        if bad: raise ValueError(f"ZIP CRC mismatch: {bad}")
    os.replace(partial,destination)
    return destination


def stage_archive(archive_path,destination):
    """Safe extraction; identical existing files are reused, changed files rejected."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        required = sum(m.file_size for m in members if not (destination/m.filename).exists())
        if shutil.disk_usage(destination).free < required+512*1024**2:
            raise RuntimeError("Insufficient extraction disk space")
        targets = set()
        for m in members:
            p = PurePosixPath(m.filename)
            target = (destination/m.filename).resolve()
            if (p.is_absolute() or ".." in p.parts or "\\" in m.filename or ":" in m.filename
                or not target.is_relative_to(destination) or target in targets
                or ((m.external_attr>>16)&0o170000) == 0o120000):
                raise ValueError("Unsafe or duplicate ZIP path")
            targets.add(target)
        for m in members:
            target = destination/m.filename
            if m.is_dir():
                target.mkdir(parents=True,exist_ok=True)
                continue
            if target.exists():
                crc = 0
                with target.open("rb") as stream:
                    for block in iter(lambda:stream.read(4*1024**2),b""): crc = zlib.crc32(block,crc)
                if target.stat().st_size != m.file_size or crc != m.CRC:
                    raise ValueError(f"Existing staged file changed: {target}")
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            partial = target.with_name(target.name+".extracting")
            with archive.open(m) as source,partial.open("wb") as stream: shutil.copyfileobj(source,stream)
            os.replace(partial,target)
    return dict(archive_name=Path(archive_path).name,sha256=file_hash(archive_path),members=len(members))


def acquire_dataset(scenes,archive_dir,dataset_root,report_path,*,archives=None,urls=None):
    """Try official auto-download first; log actual failure, require explicit fallback."""
    if not scenes: raise ValueError("Select scenes explicitly before downloading archives")
    receipts,problems = [],[]
    local = dict(archives or {})
    supplied_urls = dict(urls or {})
    already_found = {p.parent.name for p in Path(dataset_root).rglob("L??_reference.json")}
    missing = [s for s in scenes if s not in local and s not in already_found]
    if missing:
        try:
            links = official_scene_links([s for s in missing if s not in supplied_urls]) if any(s not in supplied_urls for s in missing) else {}
            for scene in missing:
                url = supplied_urls.get(scene) or links[scene]["url"]
                local[scene] = download_archive(url,Path(archive_dir)/(scene+".zip"))
        except Exception as exc:
            problems.append(dict(stage="automatic_download",type=type(exc).__name__,message=str(exc)))
    for scene in scenes:
        if scene in local:
            receipts.append(dict(scene_id=scene,**stage_archive(local[scene],dataset_root)))
    found = {p.parent.name for p in Path(dataset_root).rglob("L??_reference.json")}
    missing = sorted(set(scenes)-found)
    report = dict(official_portal=OFFICIAL_ONEDRIVE,receipts=receipts,problems=problems,missing_scenes=missing)
    write_json(report_path,report)
    if missing:
        raise RuntimeError(f"Missing scenes: {missing}. See {report_path}. Set ARCHIVES to ZIP paths in Google Drive or DATASET_ROOT to extracted data; automatic access was not verified.")
    return report
