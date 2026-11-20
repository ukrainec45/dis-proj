"""Resumable ZIP transfers, including .part files from the original downloader."""
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

RANGE = re.compile(r'^bytes (\d+)-(\d+)/(\d+)$')
UNSATISFIED = re.compile(r'^bytes \*/(\d+)$')
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
OVERLAP_BYTES = 64*1024


class DownloadInterrupted(ConnectionError):
    """A response ended before all advertised bytes arrived."""


def _valid_zip(path):
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f'ZIP integrity check failed: {bad}')


def _publish(partial, destination, checkpoint, progress):
    if progress is not None:
        progress(f'{destination.name}: checking ZIP integrity')
    _valid_zip(partial)
    if destination.exists():
        _valid_zip(destination)
        return destination
    os.replace(partial, destination)
    checkpoint.unlink(missing_ok=True)
    if progress is not None:
        progress(f'{destination.name}: complete ({destination.stat().st_size/1024**2:.1f} MiB)')
    return destination


def _save_checkpoint(path, total, etag, modified):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(dict(version=1, total=total, etag=etag, last_modified=modified)), encoding='utf-8')
    os.replace(temporary, path)


def _integer_header(headers, key):
    value = headers.get(key)
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f'Invalid {key} header')
    return parsed


def download_archive(url, destination, max_bytes=15_000_000_000, *, url_provider=None,
                     max_attempts=8, chunk_size=1024**2, timeout=60, retry_delay=2,
                     progress=print, opener=None, sleeper=time.sleep):
    """Keep completed bytes on reset/timeout, refresh URLs between retry attempts.

    A 206 response must match the requested range. If-Range uses a stored strong
    ETag; legacy partial files without validators verify a 64 KiB overlap first.
    If a server ignores Range (200), restart the partial, never append its body.
    Final publication always requires ZIP CRC verification. Signed URLs/tokens
    are not written into the checkpoint or progress log. No concurrent writers.
    """
    if type(max_attempts) is not int or max_attempts < 1 or chunk_size < 1 or max_bytes < 1:
        raise ValueError('Positive download limits required')
    if retry_delay < 0 or timeout <= 0:
        raise ValueError('Invalid retry delay/timeout')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _valid_zip(destination)
        if progress is not None:
            progress(f'{destination.name}: existing complete ZIP retained')
        return destination
    partial = destination.with_suffix(destination.suffix+'.part')
    checkpoint = destination.with_suffix(destination.suffix+'.part.meta.json')
    opener = opener or urlopen
    failures, auth_refreshes = 0, 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > max_bytes:
            raise ValueError('Existing partial exceeds configured archive budget')
        state = {}
        if checkpoint.exists():
            try:
                state = json.loads(checkpoint.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                # Treat a damaged checkpoint as a legacy partial; verify overlap.
                state = {}
        old_total = state.get('total')
        if offset and old_total == offset:
            return _publish(partial, destination, checkpoint, progress)
        etag = state.get('etag')
        strong = isinstance(etag, str) and etag.startswith('"') and etag.endswith('"')
        start = offset if strong else max(0, offset-OVERLAP_BYTES)
        headers = {'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={start}-'
            if strong:
                headers['If-Range'] = etag
        try:
            # A fresh metadata lookup can use an updated catalog between attempts.
            fresh_url = url_provider() if url_provider is not None else url
            if not isinstance(fresh_url, str) or not fresh_url.startswith('https://'):
                raise ValueError('HTTPS ZIP URL required')
            with opener(Request(fresh_url, headers=headers), timeout=timeout) as response:
                status = response.status
                if status not in (200, 206):
                    raise ValueError(f'Unexpected ZIP response status: {status}')
                content_type = response.headers.get('Content-Type', '').split(';')[0].lower()
                if content_type in ('text/html', 'application/json', 'text/plain'):
                    raise ValueError('Server returned a page/JSON instead of ZIP bytes; obtain a fresh download URL')
                if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                    raise ValueError('Encoded HTTP response cannot be resumed by byte offset')
                length = _integer_header(response.headers, 'Content-Length')
                new_etag = response.headers.get('ETag')
                modified = response.headers.get('Last-Modified')
                if status == 206:
                    match = RANGE.fullmatch(response.headers.get('Content-Range', ''))
                    if match is None:
                        raise ValueError('Missing or invalid Content-Range in partial response')
                    first, last, total = map(int, match.groups())
                    if first != start or last < first or total <= last:
                        raise ValueError('Server returned a different byte range; partial file preserved')
                    if length is not None and length != last-first+1:
                        raise ValueError('Content-Length disagrees with Content-Range')
                    if old_total is not None and total != old_total:
                        raise ValueError('Remote archive size changed; partial file preserved')
                    if etag and new_etag and etag != new_etag:
                        raise ValueError('Remote archive ETag changed; partial file preserved')
                    if state.get('last_modified') and modified and state['last_modified'] != modified:
                        raise ValueError('Remote archive modification time changed; partial file preserved')
                    overlap = offset-first
                    if last+1 < offset:
                        raise ValueError('Partial response contains no new data')
                    segment_bytes = last-first+1
                else:
                    total, overlap, segment_bytes = length, 0, length
                    if offset and progress is not None:
                        progress(f'{destination.name}: server returned a full response; restarting partial instead of appending')
                    offset = 0
                if total is not None and total > max_bytes:
                    raise ValueError('Archive exceeds configured byte budget')
                needed = max(0, total-offset) if total is not None else chunk_size
                if shutil.disk_usage(destination.parent).free < needed+512*1024**2:
                    raise RuntimeError('Insufficient free space for the remaining archive')
                consumed = 0
                if overlap:
                    with partial.open('rb') as existing:
                        existing.seek(start)
                        remaining = overlap
                        while remaining:
                            chunk = response.read(min(remaining, chunk_size))
                            if not chunk:
                                raise DownloadInterrupted('Connection ended while verifying the saved overlap')
                            if chunk != existing.read(len(chunk)):
                                raise ValueError('Saved partial differs from the remote archive; partial preserved')
                            consumed += len(chunk)
                            remaining -= len(chunk)
                _save_checkpoint(checkpoint, total, new_etag, modified)
                if progress is not None:
                    progress(f'{destination.name}: {"resuming" if offset else "starting"} at {offset/1024**2:.1f} MiB')
                saved = offset
                last_log = time.monotonic()
                with partial.open('ab' if offset else 'wb') as stream:
                    while True:
                        incomplete = False
                        try:
                            chunk = response.read(chunk_size)
                        except http.client.IncompleteRead as exc:
                            chunk, incomplete = exc.partial, True
                        if not chunk:
                            if incomplete:
                                raise DownloadInterrupted('Truncated HTTP body')
                            break
                        if saved+len(chunk) > max_bytes or (total is not None and saved+len(chunk) > total):
                            raise ValueError('Response exceeds declared archive size/budget')
                        stream.write(chunk)
                        saved += len(chunk)
                        consumed += len(chunk)
                        if progress is not None and time.monotonic()-last_log >= 10:
                            suffix = f' / {total/1024**2:.1f}' if total is not None else ''
                            progress(f'{destination.name}: {saved/1024**2:.1f}{suffix} MiB saved')
                            last_log = time.monotonic()
                        if incomplete:
                            raise DownloadInterrupted('Truncated HTTP body')
                if segment_bytes is not None and consumed != segment_bytes:
                    raise DownloadInterrupted('Response ended before its advertised Content-Length')
                if total is not None and saved < total:
                    # Valid bounded range segment: fetch the next segment normally.
                    continue
                return _publish(partial, destination, checkpoint, progress)
        except HTTPError as exc:
            if exc.code == 416 and offset:
                match = UNSATISFIED.fullmatch(exc.headers.get('Content-Range', '') if exc.headers else '')
                if match and int(match.group(1)) == offset:
                    return _publish(partial, destination, checkpoint, progress)
                raise ValueError('Server rejected the saved range; partial preserved') from None
            if exc.code in (401, 403) and url_provider is not None and auth_refreshes < 1:
                auth_refreshes += 1
            elif exc.code not in TRANSIENT_STATUS:
                raise
            reason = f'HTTP {exc.code}'
        except (ConnectionError, TimeoutError, http.client.IncompleteRead, ssl.SSLEOFError, URLError) as exc:
            if isinstance(exc, URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise
            reason = type(exc).__name__
        failures += 1
        saved = partial.stat().st_size if partial.exists() else 0
        if failures >= max_attempts:
            raise RuntimeError(f'{destination.name}: download interrupted after {max_attempts} attempts '
                f'({reason}). Saved {saved} bytes in {partial.name}. Rerun the same cell to continue; '
                'do not delete the .part file.') from None
        delay = min(30, retry_delay*2**(failures-1))
        if progress is not None:
            progress(f'{destination.name}: {reason}; {saved/1024**2:.1f} MiB retained; retry {failures+1}/{max_attempts} in {delay:g}s')
        sleeper(delay)
