import io
import json
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError
import zipfile

import pytest

from scripts.correction_models.downloads import download_archive


def zip_bytes():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('frames.bin', bytes(range(256))*512)
    return stream.getvalue()


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None, fail_after=None):
        super().__init__(data)
        self.status, self.fail_after = status, fail_after
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = str(value)

    def read(self, size=-1):
        if self.fail_after is not None:
            remaining = self.fail_after-self.tell()
            if remaining <= 0:
                raise ConnectionResetError(104, 'synthetic peer reset')
            size = min(size, remaining) if size >= 0 else remaining
        return super().read(size)


def served(request, data, *, ignore_range=False, fail_after=None, etag='"stable"'):
    value = request.get_header('Range')
    start = int(value.removeprefix('bytes=').removesuffix('-')) if value else 0
    status = 206 if value and not ignore_range else 200
    if status == 200:
        start = 0
    headers = {'Content-Length': len(data)-start, 'Content-Type': 'application/zip'}
    if etag:
        headers['ETag'] = etag
    if status == 206:
        headers['Content-Range'] = f'bytes {start}-{len(data)-1}/{len(data)}'
    return Response(data[start:], status=status, headers=headers, fail_after=fail_after)


def transfer(destination, opener, **kwargs):
    options = dict(progress=None, retry_delay=0, sleeper=lambda value: None, chunk_size=1024)
    options.update(kwargs)
    return download_archive('https://example.invalid/Scene_02.zip', destination, opener=opener, **options)


def test_reset_preserves_bytes_then_resumes_with_if_range_and_fresh_url(tmp_path):
    data, calls, urls = zip_bytes(), [], []
    destination = tmp_path/'Scene_02.zip'
    def opener(request, timeout):
        calls.append(request)
        return served(request, data, fail_after=4096 if len(calls) == 1 else None)
    def fresh():
        urls.append(1)
        return f'https://example.invalid/Scene_02.zip?temporary={len(urls)}'
    transfer(destination, opener, url_provider=fresh)
    assert destination.read_bytes() == data
    assert len(calls) == len(urls) == 2
    assert calls[1].get_header('Range') == 'bytes=4096-'
    assert calls[1].get_header('If-range') == '"stable"'
    assert not (tmp_path/'Scene_02.zip.part').exists()
    assert not (tmp_path/'Scene_02.zip.part.meta.json').exists()


def test_legacy_partial_is_reused_after_overlap_verification(tmp_path):
    data, calls = zip_bytes(), []
    destination = tmp_path/'Scene_02.zip'
    partial = tmp_path/'Scene_02.zip.part'
    partial.write_bytes(data[:90000])
    def opener(request, timeout):
        calls.append(request)
        return served(request, data)
    transfer(destination, opener)
    assert destination.read_bytes() == data
    assert calls[0].get_header('Range') == f'bytes={90000-65536}-'
    assert calls[0].get_header('If-range') is None


def test_server_ignoring_range_restarts_instead_of_appending(tmp_path):
    data = zip_bytes()
    destination = tmp_path/'Scene_02.zip'
    (tmp_path/'Scene_02.zip.part').write_bytes(data[:10000])
    transfer(destination, lambda request, timeout: served(request, data, ignore_range=True))
    assert destination.read_bytes() == data


def test_short_response_is_retried_not_published(tmp_path):
    data, calls = zip_bytes(), []
    destination = tmp_path/'Scene_02.zip'
    def opener(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            return Response(data[:5000], headers={'Content-Length': len(data), 'ETag': '"stable"'})
        return served(request, data)
    transfer(destination, opener)
    assert calls[1].get_header('Range') == 'bytes=5000-'
    assert destination.read_bytes() == data


def test_incorrect_range_preserves_existing_partial(tmp_path):
    data = zip_bytes()
    destination, partial = tmp_path/'Scene_02.zip', tmp_path/'Scene_02.zip.part'
    partial.write_bytes(data[:90000])
    def opener(request, timeout):
        response = served(request, data)
        response.headers.replace_header('Content-Range', f'bytes 1-{len(data)-1}/{len(data)}')
        return response
    with pytest.raises(ValueError, match='different byte range'):
        transfer(destination, opener)
    assert partial.read_bytes() == data[:90000] and not destination.exists()


def test_legacy_overlap_mismatch_does_not_destroy_saved_data(tmp_path):
    data = zip_bytes()
    destination, partial = tmp_path/'Scene_02.zip', tmp_path/'Scene_02.zip.part'
    original = bytearray(data[:90000])
    original[-10] ^= 1
    partial.write_bytes(original)
    with pytest.raises(ValueError, match='differs from the remote'):
        transfer(destination, lambda request, timeout: served(request, data))
    assert partial.read_bytes() == original


def test_exhausted_retries_leave_partial_and_no_tokens_in_checkpoint(tmp_path):
    data = zip_bytes()
    destination = tmp_path/'Scene_02.zip'
    def opener(request, timeout):
        return served(request, data, fail_after=2048)
    with pytest.raises(RuntimeError, match='Rerun the same cell'):
        transfer(destination, opener, max_attempts=2)
    partial = tmp_path/'Scene_02.zip.part'
    assert partial.read_bytes() == data[:4096]
    state = json.loads((tmp_path/'Scene_02.zip.part.meta.json').read_text())
    assert set(state) == {'version', 'total', 'etag', 'last_modified'}
    assert 'https' not in json.dumps(state)


def test_existing_valid_archive_does_not_contact_server(tmp_path):
    destination = tmp_path/'Scene_02.zip'
    destination.write_bytes(zip_bytes())
    def forbidden(*args, **kwargs):
        pytest.fail('Complete archive must be retained without a network request')
    assert transfer(destination, forbidden, url_provider=forbidden) == destination


def test_crc_failure_never_publishes_archive(tmp_path):
    data = bytearray(zip_bytes())
    data[100] ^= 1
    destination = tmp_path/'Scene_02.zip'
    with pytest.raises(ValueError, match='integrity check'):
        transfer(destination, lambda request, timeout: served(request, bytes(data)))
    assert not destination.exists()
    assert (tmp_path/'Scene_02.zip.part').exists()


def test_archive_budget_is_checked_before_writing(tmp_path):
    destination = tmp_path/'Scene_02.zip'
    with pytest.raises(ValueError, match='byte budget'):
        transfer(destination, lambda request, timeout: served(request, zip_bytes()), max_bytes=1000)
    assert not destination.exists()
    assert not (tmp_path/'Scene_02.zip.part').exists()


def test_transient_http_error_is_retried(tmp_path):
    data, calls = zip_bytes(), []
    destination = tmp_path/'Scene_02.zip'
    def opener(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 503, 'synthetic busy server', Message(), None)
        return served(request, data)
    transfer(destination, opener)
    assert destination.read_bytes() == data and len(calls) == 2


def test_expired_url_gets_one_refresh(tmp_path):
    data, calls, refreshed = zip_bytes(), [], []
    def opener(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 403, 'expired', Message(), None)
        return served(request, data)
    def fresh():
        refreshed.append(1)
        return 'https://example.invalid/signed'
    transfer(tmp_path/'Scene_02.zip', opener, url_provider=fresh)
    assert len(refreshed) == len(calls) == 2


def test_fully_received_partial_can_finalize_without_new_url(tmp_path):
    data = zip_bytes()
    destination = tmp_path/'Scene_02.zip'
    (tmp_path/'Scene_02.zip.part').write_bytes(data)
    (tmp_path/'Scene_02.zip.part.meta.json').write_text(json.dumps(dict(total=len(data), etag='"stable"')))
    def forbidden(*args, **kwargs):
        pytest.fail('Completed partial needs only local ZIP validation')
    transfer(destination, forbidden, url_provider=forbidden)
    assert destination.read_bytes() == data
