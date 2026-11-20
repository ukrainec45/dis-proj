import base64
import json
from urllib.error import HTTPError

import pytest

from scripts.correction_models.download_access import browser_scene_links, scene_links


def sharepoint_catalog(path, *, matching_drive=True, with_token=True):
    prefix = 'https://onedrive.live.com/_api/v2.0/drives/fixture'
    item_prefix = prefix if matching_drive else prefix+'-other'
    schema = {'.driveUrl': prefix}
    if with_token:
        schema['.driveAccessToken'] = 'access_token=temporary-test-token'
    path.write_text(json.dumps(dict(ListSchema=schema, ListData=dict(Row=[dict(
        FileLeafRef='Scene_24.zip', File_x0020_Size='326886104',
        **{'.spItemUrl': item_prefix+'/items/fixture-item?version=Published'})]))))


def test_sharepoint_rows_are_catalog_entries_not_download_urls(tmp_path):
    path = tmp_path/'catalog.json'
    sharepoint_catalog(path)
    catalog = browser_scene_links(path)
    assert set(catalog) == {'Scene_24'}
    assert catalog['Scene_24']['requires_metadata_lookup']
    assert 'url' not in catalog['Scene_24']
    assert 'temporary-test-token' not in json.dumps(catalog)


def test_matching_drive_token_resolves_the_server_provided_download_url(tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    path = tmp_path/'catalog.json'
    sharepoint_catalog(path)
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            assert request.get_header('Authorization') == 'Bearer temporary-test-token'
            body = json.dumps(dict(name='Scene_24.zip', **{
                '@content.downloadUrl': 'https://my.microsoftpersonalcontent.com/file?tempauth=temporary-test',
                '@content.downloadUrlNoAuth': 'https://my.microsoftpersonalcontent.com/file'})).encode()
            return nullcontext(SimpleNamespace(read=lambda count: body))
    item = scene_links(['Scene_24'], browser_metadata=path, opener=Opener())['Scene_24']
    assert '?tempauth=' in item['url']
    assert len(calls) == 1
    assert 'temporary-test-token' not in json.dumps(item)


@pytest.mark.parametrize('matching_drive,with_token', [(False, True), (True, False)])
def test_token_is_not_used_for_another_drive_or_when_missing(tmp_path, matching_drive, with_token):
    path = tmp_path/'catalog.json'
    sharepoint_catalog(path, matching_drive=matching_drive, with_token=with_token)
    class Opener:
        def open(self, *args, **kwargs):
            pytest.fail('Unmatched credentials must never be sent')
    with pytest.raises(ValueError, match='matching temporary drive token'):
        scene_links(['Scene_24'], browser_metadata=path, opener=Opener())


def test_expired_metadata_token_has_sanitized_error(tmp_path):
    path = tmp_path/'catalog.json'
    sharepoint_catalog(path)
    class Opener:
        def open(self, *args, **kwargs):
            raise HTTPError('https://onedrive.live.com/', 401, 'denied', None, None)
    with pytest.raises(RuntimeError) as result:
        scene_links(['Scene_24'], browser_metadata=path, opener=Opener())
    assert 'HTTP 401' in str(result.value)
    assert 'temporary-test-token' not in str(result.value)


def test_authenticated_metadata_redirects_are_not_followed():
    from scripts.correction_models.download_access import _NoCredentialRedirect
    assert _NoCredentialRedirect().redirect_request(None, None, 302, '', {}, 'https://elsewhere.invalid/') is None


def entry(metadata, *, encoded=False, host='api.onedrive.com'):
    text = json.dumps(metadata)
    return dict(request=dict(url='https://'+host+'/v1.0/items',
        headers=[dict(name='Authorization', value='not-to-be-forwarded')]),
        response=dict(status=200, headers=[], content=dict(
            encoding='base64' if encoded else '',
            text=base64.b64encode(text.encode()).decode() if encoded else text)))


def test_parse_public_browser_metadata_without_copying_authentication(tmp_path):
    path = tmp_path/'onedrive.har'
    url = 'https://public.bn.files.1drv.com/file?temporary=one'
    path.write_text(json.dumps(dict(log=dict(entries=[entry(dict(value=[
        dict(name='Scene_01.zip', **{'@microsoft.graph.downloadUrl': url}),
        dict(name='Scene_02.zip', **{'@content.downloadUrl': 'https://public.bn.files.1drv.com/two'})]))]))))
    links = browser_scene_links(path)
    assert set(links) == {'Scene_01', 'Scene_02'}
    assert links['Scene_01'] == dict(name='Scene_01.zip', url=url)
    assert 'not-to-be-forwarded' not in json.dumps(links)


def test_base64_response_and_latest_url_wins(tmp_path):
    path = tmp_path/'onedrive.har'
    old = dict(name='Scene_01.zip', downloadUrl='https://public.bn.files.1drv.com/old')
    new = dict(name='Scene_01.zip', downloadUrl='https://public.bn.files.1drv.com/new')
    path.write_text(json.dumps(dict(log=dict(entries=[entry(old), entry(new, encoded=True)]))))
    assert browser_scene_links(path)['Scene_01']['url'].endswith('/new')


def test_plain_json_catalog_and_actual_zip_response_headers(tmp_path):
    path = tmp_path/'catalog.json'
    path.write_text(json.dumps(dict(name='Scene_01.zip', download_url='https://public.bn.files.1drv.com/one')))
    assert set(browser_scene_links(path)) == {'Scene_01'}
    response = dict(request=dict(url='https://public.bn.files.1drv.com/file'),
                    response=dict(status=200, content={}, headers=[dict(name='Content-Disposition',
                    value='attachment; filename="Scene_03.zip"')]))
    path.write_text(json.dumps(dict(log=dict(entries=[response]))))
    assert set(browser_scene_links(path)) == {'Scene_03'}


def test_names_or_sharing_pages_without_download_urls_are_not_guessed(tmp_path):
    path = tmp_path/'catalog.json'
    path.write_text(json.dumps([dict(name='Scene_01.zip', webUrl='https://onedrive.live.com/folder')]))
    with pytest.raises(ValueError, match='No explicit'):
        browser_scene_links(path)


def test_unrelated_hosts_and_error_responses_are_ignored(tmp_path):
    data = dict(name='Scene_01.zip', downloadUrl='https://public.bn.files.1drv.com/file')
    denied = entry(data)
    denied['response']['status'] = 401
    path = tmp_path/'onedrive.har'
    path.write_text(json.dumps(dict(log=dict(entries=[entry(data, host='unrelated.invalid'), denied]))))
    with pytest.raises(ValueError, match='No explicit'):
        browser_scene_links(path)


@pytest.mark.parametrize('code', [401, 403])
def test_api_access_denial_has_actionable_error_and_no_automatic_retry(monkeypatch, code):
    import scripts.correction_models.download_access as module
    calls = []
    def denied(*args):
        calls.append(1)
        raise HTTPError('https://api.onedrive.com/', code, 'denied', None, None)
    monkeypatch.setattr(module, 'official_scene_links', denied)
    with pytest.raises(RuntimeError, match='ONEDRIVE_BROWSER_METADATA'):
        scene_links()
    assert len(calls) == 1


def test_browser_path_never_requests_anonymous_api(tmp_path, monkeypatch):
    import scripts.correction_models.download_access as module
    def forbidden(*args):
        pytest.fail('Browser metadata must not call anonymous API')
    monkeypatch.setattr(module, 'official_scene_links', forbidden)
    path = tmp_path/'catalog.json'
    path.write_text(json.dumps(dict(name='Scene_01.zip', downloadUrl='https://public.bn.files.1drv.com/one')))
    assert set(scene_links(['Scene_01'], browser_metadata=path)) == {'Scene_01'}
    with pytest.raises(ValueError, match='missing scenes'):
        scene_links(['Scene_02'], browser_metadata=path)
