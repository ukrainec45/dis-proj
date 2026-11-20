"""Use authorized browser metadata when anonymous OneDrive API access is denied.

Explicit HTTPS download URLs are read, or obtained from the server-provided item
metadata endpoint using its matching temporary drive token. Account cookies and
captured request headers are never copied. Tokens are never printed or persisted.
"""
import base64
from email.message import Message
import json
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, build_opener, HTTPRedirectHandler

from scripts.visibility_localization.colab import official_scene_links

ZIP_NAME = re.compile(r'^(Scene_\d{2})(?:[._-].*)?\.zip$', re.IGNORECASE)
URL_KEYS = ('@microsoft.graph.downloadurl', '@content.downloadurl', 'downloadurl', 'download_url')


def _onedrive_url(value):
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    host = (parsed.hostname or '').lower()
    return parsed.scheme == 'https' and parsed.username is None and parsed.password is None and (
        host in ('onedrive.live.com', '1drv.ms', 'api.onedrive.com') or
        any(host.endswith('.'+domain) for domain in ('1drv.com', 'onedrive.com', 'sharepoint.com', 'sharepoint.cn',
                                                    'microsoftpersonalcontent.com')))


def _browser_payloads(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict) or 'log' not in payload:
        return [payload]
    result = []
    for entry in payload['log'].get('entries', []):
        request = entry.get('request', {})
        if not _onedrive_url(request.get('url')):
            continue
        response = entry.get('response', {})
        if response.get('status', 0) >= 400:
            continue
        content = response.get('content', {})
        text = content.get('text', '')
        if content.get('encoding') == 'base64':
            try:
                text = base64.b64decode(text, validate=True).decode('utf-8-sig')
            except (ValueError, UnicodeDecodeError):
                text = ''
        if isinstance(text, str) and text:
            try:
                result.append(json.loads(text.lstrip('\ufeff')))
            except ValueError:
                pass
        if response.get('status') in (200, 206):
            headers = {h['name'].lower(): h['value'] for h in response.get('headers', [])}
            message = Message()
            message['content-disposition'] = headers.get('content-disposition', '')
            result.append(dict(name=message.get_filename(), downloadUrl=request.get('url')))
    return result


def browser_scene_links(path):
    """Read HAR-with-response-content or downloaded/copied metadata JSON locally.

    HAR headers are ignored except Content-Disposition for identifying an actual
    ZIP response. Last observed URL for the same filename wins (URLs can expire).
    Input and URLs are never printed or persisted by this function.
    """
    links = {}

    def add(name, url):
        if not isinstance(name, str) or not _onedrive_url(url):
            return
        match = ZIP_NAME.fullmatch(name)
        if match is None:
            return
        scene = 'Scene_'+match.group(1).split('_')[1]
        if scene in links and links[scene]['name'] != name:
            raise ValueError(f'Multiple different ZIP filenames for {scene}; select the intended archive explicitly')
        links[scene] = dict(name=name, url=url)

    def walk(value):
        if isinstance(value, dict):
            fields = {k.lower(): v for k, v in value.items() if isinstance(k, str)}
            name = fields.get('name') or fields.get('filename') or fields.get('fileleafref')
            for key in URL_KEYS:
                if key in fields:
                    add(name, fields[key])
            item_url = fields.get('.spitemurl')
            match = ZIP_NAME.fullmatch(name) if isinstance(name, str) else None
            if match is not None and _onedrive_url(item_url):
                scene = 'Scene_'+match.group(1).split('_')[1]
                if scene not in links or 'url' not in links[scene]:
                    links[scene] = dict(name=name, metadata_url=item_url,
                        size=fields.get('file_x0020_size'), requires_metadata_lookup=True)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for payload in _browser_payloads(path):
        walk(payload)
    if not links:
        raise ValueError('No explicit OneDrive ZIP download URLs were found. Export HAR with response content '
            'after loading the folder listing, or provide actual ZIP URLs in DIRECT_URLS. '
            'A list of filenames or a folder sharing URL is insufficient. Do not supply account cookies or passwords.')
    return links


class _NoCredentialRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the temporary bearer token to a redirected destination.
        return None


def _schemas(payload):
    if isinstance(payload, dict):
        schema = payload.get('ListSchema')
        if isinstance(schema, dict):
            yield schema
        for value in payload.values():
            if isinstance(value, (dict, list)):
                yield from _schemas(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _schemas(value)


def resolve_item_download(item, browser_metadata, *, opener=None):
    """GET only the matching drive item's metadata. No archive bytes requested."""
    item_url = item['metadata_url']
    parsed_item = urlparse(item_url)
    token = None
    for payload in _browser_payloads(browser_metadata):
        for schema in _schemas(payload):
            for url_key, token_key in (('.driveUrl', '.driveAccessToken'), ('.driveUrlV21', '.driveAccessTokenV21')):
                drive_url = schema.get(url_key, '')
                if not _onedrive_url(drive_url):
                    continue
                drive = urlparse(drive_url)
                same_origin = (drive.scheme, drive.hostname, drive.port or 443) == (
                    parsed_item.scheme, parsed_item.hostname, parsed_item.port or 443)
                if same_origin and parsed_item.path.startswith(drive.path.rstrip('/')+'/items/'):
                    token = (parse_qs(schema.get(token_key, '')).get('access_token') or [None])[0]
                    if token:
                        break
            if token:
                break
        if token:
            break
    if not token:
        raise ValueError('This catalog has item metadata addresses but no matching temporary drive token. '
                         'Save the full RenderListDataAsStream response including ListSchema, or use direct ZIP URLs.')
    request = Request(item_url, headers={'Authorization': 'Bearer '+token, 'Accept': 'application/json'})
    opener = opener or build_opener(_NoCredentialRedirect())
    try:
        with opener.open(request, timeout=30) as response:
            body = response.read(2_000_001)
    except HTTPError as exc:
        raise RuntimeError(f'OneDrive item metadata request was rejected (HTTP {exc.code}). '
                           'The temporary access may have expired or may not be usable from this server. '
                           'Obtain a fresh catalog in the browser. No ZIP bytes were requested.') from None
    if len(body) > 2_000_000:
        raise ValueError('Unexpectedly large item metadata response')
    try:
        metadata = json.loads(body)
    except ValueError:
        raise ValueError('OneDrive returned a non-JSON metadata response') from None
    if metadata.get('name') != item['name']:
        raise ValueError('OneDrive returned metadata for a different filename')
    fields = {k.lower(): v for k, v in metadata.items()}
    for key in URL_KEYS:
        url = fields.get(key)
        if _onedrive_url(url):
            return dict(item, url=url, requires_metadata_lookup=False)
    raise ValueError('Item metadata did not include an explicit permitted ZIP download URL')


def scene_links(scenes=None, *, browser_metadata=None, opener=None):
    if browser_metadata is not None:
        links = browser_scene_links(browser_metadata)
        if scenes is not None:
            missing = set(scenes)-set(links)
            if missing:
                raise ValueError(f'Browser metadata is missing scenes: {sorted(missing)}')
            return {scene: (links[scene] if 'url' in links[scene] else
                           resolve_item_download(links[scene], browser_metadata, opener=opener)) for scene in scenes}
        return links
    try:
        return official_scene_links(scenes)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                f'OneDrive rejected anonymous API access (HTTP {exc.code}). '
                'Retrying unchanged will not help. Open the public folder in your browser, '
                'export a sanitized HAR with response content, and set ONEDRIVE_BROWSER_METADATA '
                'to its path on Drive; alternatively supply explicit DIRECT_URLS. '
                'A personal OneDrive account is not required for the browser-metadata path. '
                'No archive has been transferred by this catalog request.') from exc
        raise
