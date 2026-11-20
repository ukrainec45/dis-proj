"""Pinned Jev Choice adapter. Network calls require an explicit finite budget.

    API: https://docs.typesafe.ai/primitives/choice
    Images: https://docs.typesafe.ai/images
"""
import base64
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .artifacts import canonical, digest, load_verified, save_verified, write_once
from .contracts import CLASS_NAMES, PROTOCOL_ID, probabilities
from .features import ALL_FEATURES

ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
PROMPT_VERSION = 'preflight-choice-v1'
DESCRIPTIONS = (
    'The fixed localizer rejects the correction; retain the prior position.',
    'Accepted; delta < -20 metres.', 'Accepted; -20 <= delta < -5 metres.',
    'Accepted; -5 <= delta < -1 metres.', 'Accepted; -1 <= delta < 1 metres.',
    'Accepted; 1 <= delta < 5 metres.', 'Accepted; 5 <= delta < 20 metres.',
    'Accepted; delta >= 20 metres.')
INSTRUCTIONS = (
    'Predict the outcome of a future CAMP retrieval + RoMa matching + PnP localization attempt. '
    'This is pre-flight prediction: the future camera image and true position are unknown. '
    'delta = horizontal error before minus horizontal error of the accepted map position, '
    'both against the same ground truth. An accepted pose can worsen the error. '
    'Rejection is distinct from an accepted pose with near-zero delta. '
    'The map context is an approximate local-plane footprint expanded by 3 major-axis sigmas. '
    'The simulated prior is Gaussian; fog_extreme transmission is 0.1; smoke_extreme is '
    'synthetic sRGB mixing with coverage 0.9 and transmission 0.05. These are experimental '
    'conditions, not measured weather. Use all supplied localizer settings and map features. '
    'Return probabilities over the eight mutually exclusive outcomes.')


def build_request(features, localizer_settings, model='jev-1.14.0', map_rgb=None):
    if not re.fullmatch(r'jev-\d+\.\d+\.\d+', model):
        raise ValueError('Pin a numeric Jev version; latest aliases are forbidden')
    if set(features) != set(ALL_FEATURES) or not np.isfinite(list(features.values())).all():
        raise ValueError('Jev accepts only the frozen preflight feature allowlist')
    # The localizer contract is also frozen by campaign provenance, never trial diagnostics.
    allowed = ('top_k', 'min_matches', 'min_inliers', 'min_inlier_ratio', 'max_median_reprojection_px',
               'pnp_reprojection_px', 'map_alignment', 'reference_mode', 'match_resize_ratio')
    if set(localizer_settings) != set(allowed):
        raise ValueError('Expected fixed localizer settings only')
    state = dict(features=features, localizer=localizer_settings,
                 map_context='Unavailable in numeric-only candidate.')
    request = dict(model=model, state=state, questions={'outcome': dict(type='choice',
                   instructions=INSTRUCTIONS, criteria=dict(zip(CLASS_NAMES, DESCRIPTIONS)))})
    if map_rgb is not None:
        image = np.asarray(map_rgb)
        if image.dtype != np.uint8 or image.shape != (256, 256, 3):
            raise ValueError('Expected the 256 x 256 RGB map context, never a camera frame')
        ok, data = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError('Map image encoding failed')
        reference = 'correction_map_context_7f136f3b'
        request['assets'] = {reference: dict(type='image', format='png',
                                              data_base64=base64.b64encode(data).decode())}
        state['map_context'] = reference
    return request


def parse_response(response, model):
    if response.get('model') != model:
        raise ValueError('Server returned another model version')
    answer = response['answers']['outcome']
    if answer.get('type') != 'choice' or set(answer['probabilities']) != set(CLASS_NAMES):
        raise ValueError('Unexpected Jev Choice schema')
    p = probabilities([[answer['probabilities'][k] for k in CLASS_NAMES]])[0]
    if answer.get('choice') not in CLASS_NAMES:
        raise ValueError('Unknown choice')
    return p


def http_transport(body, key, images):
    headers = {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}
    if images:
        headers['TypeSafe-Danger-Zone-Images'] = 'true'
    req = urllib.request.Request(ENDPOINT, data=canonical(body).encode(), headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


class JevClient:
    """One caller per cache. Budgets count failures and previous reservations.

    Exact request+prompt hashes address the cache. No automatic retries; an
    uncertain/failed request consumes a slot, preventing hidden extra billing.
    """
    def __init__(self, cache_dir, *, max_requests=0, transport=None):
        if type(max_requests) is not int or max_requests < 0:
            raise ValueError('Nonnegative explicit max_requests required')
        self.cache = Path(cache_dir)
        self.max_requests = max_requests
        self.transport = transport or http_transport

    def predict(self, request):
        identity = digest(dict(protocol=PROTOCOL_ID, prompt=PROMPT_VERSION, request=request))
        cached = self.cache / (identity+'.json')
        if cached.exists():
            saved = load_verified(cached)
            if saved['request'] != request:
                raise ValueError('Jev request cache mismatch')
            return parse_response(saved['response'], request['model'])
        if self.max_requests == 0:
            raise RuntimeError('Jev cache miss: network disabled (max_requests=0)')
        key = os.environ.get('TYPESAFE_API_KEY')
        if not key and self.transport is http_transport:
            raise RuntimeError('Set TYPESAFE_API_KEY outside the notebook source')
        self.cache.mkdir(parents=True, exist_ok=True)
        lock = self.cache / '.request.lock'
        try:
            # Exclusive lock spans reservation+request+cache. Stale locks require inspection.
            with lock.open('x'):
                pass
        except FileExistsError as exc:
            raise RuntimeError('Jev cache locked; inspect interrupted/concurrent request') from exc
        try:
            used = len(list(self.cache.glob('reservation-*.json')))
            if used >= self.max_requests:
                raise RuntimeError('Explicit Jev request budget exhausted')
            reservation = self.cache / f'reservation-{used:06d}.json'
            write_once(reservation, dict(request_id=identity, request=request, prompt_version=PROMPT_VERSION))
            started = time.perf_counter()
            response = self.transport(request, key, 'assets' in request)
            save_verified(cached, dict(request=request, response=response, prompt_version=PROMPT_VERSION,
                                      elapsed_s=time.perf_counter()-started))
            # Preserve even an invalid server reply for audit; a later cache hit
            # repeats validation rather than spending another request silently.
            return parse_response(response, request['model'])
        finally:
            lock.unlink()
