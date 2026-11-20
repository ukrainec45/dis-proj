"""Map-context-only CAMP descriptors, using the pinned gallery encoder semantics."""
import time
from pathlib import Path

import cv2
import numpy as np

from .artifacts import digest, save_verified, load_verified


class CampEncoder:
    def __init__(self, backend):
        self.backend = backend
        self.identity = dict(version='camp-map-context-v1', size=backend.config['retrieval_size'],
            checkpoint=backend.identity['camp_checkpoint_sha256'],
            state=backend.identity['camp_state_sha256'], upstream=backend.identity['upstream_commit'],
            normalization=bool(backend.upstream_config['RETRIEVAL_FEATURE_NORM']),
            input='prior-centred RGB map context; no future camera image')

    def __call__(self, map_rgb):
        from PIL import Image
        import torch
        import torch.nn.functional as functional
        from Retrieval_Models.feature_extract import feature_fusion_all
        image = cv2.resize(map_rgb, (self.identity['size'], self.identity['size']), interpolation=cv2.INTER_LINEAR)
        b = self.backend
        tensor = b.methods['img_transform'](Image.fromarray(image)).unsqueeze(0).to('cuda')
        b.methods['retrieval_model'].eval()
        with torch.inference_mode(), torch.autocast(device_type='cuda'):
            result = feature_fusion_all(b.methods, b.methods['retrieval_model'](tensor))
            if self.identity['normalization']:
                result = functional.normalize(result, dim=-1)
        return result.detach().float().cpu().numpy().reshape(-1)


def embedding_key(identity, patch):
    import hashlib
    return digest(dict(encoder=identity, shape=list(patch.shape), pixels=hashlib.sha256(patch.tobytes()).hexdigest()))


class CachedCampEncoder:
    """CPU-only exact-cache lookup. Unseen map contexts need a real pinned encoder."""
    def __init__(self, cache, identity):
        self.cache, self.identity = Path(cache), identity

    def __call__(self, map_rgb):
        key = embedding_key(self.identity, map_rgb)
        path = self.cache/(key+'.json')
        if not path.exists():
            raise ValueError('map_descriptor_not_cached; encode this map context with the pinned CAMP model')
        saved = load_verified(path)
        if saved['encoder'] != self.identity or saved['cache_key'] != key:
            raise ValueError('Cached encoder identity mismatch')
        return np.asarray(saved['vector'], float)


def cached_embedding(encoder, patch, cache):
    key = embedding_key(encoder.identity, patch)
    path = Path(cache)/(key+'.json')
    started = time.perf_counter()
    hit = path.exists()
    if hit:
        saved = load_verified(path)
        if saved['encoder'] != encoder.identity or saved['cache_key'] != key:
            raise ValueError('CAMP embedding identity mismatch')
        vector = np.asarray(saved['vector'], float)
    else:
        vector = np.asarray(encoder(patch), float)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError('Invalid CAMP map descriptor')
        save_verified(path, dict(cache_key=key, encoder=encoder.identity, vector=vector.tolist()))
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError('Invalid cached CAMP map descriptor')
    return vector.tolist(), dict(cache_key=key, cache_hit=hit, elapsed_s=time.perf_counter()-started)
