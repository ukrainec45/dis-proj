"""Planning-facing inference: map-only, no future camera frame and no route risk claim."""
from pathlib import Path

import numpy as np

from .artifacts import digest, load_verified, assert_frozen_environment
from .contracts import PROTOCOL_ID, CLASS_NAMES, derived
from .distribution_metrics import weighted_quantile
from .features import extract_features
from .models import FrozenModel


class Predictor:
    def __init__(self, model, reference, *, encoder=None, audited_map_ids=(), calibrated=True):
        self.model, self.reference, self.encoder = model, reference, encoder
        self.audited_maps = set(audited_map_ids)
        self.calibrated = calibrated

    def predict_correction(self, query):
        reply = dict(protocol_id=PROTOCOL_ID, model_id=self.model.model_id,
            localizer_id=digest(self.model.artifact['localizer_settings']),
            support_status='unsupported', probabilities=None,
            research_only=True, flight_validation_passed=False, delivery_probability=None)
        map_id = getattr(self.reference, 'map_id', None)
        if map_id is None:
            map_id = digest(self.reference.identity)
        if query.map_id != map_id:
            return dict(reply, reason='query_map_identity_mismatch')
        if map_id not in self.audited_maps:
            return dict(reply, reason='unverified_map_domain')
        if not np.allclose(query.covariance_xy, np.eye(2)*np.trace(query.covariance_xy)/2):
            return dict(reply, reason='anisotropic_prior_not_validated')
        try:
            reference = self.reference.window(query) if hasattr(self.reference, 'window') else self.reference
            features, patch, _ = extract_features(query, reference)
        except ValueError as exc:
            return dict(reply, reason=str(exc))
        a = self.model.artifact
        camp = []
        if a['embedding_dim']:
            if self.encoder is None or self.encoder.identity != a['encoder']:
                return dict(reply, reason='pinned_camp_encoder_required')
            try:
                camp = np.asarray(self.encoder(patch), float).tolist()
            except ValueError as exc:
                return dict(reply, reason=str(exc))
        inputs = [dict(features=features, camp=camp)]
        if self.model.support(inputs)[0]:
            return dict(reply, reason='outside_training_feature_ranges')
        joint = self.model.predict_joint(inputs, self.calibrated)
        p, w = joint['probabilities'][0], joint['weights'][0]
        classes = np.asarray(joint['bank']['class_index'])
        accepted = classes > 0
        accepted_w = w[accepted]
        xy = np.asarray(joint['bank']['error_after_xy_m'])[accepted]
        unknown = float(joint['unresolved_mass'][0])
        unknown_accepted = float(p[[c for c in joint['missing_classes'] if c > 0]].sum())
        quantiles = None
        if accepted_w.sum() > 0:
            radius = np.linalg.norm(xy, axis=1)
            quantiles = {str(q): weighted_quantile(radius, accepted_w, q) for q in (.5, .9)}
        return dict(reply, support_status='within_audited_map_and_marginal_ranges',
            support_caveat='Marginal ranges do not establish joint-domain support, real-weather calibration or flight safety.',
            probabilities=dict(zip(CLASS_NAMES, p.tolist())),
            **{k: float(v[0]) for k, v in derived(p.reshape(1, 8)).items()},
            accepted_error_distribution=dict(xy_m=xy.tolist(), weights=accepted_w.tolist(),
                error_before_xy_m=np.asarray(joint['bank']['error_before_xy_m'])[accepted].tolist(),
                class_index=classes[accepted].tolist(), weights_are='unconditional probability mass',
                unresolved_accepted_mass=unknown_accepted),
            rejection=dict(probability=float(p[0]), action='retain incoming position estimate'),
            unresolved_outcome_mass=unknown, missing_training_classes=joint['missing_classes'],
            accepted_error_upper_bounds_m=quantiles if unknown_accepted == 0. else None,
            resolved_component_upper_bounds_m=quantiles,
            bounds_caveat='Resolved-component bounds condition on observed accepted outcomes; they exclude unresolved tail mass.')

    def predict_batch(self, queries):
        return [self.predict_correction(query) for query in queries]


def load_predictor(run_dir, reference, *, encoder=None):
    root = Path(run_dir)
    run = load_verified(root/'frozen_run.json')
    assert_frozen_environment(run)
    name, variant = run['selected'].split(':')
    model = FrozenModel.load(root/'models'/(name+'.json'))
    if model.model_id != run['model_ids'][name]:
        raise ValueError('Frozen selected model changed')
    return Predictor(model, reference, encoder=encoder, audited_map_ids=run['map_ids'], calibrated=variant == 'calibrated')
