"""GT-free query contract and evaluation-only labels, in horizontal metres."""
from dataclasses import dataclass

import numpy as np
from pyproj import CRS

PROTOCOL_ID = 'correction-delta-v1'
EDGES = np.array([-20., -5., -1., 1., 5., 20.])
CLASS_NAMES = ('reject', 'harm_large', 'harm', 'harm_small', 'neutral',
               'gain_small', 'gain', 'gain_large')
CONDITIONS = ('clear', 'fog_extreme', 'smoke_extreme')


def finite_array(value, shape, name):
    a = np.array(value, dtype=float, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected finite shape {shape}')
    return a


@dataclass(frozen=True)
class CorrectionQuery:
    """Position and map use the same local ENU or projected metric frame.

    No image of the future observation, actual prior error or GT is accepted.
    map_id must identify a map whose scale/appearance domain has been checked.
    """
    position: object
    crs: str
    camera_to_enu: object
    K: object
    dist: object
    image_shape: tuple
    covariance_xy: object
    yaw_sigma_deg: float
    tilt_sigma_deg: float
    z_sigma_m: float
    scenario: str
    map_id: str

    def __post_init__(self):
        for name, shape in [('position', (3,)), ('camera_to_enu', (3, 3)),
                            ('K', (3, 3)), ('covariance_xy', (2, 2))]:
            object.__setattr__(self, name, finite_array(getattr(self, name), shape, name))
        r = self.camera_to_enu
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1):
            raise ValueError('camera_to_enu must be a proper rotation')
        c = self.covariance_xy
        if not np.allclose(c, c.T) or np.linalg.eigvalsh(c).min() <= 0:
            raise ValueError('covariance_xy must be symmetric positive definite')
        if self.crs != 'LOCAL_ENU':
            crs = CRS.from_user_input(self.crs)
            if not crs.is_projected or any(a.unit_conversion_factor != 1 for a in crs.axis_info[:2]):
                raise ValueError('Use a projected metric CRS or LOCAL_ENU')
        if len(self.image_shape) != 2 or any(int(v) != v or v < 2 for v in self.image_shape):
            raise ValueError('image_shape must be positive (height, width)')
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0 or not np.allclose(self.K[2], [0, 0, 1]):
            raise ValueError('Invalid intrinsics')
        d = np.array(self.dist, dtype=float, copy=True).reshape(-1)
        if d.size not in (4, 5, 8, 12, 14) or not np.isfinite(d).all():
            raise ValueError('Invalid OpenCV distortion coefficients')
        object.__setattr__(self, 'dist', d)
        for name in ('yaw_sigma_deg', 'tilt_sigma_deg', 'z_sigma_m'):
            v = float(getattr(self, name))
            if not np.isfinite(v) or v < 0:
                raise ValueError(f'Invalid {name}')
        if not self.map_id or not isinstance(self.scenario, str):
            raise ValueError('map_id and scenario required')


def label_outcome(prior_xyz, predicted_xyz, gt_xyz, accepted, *, technical_error=False):
    """Evaluation only. Rejection ignores any diagnostic rejected PnP position."""
    if not isinstance(accepted, (bool, np.bool_)):
        raise ValueError('accepted must be boolean')
    prior = finite_array(prior_xyz, (3,), 'prior')
    gt = finite_array(gt_xyz, (3,), 'ground_truth')
    if technical_error:
        return dict(class_index=None, delta_m=None, technical_error=True)
    before_vector = prior[:2] - gt[:2]
    after_vector = (finite_array(predicted_xyz, (3,), 'prediction')[:2] - gt[:2]
                    if accepted else before_vector.copy())
    before, after = float(np.linalg.norm(before_vector)), float(np.linalg.norm(after_vector))
    delta = before - after
    index = int(np.searchsorted(EDGES, delta, side='right') + 1) if accepted else 0
    return dict(class_index=index, class_name=CLASS_NAMES[index], delta_m=delta,
                error_before_m=before, error_after_m=after,
                error_before_xy_m=before_vector.tolist(), error_after_xy_m=after_vector.tolist(),
                accepted=bool(accepted), technical_error=False)


def probabilities(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 8 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Expected finite N x 8 probabilities in [0,1]')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-7, rtol=0):
        raise ValueError('Probabilities must sum to one')
    return p


def derived(value):
    p = probabilities(value)
    return dict(accept=p[:, 1:].sum(axis=1), improve=p[:, 5:].sum(axis=1),
                worsen=p[:, 1:4].sum(axis=1), severe_worsen=p[:, 1],
                utility=p[:, 5:].sum(axis=1)-p[:, 1:4].sum(axis=1))
