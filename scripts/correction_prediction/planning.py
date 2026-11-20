"""Batch queries and adapters for existing routes; never modifies MOA*."""
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer

from .artifacts import digest, load_verified, save_verified, assert_frozen_environment
from .contracts import PROTOCOL_ID, CLASS_NAMES, CorrectionQuery, derived
from .features import extract_features
from .jev import build_request, PROMPT_VERSION
from .models import temperature_scale


class Predictor:
    def __init__(self, model, reference, *, allowed_map_ids=(), jev_client=None, localizer_settings=None,
                 calibrated=True):
        self.model, self.reference = model, reference
        self.allowed_maps = set(allowed_map_ids)
        self.jev_client, self.localizer_settings = jev_client, localizer_settings
        self.calibrated = calibrated

    def predict_correction(self, query: CorrectionQuery):
        a = self.model.artifact if hasattr(self.model, 'artifact') else self.model
        reply = dict(protocol_id=PROTOCOL_ID, model_id=digest(a), probabilities=None,
                     support_status='unsupported', research_only=True, validation_passed=False)
        reference_id = (self.reference.map_id if hasattr(self.reference, 'map_id')
                        else digest(self.reference.identity))
        if query.map_id != reference_id:
            return dict(reply, reason='query_map_identity_mismatch')
        if query.map_id not in self.allowed_maps:
            return dict(reply, reason='unverified_map_domain')
        try:
            features, patch, _ = extract_features(query, self.reference)
        except ValueError as exc:
            return dict(reply, reason=str(exc))
        x = np.array([features[k] for k in a['features']])
        if ((x < np.asarray(a['minimum'])) | (x > np.asarray(a['maximum']))).any():
            return dict(reply, reason='outside_training_feature_ranges')
        if hasattr(self.model, 'predict'):
            p = self.model.predict([features], self.calibrated)
        else:
            if a['protocol_id'] != PROTOCOL_ID or a['prompt_version'] != PROMPT_VERSION or self.jev_client is None:
                raise ValueError('Frozen Jev protocol/client required')
            body = build_request(features, a['localizer_settings'], a['model'], patch if a['with_image'] else None)
            p = np.asarray([self.jev_client.predict(body)])
            if self.calibrated:
                p = temperature_scale(p, a['temperature'])
        d = derived(p)
        return dict(reply, probabilities=dict(zip(CLASS_NAMES, p[0].tolist())),
                    support_status='within_audited_map_and_marginal_ranges',
                    support_caveat='Range checks do not prove joint-domain support or calibrated flight risk.',
                    missing_training_classes=a.get('missing_classes', []),
                    **{k: float(v[0]) for k, v in d.items()})

    def predict_batch(self, queries):
        return [self.predict_correction(q) for q in queries]


def load_predictor(run_dir, reference, *, candidate=None, jev_client=None):
    from .models import FrozenModel
    root = Path(run_dir)
    run = load_verified(root/'frozen_run.json')
    assert_frozen_environment(run)
    chosen = candidate or run['selected']
    if chosen not in run['candidate_order']:
        raise ValueError('Candidate was not frozen')
    name, variant = chosen.split(':')
    path = root/'models'/(name+'.json')
    model = load_verified(path) if name.startswith('jev_') else FrozenModel.load(path)
    a = model.artifact if hasattr(model, 'artifact') else model
    if digest(a) != run['model_ids'][name]:
        raise ValueError('Model changed after freeze')
    return Predictor(model, reference, allowed_map_ids=run['map_ids'], jev_client=jev_client,
                     localizer_settings=run['localizer_settings'], calibrated=variant == 'calibrated')


class RasterReference:
    """North-up 3-band uint8 RGB GeoTIFF and georeferenced elevation raster.

    Reject different encodings/rotated grids explicitly; no silent Sentinel
    scaling or transfer claim. Camera rotation is relative to grid ENU axes.
    """
    def __init__(self, image_path, dem_path):
        import rasterio
        from scripts.visibility_localization.core import file_hash
        self.image_path, self.dem_path = Path(image_path), Path(dem_path)
        with rasterio.open(image_path) as src:
            crs = CRS.from_user_input(src.crs)
            if not crs.is_projected or any(a.unit_conversion_factor != 1 for a in crs.axis_info[:2]):
                raise ValueError('Map CRS must be projected in metres')
            if src.count < 3 or any(t != 'uint8' for t in src.dtypes[:3]):
                raise ValueError('Explicit uint8 RGB map preprocessing required')
            t = src.transform
            if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
                raise ValueError('Only north-up metric raster grids supported')
            self.crs, self.transform = crs.to_string(), t
            self.bgr = np.moveaxis(src.read([3, 2, 1]), 0, -1)
            self.valid_pixels = src.dataset_mask() > 0
            self.resolution = np.array([t.a, -t.e])
        with rasterio.open(dem_path) as dem:
            if dem.crs is None:
                raise ValueError('DEM CRS required')
        self.map_id = digest(dict(rgb=file_hash(image_path), dem=file_hash(dem_path)))

    def pixels_to_enu(self, pixels):
        a = np.asarray(pixels)
        return np.column_stack([self.transform.c+(a[:, 0]+.5)*self.transform.a,
                                self.transform.f+(a[:, 1]+.5)*self.transform.e])

    def enu_to_pixels(self, xy):
        xy = np.asarray(xy)
        return np.column_stack([(xy[:, 0]-self.transform.c)/self.transform.a-.5,
                                (xy[:, 1]-self.transform.f)/self.transform.e-.5])

    def plane_at(self, xy):
        import rasterio
        with rasterio.open(self.dem_path) as dem:
            transform = Transformer.from_crs(self.crs, dem.crs, always_xy=True)
            x, y = transform.transform(*xy)
            sample = next(dem.sample([(x, y)], masked=True))
            if np.ma.is_masked(sample[0]) or not np.isfinite(sample[0]):
                raise ValueError('No valid DEM height at prior')
            return float(sample[0])


def timed_route_queries(path_xy_m, edge_times_s, query_factory, interval_s):
    """Sample t=0, interval, ... <= duration; frequency independent of cell size.

    query_factory(xy, time_s) supplies the planned camera, covariance and map.
    This function does not invent an odometry/covariance propagation model.
    """
    path = np.asarray(path_xy_m, dtype=float)
    times = np.asarray(edge_times_s, dtype=float)
    if (path.ndim != 2 or path.shape[1] != 2 or not len(path) or times.shape != (len(path)-1,)
            or not np.isfinite(path).all() or not np.isfinite(times).all() or (times <= 0).any()
            or not np.isfinite(interval_s) or interval_s <= 0):
        raise ValueError('Finite path and positive edge times/interval required')
    cumulative = np.r_[0., np.cumsum(times)]
    scheduled = np.arange(0, cumulative[-1]+1e-9, interval_s)
    queries = []
    for t in scheduled:
        i = min(np.searchsorted(cumulative, t, side='right')-1, len(path)-2)
        point = path[0] if len(path) == 1 else path[i]+(path[i+1]-path[i])*(t-cumulative[i])/times[i]
        queries.append(query_factory(point, float(t)))
    return queries


def saved_route_queries(route_file, route_id, cost_map, query_factory, interval_s, *, grid_to_metric):
    """Read existing MOA result; caller explicitly defines grid georeferencing."""
    from scripts.moa.route_results import load_route_results, route_path
    path = route_path(load_route_results(route_file), route_id)
    coordinates = np.asarray([grid_to_metric(tuple(cell)) for cell in path])
    # Use the same edge time function as the planner, through its cost function.
    from scripts.moa.edge_costs import edge_objectives
    times = []
    for start, end in zip(path, path[1:]):
        cost = edge_objectives(tuple(start), tuple(end), cost_map, cost_map.v_air, cost_map.v_max)
        if cost is None:
            raise ValueError('Saved route has infeasible edge under current cost map')
        times.append(float(cost[0]))
    return timed_route_queries(coordinates, times, query_factory, interval_s)


def export_predictions(output, predictor, queries):
    save_verified(output, dict(protocol_id=PROTOCOL_ID, results=predictor.predict_batch(queries),
                              route_success_model=False, research_only=True))
