"""CPU-only JSON models and coherent empirical joint distributions of correction outcomes."""
from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize_scalar

from .artifacts import digest, save_verified, load_verified
from .contracts import CLASS_NAMES, PROTOCOL_ID, probabilities
from .features import ALL_FEATURES, GEOMETRY, FEATURE_VERSION

VISIBILITY = {'fog_extreme', 'smoke_extreme'}
CANDIDATES = {'constant': ('constant', 'geometry', True)}
for _kind in ('logistic', 'forest'):
    for _representation in ('geometry', 'map', 'camp'):
        for _visibility in (True, False):
            CANDIDATES[f'{_kind}_{_representation}'+('' if _visibility else '_no_visibility')] = (_kind, _representation, _visibility)


def labels(value):
    y = np.asarray(value)
    if y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y > 7).any():
        raise ValueError('Expected integer labels in [0,7]')
    return y.astype(int)


def temperature_scale(p, temperature):
    p = probabilities(p)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive temperature required')
    return softmax(np.log(np.clip(p, 1e-300, 1))/temperature, axis=1)


def fit_temperature(p, y):
    p, y = probabilities(p), labels(y)
    if len(p) != len(y):
        raise ValueError('Calibration lengths differ')
    result = minimize_scalar(lambda lt: -np.log(temperature_scale(p, np.exp(lt))[np.arange(len(y)), y]).mean(),
                             bounds=(-3, 3), method='bounded')
    if not result.success:
        raise RuntimeError('Temperature fit failed')
    return float(np.exp(result.x))


def input_matrix(inputs, features, embedding_dim):
    rows = []
    for value in inputs:
        if set(value) != {'features', 'camp'} or set(value['features']) != set(ALL_FEATURES):
            raise ValueError('Pre-flight feature allowlist failed; labels/GT are not model inputs')
        if not np.isfinite(list(value['features'].values())).all():
            raise ValueError('Nonfinite pre-flight features')
        row = [value['features'][k] for k in features]
        if embedding_dim:
            vector = np.asarray(value['camp'], float)
            if vector.shape != (embedding_dim,):
                raise ValueError('CAMP descriptor dimension mismatch')
            row += vector.tolist()
        rows.append(row)
    x = np.asarray(rows, float)
    if x.shape != (len(inputs), len(features)+embedding_dim) or not np.isfinite(x).all():
        raise ValueError('Nonfinite/invalid feature matrix')
    return x


def _leaf(tree, x):
    node = 0
    while tree['left'][node] != -1:
        node = tree['left'][node] if x[tree['feature'][node]] <= tree['threshold'][node] else tree['right'][node]
    return str(node)


@dataclass
class FrozenModel:
    artifact: dict

    @property
    def model_id(self):
        return digest(self.artifact)

    def base_weights(self, inputs):
        a = self.artifact
        x = input_matrix(inputs, a['features'], a['embedding_dim'])
        n = a['training_count']
        y = np.asarray(a['bank']['class_index'])
        if a['kind'] == 'constant':
            weights = np.full((len(x), n), 1/n)
        elif a['kind'] == 'logistic':
            z = (x-np.asarray(a['mean']))/np.asarray(a['scale'])
            scores = z@np.asarray(a['coef']).T+np.asarray(a['intercept'])
            if len(a['classes']) == 2:
                scores = np.column_stack([np.zeros(len(x)), scores[:, 0]])
            q = softmax(scores, axis=1)
            weights = np.zeros((len(x), n))
            for j, cls in enumerate(a['classes']):
                mask = y == cls
                weights[:, mask] = q[:, j, None]/mask.sum()
        else:
            weights = np.zeros((len(x), n))
            x = x.astype(np.float32)  # sklearn tree comparison convention
            for tree in a['trees']:
                for j, point in enumerate(x):
                    members = np.asarray(tree['members'][_leaf(tree, point)], int)
                    # Members retain bootstrap multiplicities, including repeated indices.
                    weights[j] += np.bincount(members, minlength=n)/len(members)
            weights /= len(a['trees'])
        # Small global empirical backoff supports every OBSERVED class at every query.
        # It does not create examples of classes absent from train.
        mix = 1/(n+1)
        return weights*(1-mix)+mix/n

    def predict_joint(self, inputs, calibrated=True):
        a = self.artifact
        weights = self.base_weights(inputs)
        y = np.asarray(a['bank']['class_index'])
        empirical = np.column_stack([weights[:, y == c].sum(axis=1) for c in range(8)])
        # Dirichlet +1 per class: explicit prior mass, not measured tail reliability.
        p = (a['training_count']*empirical+1)/(a['training_count']+8)
        if calibrated:
            p = temperature_scale(p, a['temperature'])
        adjusted = np.zeros_like(weights)
        for c in range(8):
            mask = y == c
            if mask.any():
                adjusted[:, mask] = weights[:, mask]*(p[:, c]/empirical[:, c])[:, None]
        unknown = p[:, a['missing_classes']].sum(axis=1)
        if not np.allclose(adjusted.sum(axis=1)+unknown, 1):
            raise RuntimeError('Joint probability mass mismatch')
        return dict(probabilities=probabilities(p), weights=adjusted, unresolved_mass=unknown,
                    bank=a['bank'], missing_classes=a['missing_classes'])

    def predict(self, inputs, calibrated=True):
        return self.predict_joint(inputs, calibrated)['probabilities']

    def support(self, inputs):
        x = input_matrix(inputs, self.artifact['features'], self.artifact['embedding_dim'])
        if not x.shape[1]:
            return np.zeros(len(inputs), bool)
        return ((x < np.asarray(self.artifact['minimum'])) | (x > np.asarray(self.artifact['maximum']))).any(axis=1)

    def save(self, path):
        save_verified(path, self.artifact)

    @classmethod
    def load(cls, path):
        a = load_verified(path)
        if a['protocol_id'] != PROTOCOL_ID or a['feature_version'] != FEATURE_VERSION or tuple(a['class_names']) != CLASS_NAMES:
            raise ValueError('Frozen model protocol mismatch')
        if a['name'] not in CANDIDATES:
            raise ValueError('Unknown candidate')
        return cls(a)


def fit_model(name, inputs, outcomes, *, forest_parameters=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    if name not in CANDIDATES:
        raise ValueError('Unknown candidate')
    kind, representation, visibility = CANDIDATES[name]
    y = labels([o['class_index'] for o in outcomes])
    if len(inputs) != len(y):
        raise ValueError('Input/outcome lengths differ')
    features = list(GEOMETRY if representation == 'geometry' else ALL_FEATURES)
    if not visibility:
        features = [k for k in features if k not in VISIBILITY]
    embedding_dim = len(inputs[0]['camp']) if representation == 'camp' else 0
    if kind == 'constant':
        features = []
    x = input_matrix(inputs, features, embedding_dim)
    counts = np.bincount(y, minlength=8)
    bank = {k: [o[k] for o in outcomes] for k in ('class_index', 'error_before_xy_m', 'error_after_xy_m')}
    if np.asarray(bank['error_after_xy_m']).shape != (len(y), 2) or not np.isfinite(bank['error_after_xy_m']).all():
        raise ValueError('Invalid error bank')
    a = dict(protocol_id=PROTOCOL_ID, feature_version=FEATURE_VERSION, class_names=list(CLASS_NAMES),
        name=name, kind=kind, representation=representation, visibility_input=visibility,
        features=features, embedding_dim=embedding_dim, temperature=1., training_count=len(y),
        class_counts=counts.tolist(), missing_classes=np.flatnonzero(counts == 0).tolist(), bank=bank,
        minimum=x.min(axis=0).tolist(), maximum=x.max(axis=0).tolist(),
        research_only=True, flight_validation_passed=False,
        smoothing='Dirichlet +1/class; unobserved classes carry unresolved outcome mass')
    if kind == 'constant' or len(np.unique(y)) == 1:
        a['kind'] = 'constant'
    elif kind == 'logistic':
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-12] = 1
        estimator = LogisticRegression(C=1., solver='lbfgs', max_iter=2000)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            estimator.fit((x-mean)/scale, y)
        a.update(mean=mean.tolist(), scale=scale.tolist(), classes=estimator.classes_.tolist(),
                 coef=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist())
    else:
        parameters = dict(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=1)
        if forest_parameters is not None:  # Small synthetic tests only; persisted and visible.
            parameters.update(forest_parameters)
        estimator = RandomForestClassifier(**parameters).fit(x, y)
        trees = []
        for tree_estimator, samples in zip(estimator.estimators_, estimator.estimators_samples_):
            t = tree_estimator.tree_
            leaves = tree_estimator.apply(x[samples].astype(np.float32))
            members = {str(int(leaf)): samples[leaves == leaf].tolist() for leaf in np.unique(leaves)}
            trees.append(dict(left=t.children_left.tolist(), right=t.children_right.tolist(),
                feature=t.feature.tolist(), threshold=t.threshold.tolist(), members=members))
        a.update(trees=trees, parameters=parameters)
    return FrozenModel(a)
