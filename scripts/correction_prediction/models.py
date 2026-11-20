"""Small frozen baselines, portable JSON forest, held-out temperature scaling."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from .artifacts import digest, load_verified, save_verified
from .contracts import CLASS_NAMES, PROTOCOL_ID, probabilities
from .features import GEOMETRY, ALL_FEATURES, FEATURE_VERSION

CANDIDATES = {'constant': (), 'logistic_geometry': GEOMETRY,
              'logistic_map': ALL_FEATURES, 'forest_geometry': GEOMETRY,
              'forest_map': ALL_FEATURES}


def matrix(rows, features):
    x = np.asarray([[r[k] for k in features] for r in rows], dtype=float)
    if x.shape != (len(rows), len(features)) or not np.isfinite(x).all():
        raise ValueError('Invalid model features')
    return x


def labels(y):
    a = np.asarray(y)
    if a.ndim != 1 or not a.size or not np.issubdtype(a.dtype, np.integer) or (a < 0).any() or (a > 7).any():
        raise ValueError('Expected nonempty integer class labels 0..7')
    return a.astype(int)


def temperature_scale(p, temperature):
    probabilities(p)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive temperature required')
    return softmax(np.log(np.clip(p, 1e-12, 1))/temperature, axis=1)


def fit_temperature(p, y):
    p, y = probabilities(p), labels(y)
    if len(p) != len(y):
        raise ValueError('Calibration row mismatch')
    def objective(log_t):
        q = temperature_scale(p, np.exp(log_t))
        return -float(np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean())
    result = minimize_scalar(objective, bounds=(-3, 3), method='bounded')
    if not result.success:
        raise RuntimeError('Temperature fitting failed')
    return float(np.exp(result.x))


@dataclass
class FrozenModel:
    artifact: dict

    @property
    def model_id(self):
        return digest(self.artifact)

    def predict(self, rows, calibrated=True):
        a = self.artifact
        x = matrix(rows, a['features'])
        n = len(rows)
        if a['kind'] == 'constant':
            p = np.tile(a['frequencies'], (n, 1))
        else:
            p = np.full((n, 8), 1e-12)
            if a['kind'] == 'logistic':
                z = (x-np.asarray(a['mean']))/np.asarray(a['scale'])
                logits = z@np.asarray(a['coef']).T+np.asarray(a['intercept'])
                if len(a['classes']) == 2:
                    logits = np.column_stack([np.zeros(n), logits[:, 0]])
                q = softmax(logits, axis=1)
            else:
                # sklearn trees compare float32-converted input at inference.
                x = x.astype(np.float32)
                q = np.zeros((n, len(a['classes'])))
                for t in a['trees']:
                    values = np.asarray(t['values'])
                    for i in range(n):
                        node = 0
                        while t['left'][node] != -1:
                            node = (t['left'][node] if x[i, t['feature'][node]] <= t['threshold'][node]
                                    else t['right'][node])
                        v = values[node]
                        q[i] += v/v.sum()
                q /= len(a['trees'])
            p[:, a['classes']] = q
            p /= p.sum(axis=1, keepdims=True)
        return temperature_scale(p, a.get('temperature', 1)) if calibrated else probabilities(p)

    def support(self, rows):
        x = matrix(rows, self.artifact['features'])
        if x.shape[1] == 0:
            return np.zeros(len(rows), dtype=bool)
        return ((x < np.asarray(self.artifact['minimum'])) |
                (x > np.asarray(self.artifact['maximum']))).any(axis=1)

    def save(self, path):
        save_verified(path, self.artifact)

    @classmethod
    def load(cls, path):
        a = load_verified(path)
        if a['protocol_id'] != PROTOCOL_ID or tuple(a['class_names']) != CLASS_NAMES or a['feature_version'] != FEATURE_VERSION:
            raise ValueError('Model protocol/feature mismatch')
        if a['name'] not in CANDIDATES or tuple(a['features']) != CANDIDATES[a['name']]:
            raise ValueError('Model feature allowlist mismatch')
        return cls(a)


def fit_model(name, rows, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    if name not in CANDIDATES:
        raise ValueError(name)
    y = labels(y)
    if len(y) != len(rows):
        raise ValueError('Training row mismatch')
    features = CANDIDATES[name]
    x = matrix(rows, features)
    counts = np.bincount(y, minlength=8)
    a = dict(protocol_id=PROTOCOL_ID, feature_version=FEATURE_VERSION, class_names=list(CLASS_NAMES),
             name=name, features=list(features), temperature=1., training_count=len(y),
             class_counts=counts.tolist(), missing_classes=np.flatnonzero(counts == 0).tolist(),
             minimum=x.min(axis=0).tolist(), maximum=x.max(axis=0).tolist(), research_only=True)
    if name == 'constant' or len(np.unique(y)) < 2:
        # Explicit Laplace smoothing; a one-class fit is marked in missing_classes.
        a.update(kind='constant', frequencies=((counts+1)/(len(y)+8)).tolist())
    elif name.startswith('logistic'):
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-12] = 1
        estimator = LogisticRegression(C=1., solver='lbfgs', max_iter=2000)
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            estimator.fit((x-mean)/scale, y)
        a.update(kind='logistic', classes=estimator.classes_.tolist(), mean=mean.tolist(),
                 scale=scale.tolist(), coef=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist())
    else:
        estimator = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                           random_state=42, n_jobs=1)
        estimator.fit(x, y)
        trees = []
        for e in estimator.estimators_:
            t = e.tree_
            trees.append(dict(left=t.children_left.tolist(), right=t.children_right.tolist(),
                              feature=t.feature.tolist(), threshold=t.threshold.tolist(),
                              values=t.value[:, 0, :].tolist()))
        a.update(kind='forest', classes=estimator.classes_.tolist(), trees=trees,
                 parameters=dict(n_estimators=300, max_depth=6, min_samples_leaf=5, seed=42))
    return FrozenModel(a)
