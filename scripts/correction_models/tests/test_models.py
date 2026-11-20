import numpy as np
import pytest

from scripts.correction_models.models import fit_model, FrozenModel, temperature_scale, fit_temperature
from scripts.correction_models.distribution_metrics import radial_crps, energy_score, evaluate_distribution
from scripts.correction_models.contracts import label_outcome
from .conftest import small_inputs


@pytest.mark.parametrize('name', ['constant', 'logistic_geometry', 'logistic_map_no_visibility',
                                'logistic_camp', 'forest_geometry', 'forest_map', 'forest_camp_no_visibility'])
def test_joint_mass_class_consistency_and_round_trip(name, tmp_path):
    x, outcomes = small_inputs()
    model = fit_model(name, x, outcomes, forest_parameters={'n_estimators': 5})
    model.artifact['temperature'] = 1.8
    for calibrated in (False, True):
        joint = model.predict_joint(x[:4], calibrated)
        classes = np.array(joint['bank']['class_index'])
        np.testing.assert_allclose(joint['weights'].sum(axis=1)+joint['unresolved_mass'], 1)
        for cls in range(8):
            np.testing.assert_allclose(joint['weights'][:, classes == cls].sum(axis=1), joint['probabilities'][:, cls])
    path = tmp_path/'model.json'
    model.save(path)
    frozen = FrozenModel.load(path)
    assert frozen.model_id == model.model_id
    np.testing.assert_allclose(frozen.predict(x), model.predict(x), atol=0, rtol=0)


def test_unseen_harm_is_unresolved_not_a_fabricated_error_vector():
    x, outcomes = small_inputs()
    keep = [i for i, o in enumerate(outcomes) if o['class_index'] != 1]
    model = fit_model('forest_map', [x[i] for i in keep], [outcomes[i] for i in keep], forest_parameters={'n_estimators': 3})
    joint = model.predict_joint(x[:3])
    assert joint['missing_classes'] == [1]
    assert (joint['probabilities'][:, 1] > 1e-12).all()
    np.testing.assert_allclose(joint['unresolved_mass'], joint['probabilities'][:, 1])
    assert 1 not in joint['bank']['class_index']
    score = evaluate_distribution(joint, outcomes[:3])
    assert score['full_attempt']['scored_count'] == 0
    assert score['full_attempt']['crps_m'] is None
    assert score['full_attempt']['resolved_component_diagnostic']['crps_m'] is not None


def test_temperature_calibration_and_proper_point_scores():
    p = np.full((8, 8), .01/7)
    np.fill_diagonal(p, .99)
    t = fit_temperature(p, np.arange(8))
    np.testing.assert_allclose(temperature_scale(p, t).sum(axis=1), 1)
    assert radial_crps([3], [1], 5) == 2
    assert energy_score([[3, 4]], [1], [0, 0]) == 5
    assert radial_crps([0, 2], [.5, .5], 1) == .5


def test_missing_rejection_does_not_invalidate_accepted_only_distribution():
    x, outcomes = small_inputs()
    keep = [i for i, o in enumerate(outcomes) if o['class_index'] != 0]
    model = fit_model('constant', [x[i] for i in keep], [outcomes[i] for i in keep])
    joint = model.predict_joint([x[1]])
    score = evaluate_distribution(joint, [outcomes[1]])
    assert score['full_attempt']['crps_m'] is None
    assert score['accepted_only']['scored_count'] == 1
    assert score['accepted_only']['crps_m'] is not None


def test_visibility_ablation_does_not_change_inputs_or_use_hidden_scenario():
    x, outcomes = small_inputs()
    model = fit_model('forest_map_no_visibility', x, outcomes, forest_parameters={'n_estimators': 3})
    original = model.predict(x[:1])
    changed = dict(x[0], features=dict(x[0]['features'], fog_extreme=1., smoke_extreme=1.))
    np.testing.assert_array_equal(original, model.predict([changed]))
    assert 'fog_extreme' not in model.artifact['features']


def test_ground_truth_is_rejected_as_a_prediction_input():
    x, outcomes = small_inputs()
    model = fit_model('constant', x, outcomes)
    with pytest.raises(ValueError, match='allowlist'):
        model.predict([dict(x[0], ground_truth=[0, 0])])


@pytest.mark.parametrize('delta,cls', [(-21, 1), (-20, 2), (-5, 3), (-1, 4), (0, 4), (1, 5), (5, 6), (20, 7)])
def test_eight_class_boundaries(delta, cls):
    assert label_outcome([200, 0, 0], [200-delta, 0, 0], [0, 0, 0], True)['class_index'] == cls


def test_rejection_preserves_prior_not_diagnostic_pnp():
    outcome = label_outcome([3, 4, 0], [0, 0, 0], [0, 0, 0], False)
    assert outcome['class_index'] == 0
    assert outcome['error_after_xy_m'] == [3, 4]
