"""Frozen main design; one prior draw is paired across visibility conditions."""
from scripts.visibility_localization.core import defaults

REVIEWED = {'Scene_05', 'Scene_07', 'Scene_08', 'Scene_11', 'Scene_21'}
SPLIT_SCENES = {'train': 8, 'validation': 2, 'calibration': 2, 'test': 4}
FRAMES_PER_SCENE = 10
APPROVED_ATTEMPTS = 1920
SEED = 42


def main_config():
    config = defaults()
    config.update(seed=SEED, reference_mode='aerial', matching_method='Roma',
                  map_alignment='prior_yaw', sigma_xy_m=[5., 20., 50., 100.],
                  trials=1, sample_count=160,
                  fog={'fog_extreme': .1},
                  smoke={'smoke_extreme': dict(coverage=.9, transmission=.05, softness=.08,
                                              color_rgb=[.5, .48, .45])})
    return config
