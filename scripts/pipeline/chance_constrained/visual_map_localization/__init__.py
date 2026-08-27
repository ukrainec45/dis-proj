"""Synthetic visual-map-localization experiments.

The package evaluates a concrete camera/matcher configuration without
pretending that same-orthophoto synthetic trials are real-flight evidence.
"""

from .camera_model import PinholeCamera
from .config import VmlExperimentConfig, load_vml_config
from .matcher import SiftMapMatcher

__all__ = ["PinholeCamera", "SiftMapMatcher", "VmlExperimentConfig",
           "load_vml_config"]
