"""Named deterministic and NPZ-backed benchmark cases."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.moa import synthetic
from scripts.moa.run_planner import load_npz


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    cost_map: object
    source: str
    metadata: dict = None


BUILTIN_SCENARIOS = {
    "lake": synthetic.lake_map,
    "foggy_valley": synthetic.foggy_valley_map,
    "tailwind": synthetic.tailwind_map,
    "nfz": synthetic.nfz_map,
    "mountain_pass": synthetic.mountain_pass_map,
    "walled": synthetic.walled_map,
    "terrain": synthetic.terrain_map,
    "realistic": synthetic.realistic_map,
}


def builtin_cases(names=None):
    names = list(BUILTIN_SCENARIOS) if names is None else names
    unknown = sorted(set(names) - set(BUILTIN_SCENARIOS))
    if unknown:
        raise ValueError(f"unknown built-in scenario(s): {', '.join(unknown)}")
    return [BenchmarkCase(name, BUILTIN_SCENARIOS[name](), f"synthetic:{name}",
                          {"case_kind": "control", "family": name})
            for name in names]


def npz_case(name, path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"map archive not found: {path}")
    cost_map = load_npz(path)
    data = np.load(path, allow_pickle=True)
    # Optional source layers support the navigation-density ablation without
    # changing the planner's stable CostMap schema.
    if "visual_richness" in data:
        cost_map.visual_richness = np.asarray(data["visual_richness"], dtype=float)
    if "rugosity" in data:
        cost_map.terrain_richness = np.asarray(data["rugosity"], dtype=float)
    return BenchmarkCase(name, cost_map, str(path), {"case_kind": "real"})
