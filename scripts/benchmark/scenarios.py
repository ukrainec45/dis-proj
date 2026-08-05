"""Named deterministic and NPZ-backed benchmark cases."""

from dataclasses import dataclass
from pathlib import Path

from scripts.moa import synthetic
from scripts.moa.run_planner import load_npz


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    cost_map: object
    source: str


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
    return [BenchmarkCase(name, BUILTIN_SCENARIOS[name](), f"synthetic:{name}")
            for name in names]


def npz_case(name, path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"map archive not found: {path}")
    return BenchmarkCase(name, load_npz(path), str(path))
