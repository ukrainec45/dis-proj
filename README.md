# Autonomous UAV Navigation in Adverse Conditions

Research project on autonomous unmanned aerial vehicle navigation when satellite navigation is unavailable or unreliable and environmental conditions are adverse.

The work investigates a hybrid navigation approach that combines onboard motion estimation, inertial propagation, terrain information, and georeferenced visual landmarks. Its central objective is to plan and execute safe, energy-aware routes while adapting to new observations during flight.

## Research scope

- Pre-flight multi-objective route planning on a raster representation of the area.
- Assessment of terrain, visibility, wind, obstacles, and navigation quality.
- Route adaptation when conditions or obstacles change.
- Localization health monitoring and safe, accurate landing support.
- Operation under limited onboard computing resources and intermittent communications.

## Repository layout

- `knowledgebase/` — research materials, requirements, models, and design documents.
- `notebooks/` — exploratory analysis and data-processing notebook.
- `notebooks/visibility_localization_experiment.ipynb` — Colab aerial + CAMP + RoMa + PnP experiment: 24 frames across 4 scenes × 6 conditions, with an 18-attempt preflight diagnostic; upload [the module folder](scripts/visibility_localization/README.md) to Drive before running; [protocol and validation status](docs/visibility_localization_experiment.md).
- `notebooks/anyvisloc_controlled_ablation.ipynb` — Separate Colab controls for the author RoMa runner, adapter parity, SP/LightGlue, and retrieval with exact/noisy priors; [protocol](docs/anyvisloc_controlled_ablation.md).
- `notebooks/anyvisloc_visibility_quick.ipynb` — Paired visibility stress test of the author pipeline: 8 frames / 40 attempts, extendable to 24 / 120; [protocol](docs/anyvisloc_visibility_quick.md) and [measured results](docs/visibility_localization_results.md).
- `path_planning/` — areas of interest, start/goal points, and restricted-area test data.
- `scripts/pipeline/` — preparation of planning layers and navigation-quality metrics.
- `scripts/pipeline/chance_constrained/` — versioned scene and directed-edge
  inputs for the chance-constrained planner (first increment; no OpenVINS,
  absolute visual-map-localization, TAN, or route-success outcome models yet).
- `scripts/moa/` — multi-objective path-planning experiments and validation utilities.
- `plots/` — generated planning visualizations.

## Development policy

The notebook may be used for exploratory research. Every implementation added outside the notebook must be covered by reliable automated tests. Tests should verify expected behavior, edge cases, invalid inputs, and the safety-critical constraints relevant to the component.

Design documents in `knowledgebase/` are the primary reference for research intent and system requirements.
