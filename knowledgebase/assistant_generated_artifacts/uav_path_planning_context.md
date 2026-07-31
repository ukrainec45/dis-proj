# UAV Pre-Flight Path Planning — Design Context

## Project Overview

Autonomous UAV navigation system. This subsystem handles **pre-flight path planning** — the first step in the UAV task execution flow. Implemented as a standalone Python module (no ROS2 dependency), later integrated into a ROS2 project.

**Target UAV types:** multirotors, VTOL-like UAVs, helicopter-like UAVs.

**Output:** array of waypoints the UAV follows during flight.

---

## Problem Formulation

### Type
The problem is **multi-objective in nature**, solved as a **true multi-objective optimization** producing a **Pareto front** of non-dominated paths. A separate decision maker selects one path from the front.

### Decision Variables

A path $P$ is an ordered sequence of waypoints:

$$P = \langle w_0, w_1, \ldots, w_n \rangle, \quad w_i = (x_i, y_i, z_i) \in \mathbb{R}^3$$

Edge set:

$$E(P) = \{(w_i, w_{i+1}) \mid i = 0, \ldots, n-1\}$$

### Space Representation
**3D grid** — planning happens on a 3D raster grid. Altitude $z_i$ is a free decision variable subject to constraints.

---

## Wind Model

Wind vector $\vec{u}(e)$ is sampled at the midpoint of each edge. It is decomposed into two components relative to flight direction $\hat{e}$ (unit vector along edge):

$$u_\parallel(e) = \vec{u}(e) \cdot \hat{e} \qquad \text{(along-track — dot product)}$$
$$u_\perp(e) = \vec{u}(e) \times \hat{e} \qquad \text{(cross-track — 2D cross product scalar)}$$

**Ground speed** (UAV flies at $v_{\text{air}}$, crabbing into crosswind to maintain track):

$$v_g(e) = \sqrt{v_{\text{air}}^2 - u_\perp(e)^2} + u_\parallel(e)$$

- The $\sqrt{v_{\text{air}}^2 - u_\perp^2}$ term comes from Pythagoras — crosswind forces the UAV to crab, consuming part of $v_{\text{air}}$ to cancel drift, leaving less airspeed for forward motion.
- $u_\parallel$ adds (tailwind) or subtracts (headwind) from ground speed linearly.
- Wind is spatially varying but temporally frozen at forecast time (standard NWP assumption).

**Flight time per edge:**

$$t(e) = \frac{d(e)}{v_g(e)}$$

Where $d(e)$ is the Euclidean length of the edge in meters.

---

## Objective Functions

### $f_1$ — Flight Time (minimise)

$$f_1(P) = \sum_{e \in E(P)} t(e) = \sum_{e \in E(P)} \frac{d(e)}{\sqrt{v_{\text{air}}^2 - u_\perp(e)^2} + u_\parallel(e)}$$

Primary objective. Time is the main optimization criterion because critical tasks in harsh environments prioritize speed of completion.

### $f_2$ — Navigation Quality Deficit (minimise)

$$f_2(P) = \sum_{e \in E(P)} \left(1 - \rho_{\text{nav}}(e)\right) \cdot t(e)$$

Where navigation feature density combines visual richness and terrain roughness as **alternative** localization sources (UAV can use whichever works):

- $\phi_{\text{vis}}(e) \in [0,1]$ — visual feature richness from satellite imagery (high = good)
- $\phi_{\text{ter}}(e) \in [0,1]$ — terrain roughness from DEM rugosity (high = good)
- **max** is used because the two are alternative navigation methods — if one fails the other takes over. Geometric mean would be used if both were required simultaneously.
- $1 - \rho_{\text{nav}}(e)$ is the **navigation quality deficit** — flying for a long time over feature-poor terrain accumulates a large deficit, which is penalised. Multiplied by $t(e)$ because deficit accrues over flight time.

### $f_3$ — Visibility Deficit (minimise)

$$f_3(P) = \sum_{e \in E(P)} \left(1 - \phi_{\text{vsb}}(e)\right) \cdot t(e)$$

$\phi_{\text{vsb}}(e) \in [0,1]$ — visibility quality along edge (1 = full visibility, high = good).
- $1 - \phi_{\text{vsb}}(e)$ is the **visibility deficit** — flying for a long time in degraded visibility accumulates a large deficit, which is penalised. Multiplied by $t(e)$ because deficit accrues over flight time.

### Combined Objective Vector

$$\min_{P} \; \mathbf{F}(P) = \bigl(f_1(P),\ f_2(P),\ f_3(P)\bigr)$$

All three objectives are **time-weighted and minimised**: $f_2$ and $f_3$ are expressed as navigation/visibility **deficits**, so no negation is needed.

---

## Hard Constraints

All constraints together form a system — every one must be satisfied simultaneously:

$$\text{subject to} \begin{cases} \text{dem}(x_i, y_i) + h_{\text{clearance}} \leq z_i \leq z_{\max} & \forall i \\[6pt] w_i \notin \mathcal{F} & \forall i \\[6pt] \|\vec{u}(e)\| < v_{\max} & \forall e \in E(P) \\[6pt] \displaystyle\sum_{e \in E(P)} \varepsilon(e) \leq \mathcal{E}_{\text{budget}} \\[6pt] \mathcal{E}_{\text{budget}} - \displaystyle\sum_{e \in E(P_0 \to w_i)} \varepsilon(e) \geq \varepsilon(w_i \to L_i) & \forall i \\[6pt] \rho(e_i, e_{i+1}) \geq r_{\min} & \forall i \\[6pt] w_0 = s, \quad w_n = g \end{cases}$$

### Constraint Explanations

**Terrain clearance and altitude ceiling:**
UAV must fly above terrain plus safety margin, and below its operational ceiling.

**No-fly zones $\mathcal{F}$:**
$\mathcal{F}$ = NFZ cells (regulatory). These are hard geometric exclusions.

**Total wind magnitude $\|\vec{u}(e)\| < v_{\max}$:**
Total wind speed must be strictly less than UAV maximum airspeed on every edge. If wind equals or exceeds airspeed, UAV cannot make forward progress in any direction. This constraint subsumes the crosswind constraint — $\|\vec{u}\|^2 = u_\perp^2 + u_\parallel^2 \geq u_\perp^2$.

**Total energy budget $\sum \varepsilon(e) \leq \mathcal{E}_{\text{budget}}$:**
Total energy consumed on the path must not exceed the energy available at takeoff.

**Energy margin (abort safety):**
At every waypoint $w_i$, remaining energy must be sufficient to reach the nearest safe landing point $L_i$. The left side is remaining energy at $w_i$ (budget minus energy spent so far). The right side is energy needed to fly from $w_i$ to $L_i$. $L_i$ is the nearest cell with sufficient flat area derived from DEM.

**Kinematic feasibility $\rho(e_i, e_{i+1}) \geq r_{\min}$:**
Turning radius implied by the angle between consecutive edges must be at least the UAV's minimum turning radius. Trivially satisfied for multirotors ($r_{\min} = 0$). Critical for fixed-wing VTOL in cruise — requires Dubins path smoothing post-planning.

**Boundary conditions:** path must start at $s$ and end at $g$.

---

## Why Constraints Are Hard, Not Soft

Quality factors (visual richness, visibility) are **soft** — encoded in objectives or as time penalties. They influence routing but never block a path. This prevents dead ends where no feasible path exists.

Only physically impossible or safety-critical conditions are hard constraints:
- Terrain collision — physically impossible
- NFZ — legal hard boundary
- Wind exceeding airspeed — UAV cannot fly
- Energy exhaustion — mission failure

---

## Input Data

| Source | Format | What It Provides |
|---|---|---|
| Start/goal points | Coordinates | Boundary conditions |
| Satellite imagery | GeoTIFF (RGB) | Visual feature density $\rho_{\text{vis}}$ |
| DEM | GeoTIFF | Terrain elevation, rugosity $\rho_{\text{flat}}$, occupancy mask |
| Weather (NWP) | GRIB2 (GFS/ECMWF) | Visibility quality $\phi_{\text{vsb}}$ |
| Wind field | GRIB2 U/V components | Wind vector $\vec{u}(e)$ per cell at cruise altitude |
| NFZ polygons | GeoJSON/Shapefile | $\mathcal{F}$ occupancy masks |
| UAV specification | Config | $v_{\text{air}}$, $v_{\max}$, $\mathcal{E}_{\text{budget}}$, $r_{\min}$, $h_{\text{clearance}}$, $z_{\max}$ |

---

## Map Representation

### Common Spatial Reference
All layers reprojected to a common UTM grid using `rasterio` + `pyproj`. Cell $(x, y)$ indexes into every layer simultaneously.

### Layer Types

| Layer | Shape | Type | Purpose |
|---|---|---|---|
| `dem` | `[H, W]` | float32 | Ground elevation in meters |
| `visual_richness` | `[H, W]` | float32 [0..1] | Feature density from imagery |
| `rugosity` | `[H, W]` | float32 [0..1] | Terrain roughness from DEM |
| `nav_density` | `[H, W]` | float32 [0..1] | max(visual, rugosity) |
| `visibility` | `[H, W]` | float32 [0..1] | Visibility quality (1 = full visibility) |
| `wind_field` | `[H, W, 2]` | float32 | [u_east, u_north] at cruise alt |
| `occupancy` | `[H, W]` | bool | Hard blocked cells |

H = number of cells north-south, W = number of cells east-west.

### Layer Normalization
All map layers (visual, rugosity, visibility) are normalized to [0..1] **once at build time** using percentile clipping to handle outliers:

```python
def normalize_robust(layer, lo_pct=2, hi_pct=98):
    lo, hi = np.percentile(layer, lo_pct), np.percentile(layer, hi_pct)
    return np.clip((layer - lo) / (hi - lo + 1e-9), 0.0, 1.0)
```

### Layer Processing

**DEM → rugosity (terrain roughness):**
```python
mean    = uniform_filter(dem, size=5)
mean_sq = uniform_filter(dem**2, size=5)
rugosity = np.sqrt(np.maximum(mean_sq - mean**2, 0))
# Then normalize_robust()
```

**Imagery → visual richness:**
Computed from satellite image using Laplacian variance, gradient magnitude, and local entropy. Combined and normalized.

**Navigation density:**
```python
nav_density = np.maximum(visual_richness, rugosity)  # alternative sources
```

**DEM → occupancy:**
```python
for z in range(altitude_layers):
    cell_altitude = alt_min + z * alt_step
    occupancy[z] = (dem + clearance) > cell_altitude
```

---

## Edge Cost Evaluation

The planner calls `edge_cost(start, end)` for every candidate edge. It returns a 3-vector of objective contributions:

```python
def edge_objectives(start, end, cost_map, v_air, v_max):
    # Hard constraints first
    if cost_map.occupancy[end[1], end[0]]:
        return None  # infeasible

    wind = sample_wind(cost_map.wind_field, start, end)
    if np.linalg.norm(wind) >= v_max:
        return None  # wind exceeds airspeed

    # Geometric properties (3D distance — altitude from DEM)
    dx = (end[0] - start[0]) * cost_map.resolution_m
    dy = (end[1] - start[1]) * cost_map.resolution_m
    dz = cost_map.dem[end[1], end[0]] - cost_map.dem[start[1], start[0]]
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    flight_dir = np.array([dx, dy]) / (distance + 1e-9)

    u_par   = np.dot(wind, flight_dir)          # along-track
    u_perp  = np.cross(wind, flight_dir)        # cross-track

    vg = np.sqrt(max(v_air**2 - u_perp**2, 0)) + u_par
    vg = max(vg, 0.1)
    t  = distance / vg

    # Map layer samples along edge
    nav  = sample_layer(cost_map.nav_density, start, end)
    vsb  = sample_layer(cost_map.visibility,  start, end)

    return (t, (1.0 - nav) * t, (1.0 - vsb) * t)  # (f1, f2, f3) contributions
```

Layer sampling walks along the edge and averages values at intermediate cells:

```python
def sample_layer(layer, start, end, n=5):
    values = []
    for t in np.linspace(0, 1, n):
        x = int(start[0] + t * (end[0] - start[0]))
        y = int(start[1] + t * (end[1] - start[1]))
        values.append(layer[y, x])
    return float(np.mean(values))
```

---

## Planning Algorithm

### Primary: MOA* (Multi-Objective A*)

Extends A* to track sets of non-dominated cost vectors per node instead of a single scalar. Produces the exact Pareto front in one run.

**Why MOA* over alternatives:**
- Data is raster-native — grid search exploits this directly
- Objectives are additive sums over edges — MOA* accumulates cost vectors naturally
- Deterministic — same input always gives same output
- Exact — finds all Pareto-optimal paths

**Why not RRT*:** RRT* is designed for continuous high-dimensional spaces. For raster-native 2D grid planning it converges slowly, does not exploit additive structure, and produces approximate non-deterministic results.

**Why not NSGA-II:** population-based evolutionary algorithm, orders of magnitude slower than graph search for additive path objectives. Appropriate for offline parameter tuning, not per-flight planning.

### Fallback: Repeated Weighted A*

Run standard A* N times with different random weight vectors (sampled from Dirichlet distribution). Collect distinct paths, remove dominated ones. Simple to implement, reuses single-objective A*. Misses non-convex Pareto regions but adequate for initial implementation.

### Grid Connectivity

Neighbors of cell $(x, y)$ are the 8 surrounding cells (including diagonals). Edges are generated on the fly during search — never stored explicitly.

---

## Decision Making (TOPSIS)

After the Pareto front is computed, TOPSIS selects one path:

1. Build decision matrix $D$ where rows = paths, columns = objective values
2. Normalize columns: $\hat{d}_{ij} = d_{ij} / \sqrt{\sum_i d_{ij}^2}$
3. Apply weights: $v_{ij} = \lambda_j \cdot \hat{d}_{ij}$
4. Find ideal $A^+$ (best value per objective) and anti-ideal $A^-$ (worst)
5. Compute distance to ideal $d_i^+$ and anti-ideal $d_i^-$ for each path
6. Closeness coefficient: $C_i = d_i^- / (d_i^+ + d_i^-)$
7. Select path with highest $C_i$

Weights $\lambda_j$ are mission-context dependent — different weight vectors for emergency vs survey vs cargo missions.

---

## UAV Type Differences

| Property | Multirotor | Fixed-Wing VTOL | Helicopter |
|---|---|---|---|
| $r_{\min}$ | 0 (turns on spot) | 40–100m at cruise | ~0 |
| Path smoothing needed | No | Yes — Dubins curves | No |
| Max descent rate | Symmetric with climb | Symmetric | Lower than climb (vortex ring state) |
| Energy model | High hover cost | Efficient cruise | Similar to multirotor |
| VTOL transition | N/A | Requires straight segment | N/A |

UAV type is abstracted via a `UAVKinematicModel` interface so the planner core is type-agnostic.

---

## Key Libraries

| Task | Library |
|---|---|
| GeoTIFF read/reproject | `rasterio` |
| Coordinate transforms | `pyproj` |
| Polygon rasterization | `rasterio.features` |
| Geospatial vector data | `geopandas`, `shapely` |
| GRIB2 weather files | `cfgrib`, `xarray` |
| Image feature analysis | `opencv-python`, `scikit-image` |
| Numerical operations | `numpy`, `scipy` |
| Visualization | `matplotlib` |

---

## Validation Approach

### Synthetic Maps
Build controlled maps with known correct answers before using real geodata. Scenarios include:
- Direct path vs detour around featureless lake (expects Pareto tradeoff)
- NFZ blocking direct route (expects single detour path)
- Tailwind routing (expects geometrically longer path to be time-optimal)
- No feasible path (expects planner to report infeasible correctly)

### Unit Tests
Each objective function tested independently with known inputs:
- $f_1$: calm air, tailwind, headwind, crosswind cases
- $f_2$: max of two navigation sources
- $f_3$: visibility quality cases
- Wind constraint: boundary cases at $\|\vec{u}\| < v_{\max}$ and $\geq v_{\max}$

### Pareto Front Validation
Verify no returned path dominates another. A path $P_j$ dominates $P_i$ if it is better or equal on all objectives and strictly better on at least one.

### Visual Inspection
Plot all map layers and overlay Pareto-optimal paths. Plot Pareto front in objective space (time vs navigation quality) to confirm expected tradeoff shape.

---

## Open Questions / Future Work

- Energy model per UAV type (power curve $P(v_{\text{air}})$) needs calibration from flight data
- Crosswind energy coefficient $k_\perp$ is UAV-specific — needs empirical determination
- $L_i$ (nearest safe landing point) computation from DEM — flat area detection algorithm needed
- Path smoothing post-processing for fixed-wing VTOL (Dubins paths)
- Integration into ROS2 as a service node
- Real geodata pipeline: GFS/ECMWF wind download, OpenAirMap NFZ fetch, DEM and imagery acquisition
