# AGENTS.md

**PhD research project** — Autonomous UAV navigation in GPS-denied adverse conditions (Ukraine-based, researcher М.О. Українець).

## Repo structure

- `knowledgebase/` — Design docs (ROS2 architecture), academic papers, functional/structural models.
- `path_planning/` — GeoJSON test data: AOI, start/goal points, no-fly zones.
- `notebooks/path_planning.ipynb` — Google Colab notebook: satellite imagery (Sentinel-2) + DEM processing, visual/terrain navigation quality metrics.
- `scripts/` — Empty (no implementation code yet).

## State

This is a **design-phase research project**. No executable code exists yet. The system is specified but not implemented. The notebook is the only runnable artifact.

## Key design facts

- **Navigation stack:** GPS-denied hybrid system — VIO (OpenVINS) + IMU dead reckoning + terrain-aided navigation + georeferenced landmark corrections. Single `/uav/pose` topic as authoritative state source.
- **Planning:** Pre-flight MOA* (multi-objective A*) on raster grid, not RRT* (RRT* is discussed in papers but MOA* is the chosen primary algorithm for the grid-based problem).
- **Control:** MPC for trajectory tracking & local obstacle avoidance.
- **Target platform:** ROS2 Humble, Python, PX4 via MAVROS.
- **Libraries for data pipeline:** `rasterio`, `pyproj`, `geopandas`, `shapely`, `cfgrib/xarray`, `opencv-python`, `scikit-image`, `numpy`, `scipy`, `matplotlib`.

## Workflow notes

- Notebook is designed for Google Colab — mounts Google Drive for dataset access.
- Satellite data source: Sentinel-2; DEM: SRTM/Copernicus.
- Wind data: GFS/ECMWF GRIB2.
- Papers target Ukrainian academic conferences (ЖВІ, ІКТ, КТІПР).
