# Onboard landmark database

This package creates one versioned SQLite landmark database for one
georeferenced reference-image acquisition and AOI. The source image must be a
north-up GeoTIFF in a projected CRS whose linear unit is metres. A DEM is
reprojected to the image grid during generation.

Install generator dependencies into the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r scripts\pipeline\landmarks\requirements.txt
```

Build a database and companion planner-grid quality layer:

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline.landmarks.build_database `
  --image data\reference\area.tif `
  --dem data\dem\area_dem.tif `
  --aoi path\to\aoi.geojson `
  --output data\landmarks\area.sqlite `
  --planner-cell-size-m 50
```

The command writes `area.sqlite` and `area.quality.npz`. The SQLite package
contains ORB descriptors, 2.5D map coordinates, terrain context, tile quality,
and input checksums. `area.quality.npz` contains `landmark_quality` and
`landmark_count`, aligned to the requested planner-cell grid.

Inspect the package or test a bounded onboard-style query:

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline.landmarks.inspect_database `
  --database data\landmarks\area.sqlite `
  --east-m 450000 --north-m 5570000 --radius-m 100
```

Use the reader from onboard Python code:

```python
from scripts.pipeline.landmarks import LandmarkDatabase

with LandmarkDatabase("data/landmarks/area.sqlite") as database:
    batch = database.query_nearby(east_m, north_m, radius_m=100, max_landmarks=300)
    # batch.map_xyz: [N, 3], batch.descriptors: [N, 32] uint8 ORB descriptors
```

The database supplies reference correspondences only. Live-camera matching,
geometric verification, and fusion into the UAV state estimator are separate
onboard stages.
