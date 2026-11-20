"""Bounded, windowed metric GeoTIFF adapter. New map domains still need an audit."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer

from scripts.visibility_localization.core import file_hash, footprint
from .artifacts import digest


@dataclass
class WindowReference:
    bgr: object
    valid_pixels: object
    transform: object
    crs: str
    resolution: object
    source_resolution: object
    plane_z: float

    def plane_at(self, xy):
        return self.plane_z

    def pixels_to_enu(self, pixels):
        p = np.asarray(pixels)
        return np.column_stack([self.transform.c+(p[:, 0]+.5)*self.transform.a,
                                self.transform.f+(p[:, 1]+.5)*self.transform.e])

    def enu_to_pixels(self, xy):
        p = np.asarray(xy)
        return np.column_stack([(p[:, 0]-self.transform.c)/self.transform.a-.5,
                                (p[:, 1]-self.transform.f)/self.transform.e-.5])


class RasterMap:
    def __init__(self, image_path, dem_path, *, max_window_side=2048):
        import rasterio
        if type(max_window_side) is not int or max_window_side < 256:
            raise ValueError('Window limit must be at least 256 pixels')
        self.image_path, self.dem_path = Path(image_path), Path(dem_path)
        self.max_window_side = max_window_side
        with rasterio.open(image_path) as src:
            crs = CRS.from_user_input(src.crs)
            if not crs.is_projected or any(v.unit_conversion_factor != 1 for v in crs.axis_info[:2]):
                raise ValueError('Preprocess imagery to a projected metric CRS')
            if src.count < 3 or any(t != 'uint8' for t in src.dtypes[:3]):
                raise ValueError('Explicit uint8 RGB preprocessing required')
            if src.transform.b or src.transform.d or src.transform.a <= 0 or src.transform.e >= 0:
                raise ValueError('North-up raster required')
            self.crs = crs.to_string()
        with rasterio.open(dem_path) as dem:
            if dem.crs is None:
                raise ValueError('DEM CRS required')
        # Full-file hashing once, not full raster loading. Audit must bind to these files.
        self.map_id = digest(dict(rgb=file_hash(image_path), dem=file_hash(dem_path)))

    def window(self, query):
        import rasterio
        from rasterio.windows import from_bounds, Window
        from rasterio.enums import Resampling
        from affine import Affine
        if CRS.from_user_input(query.crs) != CRS.from_user_input(self.crs):
            raise ValueError('query_map_crs_mismatch')
        with rasterio.open(self.dem_path) as dem:
            point = Transformer.from_crs(self.crs, dem.crs, always_xy=True).transform(*query.position[:2])
            height = next(dem.sample([point], masked=True))[0]
            if np.ma.is_masked(height) or not np.isfinite(height):
                raise ValueError('No valid DEM height at planned position')
            z = float(height)
        visible, _ = footprint(query.position, query.camera_to_enu, query.K, query.dist, query.image_shape, z, center_crop=True)
        radius = 3*np.sqrt(np.linalg.eigvalsh(query.covariance_xy)[-1])
        x0, y0, x1, y1 = visible.bounds
        with rasterio.open(self.image_path) as src:
            requested = from_bounds(x0-radius, y0-radius, x1+radius, y1+radius, src.transform)
            # Include a one-source-pixel margin for interpolation, clip to physical coverage.
            requested = Window(np.floor(requested.col_off)-1, np.floor(requested.row_off)-1,
                               np.ceil(requested.width)+3, np.ceil(requested.height)+3)
            try:
                window = requested.intersection(Window(0, 0, src.width, src.height))
            except rasterio.errors.WindowError as exc:
                raise ValueError('no_map_coverage') from exc
            scale = min(1., self.max_window_side/max(window.height, window.width))
            h, w = max(2, round(window.height*scale)), max(2, round(window.width*scale))
            rgb = src.read([3, 2, 1], window=window, out_shape=(3, h, w), resampling=Resampling.bilinear)
            mask = src.dataset_mask(window=window, out_shape=(h, w), resampling=Resampling.nearest) > 0
            transform = src.window_transform(window)*Affine.scale(window.width/w, window.height/h)
            return WindowReference(np.moveaxis(rgb, 0, -1), mask, transform, self.crs,
                np.array([transform.a, -transform.e]), np.array([src.transform.a, -src.transform.e]), z)
