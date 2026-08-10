"""GeoTIFF/AOI loading and DEM alignment for landmark generation."""

from dataclasses import dataclass

import numpy as np


class GeospatialInputError(ValueError):
    """Raised when the reference map cannot safely define a metric map frame."""


@dataclass(frozen=True)
class ReferenceData:
    """Reference image and DEM aligned to one cropped raster grid."""

    image: np.ndarray  # [bands, rows, cols]
    dem: np.ndarray  # [rows, cols]
    valid_mask: np.ndarray  # AOI mask on the cropped raster grid
    transform: object
    crs_wkt: str
    pixel_size_m: tuple[float, float]
    bounds: tuple[float, float, float, float]  # min E, min N, max E, max N


def _dependencies():
    try:
        import geopandas as gpd
        import rasterio
        from rasterio import features
        from rasterio.mask import mask
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise RuntimeError(
            "Landmark generation requires geopandas, rasterio, and their geospatial dependencies."
        ) from exc
    return gpd, rasterio, features, mask, Resampling, reproject


def validate_metric_crs(crs):
    """Require a projected CRS expressed in metres for onboard metric queries."""
    if crs is None:
        raise GeospatialInputError("reference GeoTIFF has no CRS")
    if not crs.is_projected:
        raise GeospatialInputError("reference GeoTIFF must use a projected metric CRS, not geographic coordinates")
    try:
        _, factor = crs.linear_units_factor
    except (AttributeError, ValueError):
        factor = None
    if factor is None or not np.isclose(float(factor), 1.0, rtol=0, atol=1e-9):
        raise GeospatialInputError("reference GeoTIFF CRS linear units must be metres")


def load_reference_data(image_path, dem_path, aoi_path):
    """Crop a GeoTIFF to the AOI and resample the DEM onto that exact grid."""
    gpd, rasterio, features, mask, Resampling, reproject = _dependencies()
    aoi = gpd.read_file(aoi_path)
    if aoi.empty or aoi.geometry.is_empty.all():
        raise GeospatialInputError("AOI contains no usable geometry")

    with rasterio.open(image_path) as image_src:
        validate_metric_crs(image_src.crs)
        if image_src.transform.b != 0 or image_src.transform.d != 0:
            raise GeospatialInputError("rotated reference GeoTIFF transforms are not supported")
        if image_src.transform.a <= 0 or image_src.transform.e >= 0:
            raise GeospatialInputError("reference GeoTIFF must use an east-right, north-up raster transform")
        metric_aoi = aoi.to_crs(image_src.crs)
        geometries = [geometry.__geo_interface__ for geometry in metric_aoi.geometry if not geometry.is_empty]
        try:
            image, transform = mask(image_src, geometries, crop=True, filled=False)
        except ValueError as exc:
            raise GeospatialInputError("AOI does not overlap the reference GeoTIFF") from exc
        if image.shape[1] == 0 or image.shape[2] == 0:
            raise GeospatialInputError("AOI crop is empty")
        # Convert before filling: integer GeoTIFF bands cannot represent NaN masks.
        image = np.ma.filled(image.astype(np.float32), np.nan)
        valid_mask = features.geometry_mask(geometries, out_shape=image.shape[1:],
                                            transform=transform, invert=True)
        bounds_obj = rasterio.transform.array_bounds(image.shape[1], image.shape[2], transform)
        pixel_size = (abs(float(transform.a)), abs(float(transform.e)))
        crs_wkt = image_src.crs.to_wkt()

    dem = np.full(image.shape[1:], np.nan, dtype=np.float32)
    with rasterio.open(dem_path) as dem_src:
        if dem_src.crs is None:
            raise GeospatialInputError("DEM has no CRS")
        reproject(
            source=rasterio.band(dem_src, 1), destination=dem,
            src_transform=dem_src.transform, src_crs=dem_src.crs,
            src_nodata=dem_src.nodata, dst_transform=transform,
            dst_crs=crs_wkt, dst_nodata=np.nan, resampling=Resampling.bilinear,
        )
    if not np.any(np.isfinite(dem[valid_mask])):
        raise GeospatialInputError("aligned DEM contains no valid elevation inside the AOI")
    return ReferenceData(
        image=image, dem=dem, valid_mask=valid_mask, transform=transform,
        crs_wkt=crs_wkt, pixel_size_m=pixel_size,
        bounds=(float(bounds_obj[0]), float(bounds_obj[1]), float(bounds_obj[2]), float(bounds_obj[3])),
    )
