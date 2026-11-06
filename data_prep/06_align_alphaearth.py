"""
Align (warp + crop) an AlphaEarth GeoTIFF to match a known-working S2 GeoTIFF grid.

Goal:
- Given:
    1) ref S2 GeoTIFF (your "known working" grid, e.g., merged_B02_6933.tif or your s2_stack.tif)
    2) alpha_export.tif (from Earth Engine; any CRS/transform/shape)
- Produce:
    alpha_aligned.tif with:
      - SAME CRS as ref
      - SAME pixel grid (transform) as ref for the chosen window
      - EXACT SAME shape as the chosen window (e.g., 64 x 1000 x 1000)

This mirrors your existing S2/AUX window snapping approach:
- transform center -> ref CRS
- build bbox (10km x 10km)
- from_bounds(..., ref.transform) + round_offsets/round_lengths -> win
- warp AlphaEarth onto that window transform/shape

Usage:
    python align_alpha_to_ref.py
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import transform as win_transform
from rasterio.warp import reproject, Resampling
from pyproj import Transformer


def bbox_from_center(x, y, size_m):
    half = size_m / 2.0
    return (x - half, y - half, x + half, y + half)  # xmin, ymin, xmax, ymax


def compute_snapped_window_on_ref(
    ref_tif: str,
    center_lon: float,
    center_lat: float,
    aoi_size_m: float = 10_000,
):
    """Return (win, bbox_ref_crs) where win is snapped to ref grid."""
    with rasterio.open(ref_tif) as ref:
        # lon/lat -> ref CRS
        to_ref = Transformer.from_crs("EPSG:4326", ref.crs, always_xy=True)
        cx, cy = to_ref.transform(center_lon, center_lat)

        bbox = bbox_from_center(cx, cy, aoi_size_m)

        win = from_bounds(*bbox, transform=ref.transform)
        win = win.round_offsets().round_lengths()

    return win, bbox


def warp_alpha_to_ref_window(
    ref_tif: str,
    alpha_src_tif: str,
    out_alpha_tif: str,
    window,
    resampling=Resampling.bilinear,  # embeddings are continuous
):
    """
    Warp AlphaEarth (any grid) -> exact ref grid for `window`.
    Output will have:
      - dst_crs = ref.crs
      - dst_transform = window transform on ref grid
      - dst shape = window (width/height)
      - bands = alpha_src.count (should be 64)
    """
    with rasterio.open(ref_tif) as ref, rasterio.open(alpha_src_tif) as src:
        dst_crs = ref.crs
        dst_transform = win_transform(window, ref.transform)
        dst_h = int(window.height)
        dst_w = int(window.width)

        # Build output profile
        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            count=src.count,
            height=dst_h,
            width=dst_w,
            transform=dst_transform,
            crs=dst_crs,
            dtype="float32",
            nodata=None,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        Path(out_alpha_tif).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_alpha_tif, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                dst_arr = np.zeros((dst_h, dst_w), dtype=np.float32)

                reproject(
                    source=rasterio.band(src, b),
                    destination=dst_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                    src_nodata=src.nodata,
                    dst_nodata=None,
                )
                dst.write(dst_arr, b)

    print(f"Wrote aligned alpha: {out_alpha_tif}  ->  ({src.count}, {dst_h}, {dst_w})")


if __name__ == "__main__":
    # ------------------- EDIT THESE -------------------
    # Use your known-working reference grid.
    # Best ref is the SAME raster you used to make your 1000x1000 S2/AUX stacks:
    # e.g. merged_B02_6933.tif OR your s2_stack_*.tif (both are fine if grid is same)
    ref_tif = "data/stacks/s2_stack_50km_summer_umbs.tif"

    # AlphaEarth GeoTIFF exported from Earth Engine (any CRS/shape is ok)
    alpha_src_tif = "alphaearth_patch_2024_15km_2.tif"

    # Output aligned alpha (will match ref grid window exactly)
    out_alpha_tif = "alphaearth_2024_10km_aligned_to_ref.tif"

    # Center coordinate (lon/lat)
    center_lon=-85.424818
    center_lat=42.630143

    AOI_SIZE_M = 10_000  # 10 km -> should yield 1000x1000 if ref pixel size is 10m
    # ---------------------------------------------------

    win, bbox = compute_snapped_window_on_ref(
        ref_tif=ref_tif,
        center_lon=center_lon,
        center_lat=center_lat,
        aoi_size_m=AOI_SIZE_M,
    )

    print("BBox on ref CRS:", bbox)
    print("Snapped window:", win)
    print("Output pixels (W,H):", int(win.width), int(win.height))

    warp_alpha_to_ref_window(
        ref_tif=ref_tif,
        alpha_src_tif=alpha_src_tif,
        out_alpha_tif=out_alpha_tif,
        window=win,
        resampling=Resampling.bilinear,
    )
