import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point
import rasterio
from pathlib import Path
from glob import glob
from rasterio.windows import from_bounds
from tqdm import tqdm

HALF = 105
COMMON_CRS = "EPSG:6933"

# h5_paths = [
#     "model/data/GEDI04_A_2025066084001_O35306_03_T10149_02_004_01_V002.h5",
#     "model/data/GEDI04_A_2025039190333_O34894_03_T08527_02_004_01_V002.h5",
#     "model/data/GEDI04_A_2025054132536_O35123_03_T03539_02_004_01_V002.h5",
# ]


# center in lon/lat and radius in km
center_lat, center_lon = 42.83088331386743, -84.77989676474816
radius_km = 350

s2 = "merged1"
season = "fall"
layers = {
    "canopy":      f"model/data/canopy_6933_{s2}/{season}/reproject_merged_canopy.tif",
    "srtm":        f"model/data/srtm_6933_{s2}/{season}/reproject_merged_srtm.tif",
    "daymet_prcp": f"model/data/daymet_6933_{s2}/{season}/reproject_merged_daymet.tif",
    "daymet_tmax": f"model/data/daymet_6933_{s2}/{season}/reproject_merged_daymet_tmax.tif",
    "daymet_tmin": f"model/data/daymet_6933_{s2}/{season}/reproject_merged_daymet_tmin.tif",
    "nlcd":        f"model/data/nlcd_6933_{s2}/{season}/reproject_merged_nlcd.tif",
}

h5_paths = glob(f"GEDI/{season}/GEDI04_*.h5")
# h5_paths = glob(f"model/GEDI04_*.h5")
# h5_paths = glob("model/GEDI04_A_2024174094541_O31305_02_T10513_02_004_01_V002.h5")

out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv  = out_dir / f"gedi_valid_all_layers_from_h5_6_{season}_umbs_2.csv"
out_gpkg = out_dir / f"gedi_valid_all_layers_from_h5_{season}.gpkg"
# --------------------------

def _get_any(group, names):
    """
    Try multiple dataset names and common subgroups; return array or None.
    This handles small naming differences between GEDI versions.
    """
    for n in names:
        if n in group:
            return group[n][...]
    for sub in ("agbd_prediction", "geolocation", "quality", "l4a_production"):
        if sub in group:
            for n in names:
                if n in group[sub]:
                    return group[sub][n][...]
    return None


def read_gedi_l4a_points_one_file(h5_path):
    """
    Read lon/lat/AGBD + quality flags from a single GEDI L4A HDF5 file.
    Returns a pandas DataFrame with columns: lon, lat, agbd, beam, flags..., file_id
    """
    rows = []
    with h5py.File(h5_path, "r") as f:
        beams = [k for k in f.keys() if k.startswith("BEAM")]
        if not beams:
            print(f"[WARN] No BEAM groups found in GEDI file: {h5_path}")
            return pd.DataFrame()

        for b in beams:
            g = f[b]
            lon  = _get_any(g, ["lon_lowestmode", "longitude_lowestmode", "lon", "longitude"])
            lat  = _get_any(g, ["lat_lowestmode", "latitude_lowestmode", "lat", "latitude"])
            agbd = _get_any(g, ["agbd", "agb"])
            if lon is None or lat is None or agbd is None:
                continue

            l4q  = _get_any(g, ["l4_quality_flag", "l4a_quality_flag", "quality_flag"])
            l2q  = _get_any(g, ["l2_quality", "l2_quality_flag", "l2a_quality_flag"])
            degr = _get_any(g, ["degrade_flag", "degraded_flag"])

            n = len(np.asarray(agbd))

            def arr(a, fill=np.nan):
                if a is None:
                    return np.full(n, fill)
                a = np.asarray(a)
                if a.ndim == 0:
                    a = np.full(n, a)
                return a

            rows.append(pd.DataFrame({
                "file": h5_path,
                "beam": b,
                "lon":  np.asarray(lon),
                "lat":  np.asarray(lat),
                "agbd": np.asarray(agbd, dtype="float32"),
                "l4_quality_flag": arr(l4q),
                "l2_quality":      arr(l2q),
                "degrade_flag":    arr(degr),
            }))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def read_gedi_l4a_points_many(h5_paths):
    """
    Read multiple GEDI HDF5 files and concatenate all points.
    """
    dfs = []
    print(h5_paths)
    print(f"Total: {len(h5_paths)}")
    for p in h5_paths:
        print(p)
        print(f"Reading GEDI file: {p}")
        df_one = read_gedi_l4a_points_one_file(p)
        if not df_one.empty:
            dfs.append(df_one)
    if not dfs:
        raise RuntimeError("No GEDI points with (lon, lat, agbd) found in any file.")
    return pd.concat(dfs, ignore_index=True)


# 1) Read + quality filter GEDI shots (in lon/lat, across all files)
df = read_gedi_l4a_points_many(h5_paths)

valid = np.isfinite(df.agbd)
if "l4_quality_flag" in df:
    valid &= (pd.to_numeric(df.l4_quality_flag, errors="coerce") == 1)
if "l2_quality" in df:
    valid &= (pd.to_numeric(df.l2_quality, errors="coerce") >= 1)
if "degrade_flag" in df:
    valid &= (pd.to_numeric(df.degrade_flag, errors="coerce") == 0)
valid &= (df.agbd >= 0) & (df.agbd <= 600)  # physical sanity

dfv = df.loc[valid, ["file", "beam", "lon", "lat", "agbd"]].copy()
print(f"Total GEDI shots (all files): {len(df):,} | Valid by flags: {len(dfv):,}")


# 2) Radius filter in lon/lat, implemented via projection to EPSG:6933

to_6933 = Transformer.from_crs("EPSG:4326", COMMON_CRS, always_xy=True)

# NOTE: Transformer expects (lon, lat)
cx, cy = to_6933.transform(center_lon, center_lat)
x_6933, y_6933 = to_6933.transform(dfv["lon"].values, dfv["lat"].values)

dfv["x_6933"] = x_6933
dfv["y_6933"] = y_6933

dx = x_6933 - cx
dy = y_6933 - cy
dist_m = np.hypot(dx, dy)

mask_radius = dist_m <= (radius_km * 1000.0)
dfv = dfv.loc[mask_radius].copy()

print(f"Valid GEDI shots within {radius_km} km: {len(dfv):,}")
if dfv.empty:
    print("No valid GEDI within the search radius. Exiting.")
    raise SystemExit


# 3) Turn into GeoDataFrame in EPSG:6933 (for later convenience)
gdf = gpd.GeoDataFrame(
    dfv,
    geometry=gpd.points_from_xy(dfv["x_6933"], dfv["y_6933"]),
    crs=COMMON_CRS,
)

gdf["x"] = gdf["x_6933"]
gdf["y"] = gdf["y_6933"]

layer_names = list(layers.keys())

# We'll store a boolean "is_valid_<layer>" per shot for debugging
for ln in layer_names:
    gdf[f"is_valid_{ln}"] = False

valid_all = np.ones(len(gdf), dtype=bool)
gdf = gdf.reset_index(drop=True)
# Open all rasters once

# print((gdf.head()))
# print(np.shape(valid_all))
with rasterio.open(layers["canopy"]) as ds_can, \
     rasterio.open(layers["srtm"])   as ds_srtm, \
     rasterio.open(layers["daymet_prcp"]) as ds_day, \
     rasterio.open(layers["daymet_tmax"]) as ds_day_tmax, \
     rasterio.open(layers["daymet_tmin"]) as ds_day_tmin, \
     rasterio.open(layers["nlcd"])   as ds_nlcd:

    # Sanity check CRS
    for name, ds in zip(layer_names, [ds_can, ds_srtm, ds_day, ds_nlcd, ds_day_tmax, ds_day_tmin]):
        if ds.crs is None or ds.crs.to_string() != COMMON_CRS:
            raise ValueError(f"Layer {name} is not {COMMON_CRS} (got {ds.crs}).")

    # Check resolution for expected 10x10 patch
    resx, resy = ds_can.res
    expected_nx = int(round((2 * HALF) / abs(resx)))  # 100 / 10 = 10
    expected_ny = int(round((2 * HALF) / abs(resy)))  # 10

    def mask_nodata(arr, ds):
        nd = ds.nodata
        if nd is not None:
            arr = np.where(np.isclose(arr, nd), np.nan, arr)
        return arr

    def frac_finite(arr):
        total = arr.size
        if total == 0:
            return 0.0
        return np.isfinite(arr).sum() / total

    valid1 = 0
    valid2 = 0
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
        x = row["x"]; y = row["y"]

        xmin, xmax = x - HALF, x + HALF
        ymin, ymax = y - HALF, y + HALF

        w_can  = from_bounds(xmin, ymin, xmax, ymax, ds_can.transform)
        w_srtm = from_bounds(xmin, ymin, xmax, ymax, ds_srtm.transform)
        w_day  = from_bounds(xmin, ymin, xmax, ymax, ds_day.transform)
        w_day_tmax  = from_bounds(xmin, ymin, xmax, ymax, ds_day_tmax.transform)
        w_day_tmin  = from_bounds(xmin, ymin, xmax, ymax, ds_day_tmin.transform)
        w_nlcd = from_bounds(xmin, ymin, xmax, ymax, ds_nlcd.transform)

        try:
            arr_can  = ds_can.read(1, window=w_can, boundless=False)
            arr_srtm = ds_srtm.read(1, window=w_srtm, boundless=False)
            arr_day  = ds_day.read(1, window=w_day, boundless=False)
            arr_day_tmax  = ds_day_tmax.read(1, window=w_day_tmax, boundless=False)
            arr_day_tmin  = ds_day_tmin.read(1, window=w_day_tmin, boundless=False)
            arr_nlcd = ds_nlcd.read(1, window=w_nlcd, boundless=False)
        except Exception as e:
            print(e)
            valid_all[idx] = False
            continue

        # Reject empty or wrong-size windows
        if (arr_can.shape != (expected_ny, expected_nx) or
            arr_srtm.shape != (expected_ny, expected_nx) or
            arr_day.shape  != (expected_ny, expected_nx) or
            arr_day_tmax.shape  != (expected_ny, expected_nx) or
            arr_day_tmin.shape  != (expected_ny, expected_nx) or
            arr_nlcd.shape != (expected_ny, expected_nx)):
            valid_all[idx] = False
            continue

        arr_can  = mask_nodata(arr_can, ds_can)
        arr_srtm = mask_nodata(arr_srtm, ds_srtm)
        arr_day  = mask_nodata(arr_day, ds_day)
        arr_day_tmax  = mask_nodata(arr_day_tmax, ds_day_tmax)
        arr_day_tmin  = mask_nodata(arr_day_tmin, ds_day_tmin)
        arr_nlcd = mask_nodata(arr_nlcd, ds_nlcd)

        arr_can = np.nan_to_num(arr_can, nan=0.0)

        # all finite
        is_valid_can  = np.all(np.isfinite(arr_can))
        is_valid_srtm = np.all(np.isfinite(arr_srtm))
        is_valid_day  = np.all(np.isfinite(arr_day))
        is_valid_day_tmax  = np.all(np.isfinite(arr_day_tmax))
        is_valid_day_tmin  = np.all(np.isfinite(arr_day_tmin))
        is_valid_nlcd = np.all(np.isfinite(arr_nlcd))

        # Or relaxed version (e.g. >= 95% finite):
        # is_valid_can  = frac_finite(arr_can)  >= 0.95
        # is_valid_srtm = frac_finite(arr_srtm) >= 0.95
        # is_valid_day  = frac_finite(arr_day)  >= 0.95
        # is_valid_nlcd = frac_finite(arr_nlcd) >= 0.95

        gdf.at[idx, "is_valid_canopy"] = is_valid_can
        gdf.at[idx, "is_valid_srtm"]   = is_valid_srtm
        gdf.at[idx, "is_valid_daymet_prcp"] = is_valid_day
        gdf.at[idx, "is_valid_daymet_tmax"] = is_valid_day_tmax
        gdf.at[idx, "is_valid_daymet_tmin"] = is_valid_day_tmin
        gdf.at[idx, "is_valid_nlcd"]   = is_valid_nlcd

        # if not is_valid_can:
        #     valid1 += 1
        #     to_4326 = Transformer.from_crs(COMMON_CRS, "EPSG:4326", always_xy=True)
        #     a, b = to_4326.transform(x, y)
        #     print(a)
        #     print(b)
        #     print(arr_can)
        #     exit()
        # if not is_valid_srtm:
        #     valid2 += 1

        valid_all[idx] = is_valid_can and is_valid_srtm and is_valid_day and is_valid_nlcd and is_valid_day_tmax and is_valid_day_tmin
        
# print(valid1)
# print(valid2)

print("\n--- Per-layer 100m-window validity summary (within radius) ---")
N = len(gdf)
for ln in layer_names:
    mask_layer = gdf[f"is_valid_{ln}"].values
    vcnt = int(mask_layer.sum())
    print(f"{ln:10s}: valid_window={vcnt:,}  invalid_window={N - vcnt:,}")

all_valid_cnt = int(valid_all.sum())
print(f"\nALL layers valid over 100m window: {all_valid_cnt:,} / {N:,}")

# 6) Save only shots with all 4 layers valid over the 100m window
gdf_valid = gdf.loc[valid_all].copy()
gdf_valid = gdf_valid.reset_index(drop=True)
gdf_valid["id"] = gdf_valid.index

# Also store lon/lat explicitly
gdf_valid_wgs84 = gdf_valid.to_crs("EPSG:4326")
gdf_valid["lon"] = gdf_valid_wgs84.geometry.x
gdf_valid["lat"] = gdf_valid_wgs84.geometry.y

# Save GPKG with geometry in WGS84 and CSV without geometry
gdf_valid_out = gpd.GeoDataFrame(
    gdf_valid,
    geometry=gpd.points_from_xy(gdf_valid["lon"], gdf_valid["lat"]),
    crs="EPSG:4326",
)
gdf_valid_out.to_file(out_gpkg, driver="GPKG")
gdf_valid_out.drop(columns=["geometry"], errors="ignore").to_csv(out_csv, index=False)

print(f"\nWrote CSV  → {out_csv}")
print(f"Wrote GPKG → {out_gpkg}")
