import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from pathlib import Path
from tqdm import tqdm
from display_helper import color_optimize, ndvi_colorize, ndre_colorize
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from PIL import Image

season = "fall"
csv_path = f"outputs/gedi_valid_all_layers_from_h5_6_{season}1.csv"

out_dir = Path(f"s2_np/21x21_new/{season}1")
# (out_dir / "true").mkdir(parents=True, exist_ok=True)
(out_dir / "img").mkdir(parents=True, exist_ok=True)
(out_dir / "np_s2").mkdir(parents=True, exist_ok=True)
(out_dir / "np_aux").mkdir(parents=True, exist_ok=True)
(out_dir / "np_scl").mkdir(parents=True, exist_ok=True)
# (out_dir / "ndvi").mkdir(parents=True, exist_ok=True)
# (out_dir / "ndre").mkdir(parents=True, exist_ok=True)

# HALF = 5  → 10 x 10 m (at 10 m resolution)
# HALF = 25 → 50 x 50 m, etc.
HALF = 105

path_B02 = f"model/data/s2_6933/{season}/merged/merged_B02_6933.tif"
path_B03 = f"model/data/s2_6933/{season}/merged/merged_B03_6933.tif"
path_B04 = f"model/data/s2_6933/{season}/merged/merged_B04_6933.tif"
path_B05 = f"model/data/s2_6933/{season}/merged/merged_B05_6933.tif"
path_B06 = f"model/data/s2_6933/{season}/merged/merged_B06_6933.tif"
path_B07 = f"model/data/s2_6933/{season}/merged/merged_B07_6933.tif"
path_B08 = f"model/data/s2_6933/{season}/merged/merged_B08_6933.tif"
path_B8A = f"model/data/s2_6933/{season}/merged/merged_B8A_6933.tif"
path_B11 = f"model/data/s2_6933/{season}/merged/merged_B11_6933.tif"
path_B12 = f"model/data/s2_6933/{season}/merged/merged_B12_6933.tif"
path_scl = f"model/data/s2_6933/{season}/merged/merged_SCL_6933.tif"

s2 = "merged1"
layers = {
    "canopy": f"model/data/canopy_6933_{s2}/{season}/reproject_merged_canopy.tif",
    "srtm":   f"model/data/srtm_6933_{s2}/{season}/reproject_merged_srtm.tif",
    "daymet": f"model/data/daymet_6933_{s2}/{season}/reproject_merged_daymet.tif",
    "daymet_tmax": f"model/data/daymet_6933_{s2}/{season}/reproject_merged_daymet_tmax.tif",
    "daymet_tmin": f"model/data/daymet_6933_{s2}/{season}/reproject_merged_daymet_tmin.tif",
    "nlcd":   f"model/data/nlcd_6933_{s2}/{season}/reproject_merged_nlcd.tif",
}




def read_box(src, xmin, ymin, xmax, ymax):
    """Read a window from an already-open raster."""
    win = from_bounds(xmin, ymin, xmax, ymax, src.transform)
    return src.read(1, window=win, boundless=False)


df = pd.read_csv(csv_path)
cols = {c.lower(): c for c in df.columns}

if "lon" not in cols or "lat" not in cols:
    raise ValueError("CSV must contain columns 'lon' and 'lat' in WGS84.")

lon_col = cols["lon"]
lat_col = cols["lat"]

id_col = None
for cand in ("shot_id", "shot_number", "id", "idx"):
    if cand in cols:
        id_col = cols[cand]
        break

lons = df[lon_col].to_numpy(dtype=float)
lats = df[lat_col].to_numpy(dtype=float)


def _open_datasets():
    ds = {}
    # ds['B02'] = rasterio.open(path_B02)
    # ds['B03'] = rasterio.open(path_B03)
    # ds['B04'] = rasterio.open(path_B04)
    # ds['B05'] = rasterio.open(path_B05)
    # ds['B06'] = rasterio.open(path_B06)
    # ds['B07'] = rasterio.open(path_B07)
    # ds['B08'] = rasterio.open(path_B08)
    # ds['B8A'] = rasterio.open(path_B8A)
    # ds['B11'] = rasterio.open(path_B11)
    # ds['B12'] = rasterio.open(path_B12)
    ds['scl'] = rasterio.open(path_scl)
    # ds['canopy'] = rasterio.open(layers['canopy'])
    # ds['srtm'] = rasterio.open(layers['srtm'])
    # ds['day'] = rasterio.open(layers['daymet'])
    # ds['day_tmax'] = rasterio.open(layers['daymet_tmax'])
    # ds['day_tmin'] = rasterio.open(layers['daymet_tmin'])
    # ds['nlcd'] = rasterio.open(layers['nlcd'])
    return ds


def _read_box_from_ds(ds, src_key, xmin, ymin, xmax, ymax):
    src = ds[src_key]
    win = from_bounds(xmin, ymin, xmax, ymax, src.transform)
    return src.read(1, window=win, boundless=False)


def _process_one(args):
    i, x, y, tag = args
    # cache datasets per worker process
    if not globals().get('_DS_CACHE'):
        globals()['_DS_CACHE'] = _open_datasets()
    ds = globals()['_DS_CACHE']

    if not np.isfinite(x) or not np.isfinite(y):
        return (0, 1)

    xmin, xmax = x - HALF, x + HALF
    ymin, ymax = y - HALF, y + HALF

    try:
        # B02 = _read_box_from_ds(ds, 'B02', xmin, ymin, xmax, ymax)
        # B03 = _read_box_from_ds(ds, 'B03', xmin, ymin, xmax, ymax)
        # B04 = _read_box_from_ds(ds, 'B04', xmin, ymin, xmax, ymax)
        # B05 = _read_box_from_ds(ds, 'B05', xmin, ymin, xmax, ymax)
        # B06 = _read_box_from_ds(ds, 'B06', xmin, ymin, xmax, ymax)
        # B07 = _read_box_from_ds(ds, 'B07', xmin, ymin, xmax, ymax)
        # B08 = _read_box_from_ds(ds, 'B08', xmin, ymin, xmax, ymax)
        # B8A = _read_box_from_ds(ds, 'B8A', xmin, ymin, xmax, ymax)
        # B11 = _read_box_from_ds(ds, 'B11', xmin, ymin, xmax, ymax)
        # B12 = _read_box_from_ds(ds, 'B12', xmin, ymin, xmax, ymax)
        SCL = _read_box_from_ds(ds, 'scl', xmin, ymin, xmax, ymax)
        # w_can  = _read_box_from_ds(ds, 'canopy', xmin, ymin, xmax, ymax)
        # w_srtm = _read_box_from_ds(ds, 'srtm', xmin, ymin, xmax, ymax)
        # w_day  = _read_box_from_ds(ds, 'day', xmin, ymin, xmax, ymax)
        # w_day_tmax  = _read_box_from_ds(ds, 'day_tmax', xmin, ymin, xmax, ymax)
        # w_day_tmin  = _read_box_from_ds(ds, 'day_tmin', xmin, ymin, xmax, ymax)
        # w_nlcd = _read_box_from_ds(ds, 'nlcd', xmin, ymin, xmax, ymax)

        # if B04.size == 0 or B03.size == 0 or B02.size == 0 or B08.size == 0 or B05.size == 0:
        #     return (0, 1)

        # rgb  = color_optimize(B04, B03, B02)

        # stack = np.dstack([B02, B03, B04, B05, B06, B07, B8A, B08, B11, B12])
        # stack = np.dstack([SCL])
        # stack = stack.astype("float32") / 10000.0

        # stack2 = np.dstack([w_can, w_srtm, w_day, w_day_tmax, w_day_tmin, w_nlcd])
        # if (not np.all(~np.isnan(stack2))):
        #     return (0, 1)

        stack3 = np.dstack([SCL])

        # np.save(out_dir / "np_s2" / f"s2_{tag}.npy", stack.astype("float32"))
        # np.save(out_dir / "np_aux" / f"aux_{tag}.npy", stack2.astype("float32"))
        np.save(out_dir / "np_scl" / f"s2_scl_{tag}.npy", stack3.astype("float32"))

        # use Pillow for faster image writes
        # if rgb.dtype != np.uint8:
        #     rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype('uint8')
        # else:
        #     rgb_uint8 = rgb
        # img = Image.fromarray(rgb_uint8)
        # img.save(out_dir / "img" / f"s2_{tag}.png")

        return (1, 0)

    except Exception:
        return (0, 1)


def main():
    # build transformer using one S2 band to get CRS
    with rasterio.open(path_B04) as tmp:
        s2_crs = tmp.crs
        if s2_crs is None:
            raise ValueError("S2 band has no CRS; cannot build transformer.")
        to_s2 = Transformer.from_crs("EPSG:4326", s2_crs, always_xy=True)

    print(f"S2 CRS: {s2_crs}")
    print(f"Using HALF={HALF} m window around each GEDI point.")

    xs, ys = to_s2.transform(lons, lats)

    # prepare args for workers
    args = []
    for i in range(min(len(df), 21000)):
        if id_col is not None:
            tag = str(df.at[i, id_col])
        else:
            tag = f"row{i}"
        args.append((i, xs[i], ys[i], tag))

    saved = 0
    skipped = 0

    n_workers = min( max(1, cpu_count() - 1), 12)
    print(f"Processing with {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(_process_one, a): a[0] for a in args}
        for f in tqdm(as_completed(futures), total=len(futures)):
            s, k = f.result()
            saved += s
            skipped += k

    print(f"Done. Saved {saved} PNGs to '{out_dir}'. Skipped {skipped} points.")


if __name__ == '__main__':
    main()
