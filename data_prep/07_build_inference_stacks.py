from pathlib import Path
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from display_helper import color_optimize
import os
from glob import glob

s2_dir = glob("test_files/S2_*TFN*")
season = "spring"
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

# print(os.getcwd())

# path_B02 = os.path.join(os.getcwd(),f"S2C/S2C/B02_epsg6933_10m.tif")
# path_B03 = os.path.join(os.getcwd(),f"S2C/S2C/B03_epsg6933_10m.tif")
# path_B04 = os.path.join(os.getcwd(),f"S2C/S2C/B04_epsg6933_10m.tif")
# path_B05 = os.path.join(os.getcwd(),f"S2C/S2C/B05_epsg6933_20m.tif")
# path_B06 = os.path.join(os.getcwd(),f"S2C/S2C/B06_epsg6933_20m.tif")
# path_B07 = os.path.join(os.getcwd(),f"S2C/S2C/B07_epsg6933_20m.tif")
# path_B08 = os.path.join(os.getcwd(),f"S2C/S2C/B08_epsg6933_10m.tif")
# path_B8A = os.path.join(os.getcwd(),f"S2C/S2C/B8A_epsg6933_20m.tif")
# path_B11 = os.path.join(os.getcwd(),f"S2C/S2C/B11_epsg6933_20m.tif")
# path_B12 = os.path.join(os.getcwd(),f"S2C/S2C/B12_epsg6933_20m.tif")

s2 = "merged1"
layers = {
    "canopy": f"model/data/canopy_6933_{s2}/{season}/reproject_merged_canopy.tif",
    "srtm":   f"model/data/srtm_6933_{s2}/{season}/reproject_merged_srtm.tif",
    "daymet": f"model/data/daymet_6933_{s2}/{season}/reproject_merged_daymet.tif",
    "daymet_tmax": f"model/data/daymet_6933_{s2}/{season}/reproject_merged_daymet_tmax.tif",
    "daymet_tmin": f"model/data/daymet_6933_{s2}/{season}/reproject_merged_daymet_tmin.tif",
    "nlcd":   f"model/data/nlcd_6933_{s2}/{season}/reproject_merged_nlcd.tif",
}
# layers = {
#     "canopy": f"S2C/S2C/reproject_merged_canopy.tif",
#     "srtm":   f"S2C/S2C/reproject_merged_srtm.tif",
#     "daymet": f"S2C/S2C//reproject_merged_daymet.tif",
#     "daymet_tmax": f"S2C/S2C/reproject_merged_daymet_tmax.tif",
#     "daymet_tmin": f"S2C/S2C/reproject_merged_daymet_tmin.tif",
#     "nlcd":   f"S2C/S2C/reproject_merged_nlcd.tif",
# }
S2_BANDS = [path_B02, path_B03, path_B04, path_B05, path_B06, path_B07, path_B8A, path_B08, path_B11, path_B12]
AUX_ORDER = ["canopy", "srtm", "daymet", "daymet_tmax", "daymet_tmin", "nlcd"]

# center_lat, center_lon = 45.558781390006594, -84.67739353959658
# center_lat, center_lon = 43.80508,-83.97742
center_lon=-85.424818
center_lat=42.630143

center_x_6933 = None
center_y_6933 = None

AOI_SIZE_M = 5_000  # 10 km square

out_dir = Path("stacks")
out_dir.mkdir(parents=True, exist_ok=True)
out_s2  = out_dir / f"s2_stack_{1}.tif"
out_aux = out_dir / f"aux_stack_{1}.tif"


def bbox_from_center(x, y, size_m):
    half = size_m / 2.0
    return (x - half, y - half, x + half, y + half)  # xmin, ymin, xmax, ymax


def assert_aligned(ref_ds, other_path):
    with rasterio.open(other_path) as ds:
        if ds.crs != ref_ds.crs:
            raise ValueError(f"CRS mismatch:\nref={ref_ds.crs}\n{other_path}={ds.crs}")
        if ds.transform != ref_ds.transform:
            raise ValueError(f"Transform mismatch for {other_path}")
        # if (ds.width != ref_ds.width) or (ds.height != ref_ds.height):
        #     print(ds.width)
        #     print(ref_ds.width)
        #     print(ds.height)
        #     print(ref_ds.height)
        #     raise ValueError(f"Shape mismatch for {other_path}")


def write_stack(out_path, ref_ds, band_paths, window):
    profile = ref_ds.profile.copy()
    profile.update(
        driver="GTiff",
        count=len(band_paths),
        height=int(window.height),
        width=int(window.width),
        transform=rasterio.windows.transform(window, ref_ds.transform),
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=None,
        dtype="float32",
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, p in enumerate(band_paths, start=1):
            with rasterio.open(p) as ds:
                arr = ds.read(1, window=window).astype("float32", copy=False)
                print(arr)
                dst.write(arr, i)


def main():
    # with rasterio.open(path_B02) as ref:
    #     for p in S2_BANDS[1:]:
    #         assert_aligned(ref, p)
    #     # for k in AUX_ORDER:
    #     #     assert_aligned(ref, layers[k])

    #     # center -> EPSG:6933
    #     if center_x_6933 is not None and center_y_6933 is not None:
    #         cx, cy = center_x_6933, center_y_6933
    #     else:
    #         to_6933 = Transformer.from_crs("EPSG:4326", ref.crs, always_xy=True)
    #         cx, cy = to_6933.transform(center_lon, center_lat)

    #     bbox = bbox_from_center(cx, cy, AOI_SIZE_M)

    #     # window on the aligned grid
    #     win = from_bounds(*bbox, transform=ref.transform)
    #     win = win.round_offsets().round_lengths()

    #     print("AOI bbox (6933):", bbox)
    #     print("Window:", win)
    #     print("Output pixels (W,H):", int(win.width), int(win.height))

    #     print("Writing:", out_s2)
    #     write_stack(str(out_s2), ref, S2_BANDS, win)

    #     # aux_paths = [layers[k] for k in AUX_ORDER]
    #     # print("Writing:", out_aux)
    #     # write_stack(str(out_aux), ref, aux_paths, win)

    # print("Done.")
    s2_dir = glob("test_files/S2_*TFN*")

    print(len(s2_dir))
    for file in s2_dir:
        print(file)

if __name__ == "__main__":
    main()
