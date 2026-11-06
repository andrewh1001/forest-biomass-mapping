import rioxarray as rxr
from rasterio.enums import Resampling
from pathlib import Path
import numpy as np

DAYMET_CRS = (
    "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 "
    "+lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

def reproject_daymet_to_6933(
    input_path,
    output_path,
    ref_s2_path=None,   # Optional: align to S2 6933 grid
    nodata_value=-9999.0
):
    """
    Reproject Daymet raster to EPSG:6933. If ref_s2_path is provided,
    the output will be pixel-aligned to the S2 raster grid.
    
    Parameters:
        input_path: str or Path -- Path to original Daymet TIFF
        output_path: str or Path -- Where to save the reprojected TIFF
        ref_s2_path: str or Path -- Optional S2 band already in EPSG:6933
        nodata_value: float -- value to use for nodata (Daymet default = -9999)
    """
    print("Processing:", input_path)
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    da = rxr.open_rasterio(input_path, BIGTIFF="YES").squeeze()

    # 2) If CRS missing, assign Daymet LCC CRS
    if da.rio.crs is None:
        da = da.rio.write_crs(DAYMET_CRS)

    # 3) Safely handle dtype so nodata works
    if str(da.dtype).startswith("int"):
        da = da.astype("float32")

    da = da.rio.write_nodata(float(nodata_value))

    # 4) Reproject
    if ref_s2_path is None:
        da_6933 = da.rio.reproject(
            "EPSG:6933",
            resampling=Resampling.bilinear
        )
    else:
        # Align to Sentinel-2 grid
        ref = rxr.open_rasterio(ref_s2_path)
        da_6933 = da.rio.reproject_match(ref, resampling=Resampling.bilinear)

    # 5) Save output
    da_6933.rio.to_raster(output_path, BIGTIFF="YES", compress="LZW")
    print("Wrote:", output_path)
    da.close()


# s2 = "20251119T162611"
# reproject_daymet_to_6933(
#     "model/data/daymet_v4_prcp_annttl_na_2024 (1).tif",
#     f"model/data/model/data/daymet_6933_{s2}/daymet_prcp_annttl_na_2024.tif",
#     ref_s2_path="model/data/model/data/s2_6933/20251119T162611/S2C_MSIL2A_20251119T162611_N0511_R040_T16TGN_20251119T200117.SAFE_B02_epsg6933_10m.tif"
# )
