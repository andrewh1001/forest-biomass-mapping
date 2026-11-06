from pathlib import Path
import rioxarray as rxr
from rasterio.enums import Resampling
from reproject_daymet_6933 import reproject_daymet_to_6933

def reproject_to_match(
    in_path,
    ref_da,
    out_path,
    resampling=Resampling.nearest,
):
    print("Processing:", in_path)
    da = rxr.open_rasterio(in_path)
    da_6933 = da.rio.reproject_match(ref_da, resampling=resampling)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da_6933.rio.to_raster(out_path, compress="LZW")
    print("Wrote:", out_path)



if __name__ == "__main__":
    ref_s2_path="model/data/model/data/s2_6933/merged/merged_B02_6933.tif"
    ref_da = rxr.open_rasterio(ref_s2_path)
    s2 = "merged1"
    reproject_to_match(
        "merged_nlcd.tif",
        ref_da,
        f"model/data/nlcd_6933_{s2}/reproject_merged_nlcd.tif",
    )
    reproject_to_match(
        "merged_canopy.tif",
        ref_da,
        f"model/data/canopy_6933_{s2}/reproject_merged_canopy.tif",
    )
    reproject_to_match(
        "merged_srtm.tif",
        ref_da,
        f"model/data/srtm_6933_{s2}/reproject_merged_srtm.tif",
    )
    reproject_daymet_to_6933(
        "model/data/daymet_v4_prcp_annttl_na_2024 (1).tif",
        f"model/data/daymet_6933_{s2}/reproject_merged_daymet.tif",
        ref_s2_path=ref_s2_path
    )

