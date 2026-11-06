from reproject_s2_6933 import reproject_s2_scene_to_6933
from reproject_6933 import reproject_to_match
from reproject_daymet_6933 import reproject_daymet_to_6933
from merge_files import merge_raster
from merge_s2 import merge_s2 
from glob import glob
from pathlib import Path
import rioxarray as rxr
from tqdm import tqdm

if __name__ == "__main__":
    # Merge data files
    # merge_raster(
    #     in_file="output_*.tif",
    #     out_file="merged_srtm.tif"
    # )
    # merge_raster(
    #     in_file="ETH_GlobalCanopyHeight_*.tif",
    #     out_file="merged_canopy.tif"
    # )
    # merge_raster(
    #     in_file="Annual_NLCD_*.tiff",
    #     out_file="merged_nlcd.tif"
    # )

    # Reproject S2 to 6933
    # bands = ["B02","B03","B04","B05","B06","B07","B8A","B08","B11","B12",]
    # bands = ["B12"]
    # bands = ["B12", "B8A"]
    bands = ["SCL"]
    season = "summer"

    # s2_safe_folders = glob(f"S2C")  
    s2_safe_folders = glob(f"{season}/*.SAFE")  
    # season = "spring"
    print(len(s2_safe_folders))
    for safe_path in tqdm(s2_safe_folders):
        safe_path = Path(safe_path)
        out_name = safe_path.name.removesuffix(".SAFE")
        out_dir = Path(f"model/data/s2_6933/{season}") / out_name

        reproject_s2_scene_to_6933(
            safe_path,
            out_dir,
            bands=bands
        )

    # Merge reprojected S2 files
    data_dir = Path(f"model/data/s2_6933/{season}")
    out_dir = data_dir / "merged"
    out_dir.mkdir(exist_ok=True)
    merge_s2(
        data_dir,
        out_dir,
        bands=bands
    )

    # Reproject merged raster files to 6933
    s2 = "merged1"
    # ref_s2_path=f"model/data/s2_6933/{season}/merged/merged_B02_6933.tif"
    ref_s2_path=f"S2C/S2C/B02_epsg6933_10m.tif"
    # ref_da = rxr.open_rasterio(ref_s2_path)
    # reproject_to_match(
    #     "merged_nlcd.tif",
    #     ref_da,
    #     f"S2C/S2C/reproject_merged_nlcd.tif",
    # )
    # reproject_to_match(
    #     "merged_canopy.tif",
    #     ref_da,
    #     f"S2C/S2C/reproject_merged_canopy.tif",
    # )
    # reproject_to_match(
    #     "merged_srtm.tif",
    #     ref_da,
    #     f"S2C/S2C/reproject_merged_srtm.tif",
    # )
    # reproject_daymet_to_6933(
    #     "daymet_v4_prcp_annttl_na_2024 (1).tif",
    #     f"S2C/S2C/reproject_merged_daymet.tif",
    #     ref_s2_path=ref_s2_path
    # )
    # reproject_daymet_to_6933(
    #     "daymet_v4_tmin_annavg_na_2024.tif",
    #     f"S2C/S2C/reproject_merged_daymet_tmin.tif",
    #     ref_s2_path=ref_s2_path
    # )
    # reproject_daymet_to_6933(
    #     "daymet_v4_tmax_annavg_na_2024.tif",
    #     f"S2C/S2C/reproject_merged_daymet_tmax.tif",
    #     ref_s2_path=ref_s2_path
    # )