# Getting the data

No data is stored in this repository; everything is public. The study area is centered on central Michigan
(42.83° N, 84.78° W) with a 350 km radius for GEDI shots. All rasters are reprojected to the equal-area
grid **EPSG:6933** at 10 m.

You need free accounts with:

- **NASA Earthdata** (https://urs.earthdata.nasa.gov) for GEDI, MODIS, SRTM and Daymet
- **Copernicus Data Space** (https://dataspace.copernicus.eu) for Sentinel-2
- **Google Earth Engine** with a Cloud project (https://earthengine.google.com) for AlphaEarth

## 1. Sources

| Dataset | What to download | Where |
|---|---|---|
| GEDI L4A v2.1 | `GEDI04_A_*.h5` granules for 2024–2025 that intersect the study area | [NASA Earthdata Search](https://search.earthdata.nasa.gov), search "GEDI L4A Footprint Level Aboveground Biomass Density V2.1" (doi:10.3334/ORNLDAAC/2056) and filter by bounding box and date |
| Sentinel-2 L2A | `.SAFE` products for spring, summer and fall 2024–2025, low cloud cover. The 50 km inference AOI is tile T16TFN | [Copernicus Browser](https://browser.dataspace.copernicus.eu), collection "Sentinel-2 L2A" |
| AlphaEarth embeddings | 2024 annual 64-band embedding, sampled at every GEDI shot and exported for inference AOIs | Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, see step 6 |
| Annual NLCD 2024 | Land cover, CONUS | [MRLC Annual NLCD](https://www.mrlc.gov/data) |
| ETH Global Canopy Height 2020 | 10 m tiles covering N39–N45, W084–W090 | [ETH Zurich / Lang et al. 2023](https://langnico.github.io/globalcanopyheight/) |
| SRTMGL1 v3 | 30 m DEM over the study area | NASA Earthdata or [OpenTopography](https://opentopography.org) |
| Daymet V4 | 2024 annual precipitation total and annual mean Tmax/Tmin, North America | [ORNL DAAC Daymet](https://daymet.ornl.gov) |
| TreeMap 2022 CONUS | `TreeMap2022_CONUS.tif` and `TreeMap2022_CONUS_Tree_Table.csv` | [USDA Forest Service Research Data Archive](https://www.fs.usda.gov/rds/archive/) (search "TreeMap 2022") |
| BAAD | `baad_data.csv` | [github.com/dfalster/baad](https://github.com/dfalster/baad) (Falster et al. 2015) |
| MODIS MCD15A2H v061 | 8-day LAI and QC, 2020–2026, over the harmonic study AOI | [AppEEARS](https://appeears.earthdatacloud.nasa.gov) area request (GeoTIFF output), then `leaf_prior/download_appeears.py` |

Jenkins et al. (2003) foliage ratio coefficients are built into `leaf_prior/prepare_leaf_prior.py`, so they need no
download.

## 2. Expected layout

Most scripts run from `model/` or `model/data/` and use relative paths. The `season` variable
(`spring`, `summer`, `fall`) at the top of each script selects which set to process.

```
model/data/
  GEDI/{season}/GEDI04_A_*.h5
  {season}/*.SAFE                         Sentinel-2 L2A products
  s2_6933/{season}/merged/merged_B02_6933.tif … merged_SCL_6933.tif
  canopy_6933_merged1/{season}/reproject_merged_canopy.tif
  srtm_6933_merged1/{season}/reproject_merged_srtm.tif
  daymet_6933_merged1/{season}/reproject_merged_daymet*.tif
  nlcd_6933_merged1/{season}/reproject_merged_nlcd.tif
  outputs/gedi_valid_all_layers_from_h5_6_{season}.csv
  outputs/alphaearth_vec_2024_{season}.csv
  s2_np/21x21_new/{season}/np_s2/s2_{id}.npy
  s2_np/21x21_new/{season}/np_scl/s2_scl_{id}.npy
  s2_np/21x21_new/{season}/img/s2_{id}.png
model/nlcd_4326_mi.tif                    NLCD clipped to Michigan, EPSG:4326 (forest-type lookup)
model/invalid_img_ids.npy                 ids of patches with no-data pixels (from model/helper.py)
```

## 3. Processing order

| Step | Script | Output |
|---|---|---|
| 1 | `data_prep/01_merge_reproject_s2.py` | Each band of every `.SAFE` reprojected to EPSG:6933 at 10 m and mosaicked per season |
| 2 | `data_prep/02_reproject_aux_6933.py`, `02b_reproject_daymet_6933.py` | Canopy height, SRTM, Daymet and NLCD snapped onto the S2 grid |
| 3 | `data_prep/03_extract_gedi_shots.py` | Quality-filtered GEDI shots (`l4_quality_flag`, `degrade_flag`) within 350 km, with per-layer validity flags, written to `outputs/gedi_valid_all_layers_from_h5_6_{season}.csv` |
| 4 | `data_prep/04_extract_patches.py` | 21 × 21 S2 and SCL patches (`.npy`) and RGB previews (`.png`) per shot |
| 5 | `data_prep/05_combine_seasons.py` | Seasons concatenated into the `all` CSV, with spring and summer ids offset by the preceding season counts. Copy each season's patches into `s2_np/21x21_new/all/` with the same id offsets |
| 6 | `data_prep/export_alphaearth_points.py` | Per-shot AlphaEarth CSV exported from Earth Engine to Google Drive |
| 7 | `model/helper.py` → `get_invalid_img` | Patches with no-data (black) pixels listed in `invalid_img_ids.npy` |
| 8 | `model/main.py` | Train or evaluate |
| 9 | `data_prep/07_build_inference_stacks.py`, `06_align_alphaearth.py`, `export_alphaearth_aoi.ipynb` | Wall-to-wall S2 and AlphaEarth stacks for an AOI, aligned to the same grid |
| 10 | `model/inference_map.py` | 10 m AGBD, leaf and wood GeoTIFFs and `agbd_timeseries.csv` |

**Leaf prior and phenology** (independent of the steps above, except where noted):

| Step | Script | Notes |
|---|---|---|
| A | `leaf_prior/prepare_leaf_prior.py` | Needs NLCD (EPSG:4326, Michigan), TreeMap reprojected to the NLCD grid (`treemap_aligned.tif`, nearest neighbor), the TreeMap tree table and BAAD |
| B | `leaf_prior/harmonic_decomposition.py` | Reads `agbd_timeseries.csv` from step 10 |
| C | `leaf_prior/download_appeears.py` | Set `TASK_ID` to your AppEEARS request. It asks for your Earthdata login at runtime and stores nothing |
| D | `leaf_prior/validate_lai.py` | Compares MODIS LAI with the harmonic leaf component |

## 4. AlphaEarth CSV format

`model/dataset.py` reads the AlphaEarth table positionally. The columns must be `system:index, A00 … A63, id, .geo`,
which is the default layout of an Earth Engine CSV export from `sampleRegions` with an `id` property.
`export_alphaearth_points.py` produces exactly this. Shots that fall on masked pixels are missing from the export.
Drop them from the GEDI CSV before training.

## 5. Notes

- GEDI L4A AGBD is in Mg/ha. The quality filter keeps `l4_quality_flag == 1` and `degrade_flag == 0`.
- Patches whose center pixel is cloud (SCL classes 3, 8, 9, 10), or that are more than 20% cloud, are skipped
  during training and evaluation.
- Both checkpoints were trained on raw L2A digital numbers (patches are not divided by 10,000). `inference_map.py`
  multiplies its input stacks by `S2_SCALE = 1/10000`. If your inference stacks are plain L2A digital numbers like
  the training patches, set `S2_SCALE = None`, so inputs match training.
