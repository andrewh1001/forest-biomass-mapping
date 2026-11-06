from torch.utils.data import Dataset
import os
import re
import torch
import numpy as np
import datetime

try:
    import rasterio
    from rasterio.transform import rowcol
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("[dataset] Warning: rasterio not installed. Forest type will default to 'mixed'.")


class MultiBandAGBDDataset(Dataset):
    def __init__(
        self,
        df,
        alpha_df,
        patch_dir_s2,
        patch_dir_scl,
        target_col,
        alpha_mean,
        alpha_std,
        cache=True,
        scale_to_reflectance: bool = True,
        reflectance_denominator: float = 10000.0,
        snow_code=11,
        cloud_codes=(3, 8, 9, 10),
        nlcd_path: str = None,          # path to NLCD GeoTIFF for Michigan
    ):
        self.df = df
        self.alpha_df = alpha_df
        self.ids = df["id"].astype(str).tolist()
        self.patch_dir_s2 = patch_dir_s2
        self.patch_dir_scl = patch_dir_scl
        self.target_col = target_col
        self.scale_to_reflectance = scale_to_reflectance
        self.reflectance_denominator = reflectance_denominator
        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        self.cache = cache

        self.snow_code = snow_code
        self.cloud_codes = tuple(cloud_codes)

        self._cache_s2  = {}
        self._cache_ae  = {}
        self._cache_scl = {}

        self._nlcd_lut = {
            11: 0, 12: 1, 21: 2, 22: 3, 23: 4, 24: 5,
            31: 6, 41: 7, 42: 8, 43: 9, 52: 10,
            71: 11, 81: 12, 82: 13, 90: 14, 95: 15,
        }
        self.n_nlcd = len(self._nlcd_lut)

        # NLCD class → Michigan forest type
        self.NLCD_TO_FOREST = {
            41: 'temperate_deciduous',
            42: 'boreal_conifer',
            43: 'mixed',
        }

        # Load NLCD raster once; keep open for fast per-sample lookup
        self.nlcd_src  = None
        self.nlcd_data = None
        if nlcd_path is not None and RASTERIO_AVAILABLE:
            self.nlcd_src  = rasterio.open(nlcd_path)
            self.nlcd_data = self.nlcd_src.read(1)   # single band, full array
            print(f"[dataset] NLCD raster loaded from: {nlcd_path}")
        elif nlcd_path is not None and not RASTERIO_AVAILABLE:
            print("[dataset] Warning: nlcd_path supplied but rasterio not available. "
                  "Install rasterio to enable forest-type lookup.")

        # Pre-cache forest_type per row if NLCD is available and df has lat/lon
        self._cache_forest_type = {}
        if self.nlcd_src is not None and "lat" in df.columns and "lon" in df.columns:
            print("[dataset] Pre-caching forest types from NLCD …")
            for idx in range(len(df)):
                row = df.iloc[idx]
                self._cache_forest_type[idx] = self._lookup_forest_type(
                    row["lat"], row["lon"]
                )
            print("[dataset] Forest type caching done.")

        # Pre-cache acquisition date per row from GEDI filename
        # Format: GEDI04_A_YYYYDDDHHMMSS_... → extract YYYY + DOY
        self._acq_dates: list = []
        if "file" in df.columns:
            print("[dataset] Parsing acquisition dates from 'file' column ...")
            for idx in range(len(df)):
                self._acq_dates.append(
                    self._parse_gedi_date(str(df.iloc[idx]["file"]))
                )
            n_ok = sum(1 for d in self._acq_dates if d)
            print(f"[dataset] Dates parsed: {n_ok}/{len(df)} valid")
        else:
            self._acq_dates = [""] * len(df)
            print("[dataset] Warning: 'file' column not found — "
                  "BAAD fallback will be used for all samples.")

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_gedi_date(filepath: str) -> str:
        """
        Extract acquisition date from a GEDI L4A filename.

        Expected pattern: GEDI04_A_YYYYDDDHHMMSS_...
        e.g. GEDI04_A_2024322043153_O33597_03_T03738_02_004_01_V002.h5
             → year=2024, DOY=322 → "2024-11-17"

        Returns "YYYY-MM-DD" string, or "" if pattern not found.
        """
        basename = os.path.basename(filepath)
        m = re.search(r'GEDI\d+_[A-Z]_(\d{4})(\d{3})\d+', basename)
        if not m:
            return ""
        try:
            year = int(m.group(1))
            doy  = int(m.group(2))
            date = (datetime.date(year, 1, 1)
                    + datetime.timedelta(days=doy - 1))
            return date.strftime("%Y-%m-%d")
        except Exception:
            return ""

    # ------------------------------------------------------------------
    def _lookup_forest_type(self, lat: float, lon: float) -> str:
        """Convert GEDI centroid (lat, lon) → forest type string via NLCD."""
        try:
            r, c = rowcol(self.nlcd_src.transform, lon, lat)
            nlcd_class = int(self.nlcd_data[r, c])
            return self.NLCD_TO_FOREST.get(nlcd_class, "mixed")
        except Exception:
            return "mixed"

    def _get_forest_type(self, idx: int, row) -> str:
        """Return forest type for sample idx, using cache when available."""
        if idx in self._cache_forest_type:
            return self._cache_forest_type[idx]
        if self.nlcd_src is not None and "lat" in self.df.columns:
            ft = self._lookup_forest_type(row["lat"], row["lon"])
            self._cache_forest_type[idx] = ft
            return ft
        return "mixed"     # fallback: no NLCD or no lat/lon column

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.df)

    def _load_patch(self, patch_path, scale_to_reflectance=False):
        cube = np.load(patch_path)
        if cube.ndim == 2:
            cube = cube[..., None]
        cube_CHW = np.transpose(cube, (2, 0, 1)).astype("float32")
        if scale_to_reflectance:
            cube_CHW = cube_CHW / self.reflectance_denominator
            cube_CHW = np.clip(cube_CHW, 0.0, 1.0)
        return cube_CHW

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- Sentinel-2 patch ----
        patch_name = f"s2_{row['id']}.npy"
        if self.cache and patch_name in self._cache_s2:
            s2 = self._cache_s2[patch_name]
        else:
            s2_np = self._load_patch(
                os.path.join(self.patch_dir_s2, patch_name),
                scale_to_reflectance=False,
            )
            s2 = torch.from_numpy(s2_np)
            if self.cache:
                self._cache_s2[patch_name] = s2

        # ---- AlphaEarth embedding ----
        ae_key = int(row["id"])
        if self.cache and ae_key in self._cache_ae:
            alpha = self._cache_ae[ae_key]
        else:
            alpha_row = self.alpha_df.loc[self.alpha_df["id"] == row["id"]].iloc[0]
            alpha = torch.tensor(alpha_row.tolist()[1:-2], dtype=torch.float32)
            if self.alpha_mean is not None and self.alpha_std is not None:
                mean = torch.as_tensor(self.alpha_mean, dtype=alpha.dtype)
                std  = torch.as_tensor(self.alpha_std,  dtype=alpha.dtype)
                alpha = (alpha - mean) / (std + 1e-6)
            if self.cache:
                self._cache_ae[ae_key] = alpha

        # ---- SCL cloud / snow masks ----
        snow_frac  = torch.tensor(0.0, dtype=torch.float32)
        cloud_frac = torch.tensor(0.0, dtype=torch.float32)
        scl        = None

        if self.patch_dir_scl is not None:
            scl_name = f"s2_scl_{row['id']}.npy"
            if self.cache and scl_name in self._cache_scl:
                scl = self._cache_scl[scl_name]
            else:
                scl = self._load_patch(
                    os.path.join(self.patch_dir_scl, scl_name),
                    scale_to_reflectance=False,
                )
                if self.cache:
                    self._cache_scl[scl_name] = scl

            snow_frac  = torch.tensor((scl == self.snow_code).mean(),          dtype=torch.float32)
            cloud_frac = torch.tensor(np.isin(scl, self.cloud_codes).mean(),   dtype=torch.float32)

        center_cloud = False
        if scl is not None:
            center_cloud = int(scl[0, 10, 10]) in self.cloud_codes
        center_cloud = torch.tensor(center_cloud, dtype=torch.bool)

        # ---- Assemble inputs ----
        ae_aug  = torch.cat([alpha, snow_frac[None], cloud_frac[None]], dim=0)
        target  = torch.tensor(row[self.target_col], dtype=torch.float32)

        # ---- Forest type (string — used by dual_loss in train loop) ----
        forest_type = self._get_forest_type(idx, row)

        # ---- Acquisition date — from pre-cached GEDI filename parse ----
        acq_date = self._acq_dates[idx]

        return {
            "s2":           s2,
            "ae":           ae_aug,
            "cloud_frac":   cloud_frac,
            "center_cloud": center_cloud,
            "target":       target,
            "patch_name":   row["id"],
            "forest_type":  forest_type,
            "acq_date":     acq_date,      # YYYY-MM-DD string for harmonic lookup
        }
