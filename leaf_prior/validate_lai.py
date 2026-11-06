"""
validate_lai.py
---------------
Process MCD15A2H MODIS LAI data and validate against the harmonic leaf
decomposition from harmonic_decomposition.py.

Inputs:
  - Directory of MCD15A2H HDF4 or GeoTIFF files (AppEEARS output)
  - agbd_timeseries_harmonic.csv  (from harmonic_decomposition.py)
  - AOI bounding box or shapefile (configured below)

Outputs:
  - lai_timeseries.csv            — mean LAI per 8-day composite over AOI
  - lai_vs_leaf_comparison.csv    — LAI and harmonic leaf on common S2 dates
  - plots/lai_validation_*.png    — comparison figures
"""

import os
import glob
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr

# Rasterio / GDAL
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from shapely.geometry import box, mapping

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ---------------------------------------------------------------------------
# CONFIGURATION — edit these paths
# ---------------------------------------------------------------------------

# Directory containing AppEEARS output.
# AppEEARS exports MCD15A2H as GeoTIFFs named like:
#   MCD15A2H.061_Lai_500m_doy2020001_aid0001.tif
# Point this at the folder containing those files.
LAI_DIR = "data/data/modis_lai/MCD15A2H.061_2019359_to_2026120"          # <-- change as needed

# Harmonic timeseries CSV produced by harmonic_decomposition.py
HARMONIC_CSV = "agbd_timeseries_harmonic.csv"

# Output directory
OUT_DIR = "outputs/lai_validation"

# AOI in EPSG:4326 (lon_min, lat_min, lon_max, lat_max)
AOI_BOUNDS = (-85.44179, 42.623034, -85.407286, 42.637305)

# MODIS LAI scale factor and fill value (MCD15A2H product spec)
LAI_SCALE   = 0.1    # stored as uint8, multiply by 0.1 → m²/m²
LAI_FILL    = 255    # fill / no-data value
LAI_MAX_RAW = 100    # valid range 0–100 (scaled: 0–10 m²/m²)

# QC band name pattern (Lai_500m vs FparLai_QC); set False to skip QC masking
USE_QC = True

# ---------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Discover LAI files
# ---------------------------------------------------------------------------

def discover_files(lai_dir: str):
    """
    Return sorted lists of (date, lai_path, qc_path) tuples.
    AppEEARS naming: MCD15A2H.061_Lai_500m_doy{YYYYDDD}_aid*.tif
                     MCD15A2H.061_FparLai_QC_doy{YYYYDDD}_aid*.tif
    """
    lai_pattern = os.path.join(lai_dir, "*Lai_500m_doy*.tif")
    qc_pattern  = os.path.join(lai_dir, "*FparLai_QC_doy*.tif")

    lai_files = sorted(glob.glob(lai_pattern))
    qc_files  = sorted(glob.glob(qc_pattern))

    if not lai_files:
        # Fallback: try HDF files (require gdal HDF4 driver)
        lai_files = sorted(glob.glob(os.path.join(lai_dir, "*.hdf")))
        if not lai_files:
            raise FileNotFoundError(
                f"No MCD15A2H GeoTIFF or HDF files found in {lai_dir}.\n"
                "Export from AppEEARS as GeoTIFF (one file per variable per date)."
            )

    # Build doy→qc lookup
    doy_re = re.compile(r"doy(\d{7})")  # YYYYDDD

    def extract_doy(path):
        m = doy_re.search(os.path.basename(path))
        return m.group(1) if m else None

    qc_lookup = {extract_doy(p): p for p in qc_files if extract_doy(p)}

    records = []
    for lp in lai_files:
        doy_str = extract_doy(lp)
        if doy_str is None:
            continue
        year = int(doy_str[:4])
        doy  = int(doy_str[4:])
        date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        qp   = qc_lookup.get(doy_str)
        records.append({"date": date, "doy_str": doy_str, "lai_path": lp, "qc_path": qp})

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"Found {len(df)} LAI composites ({df['date'].min().date()} → {df['date'].max().date()})")
    return df


# ---------------------------------------------------------------------------
# 2. AOI geometry (EPSG:4326)
# ---------------------------------------------------------------------------

def aoi_geometry():
    lon_min, lat_min, lon_max, lat_max = AOI_BOUNDS
    geom = box(lon_min, lat_min, lon_max, lat_max)
    return [mapping(geom)]


# ---------------------------------------------------------------------------
# 3. Extract mean LAI over AOI for one file
# ---------------------------------------------------------------------------

def extract_mean_lai(lai_path: str, qc_path, aoi_geoms) -> float:
    """
    Reproject AOI to the LAI file's CRS, mask, apply QC, return mean LAI (m²/m²).
    Returns NaN if no valid pixels.
    """
    aoi_crs = CRS.from_epsg(4326)

    with rasterio.open(lai_path) as src:
        file_crs = src.crs

        # Reproject AOI geometry to file CRS if needed
        if file_crs and file_crs != aoi_crs:
            from rasterio.warp import transform_geom
            geoms_proj = [transform_geom(aoi_crs, file_crs, g) for g in aoi_geoms]
        else:
            geoms_proj = aoi_geoms

        try:
            data, _ = rio_mask(src, geoms_proj, crop=True, nodata=LAI_FILL)
        except Exception:
            return np.nan

        data = data[0].astype(float)  # first band

    # Mask fill and out-of-range
    valid = (data != LAI_FILL) & (data <= LAI_MAX_RAW)

    # Optional QC masking
    if USE_QC and qc_path and os.path.exists(qc_path):
        with rasterio.open(qc_path) as qsrc:
            if file_crs and file_crs != aoi_crs:
                qgeoms = geoms_proj
            else:
                qgeoms = aoi_geoms
            try:
                qdata, _ = rio_mask(qsrc, qgeoms, crop=True, nodata=255)
                qdata = qdata[0]
                # Bits 0-1: MODLAND QC (0=good, 1=other, 2=snow/ice, 3=cloud)
                # Keep only bits 0-1 == 0 or 1 (good quality)
                qc_bits = qdata & 0b11
                valid = valid & (qdata != 255) & (qc_bits <= 1)
            except Exception:
                pass  # proceed without QC if mask fails

    if not valid.any():
        return np.nan

    return float(data[valid].mean() * LAI_SCALE)


# ---------------------------------------------------------------------------
# 4. Build LAI timeseries over AOI
# ---------------------------------------------------------------------------

def build_lai_timeseries(file_df: pd.DataFrame) -> pd.DataFrame:
    aoi_geoms = aoi_geometry()
    records = []
    for _, row in file_df.iterrows():
        lai_val = extract_mean_lai(row["lai_path"], row.get("qc_path"), aoi_geoms)
        records.append({"date": row["date"], "lai_m2m2": lai_val})
        status = f"{lai_val:.3f}" if not np.isnan(lai_val) else "NaN"
        print(f"  {row['date'].date()}  LAI={status}")

    df = pd.DataFrame(records)
    out_path = os.path.join(OUT_DIR, "lai_timeseries.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved LAI timeseries → {out_path}")
    return df


# ---------------------------------------------------------------------------
# 5. Interpolate LAI to S2 acquisition dates
# ---------------------------------------------------------------------------

def interpolate_lai_to_s2(lai_df: pd.DataFrame, s2_dates: pd.Series) -> np.ndarray:
    """
    Linear interpolation of 8-day LAI composites to arbitrary S2 dates.
    Extrapolation is clamped to 0 outside the LAI date range.
    """
    lai_clean = lai_df.dropna(subset=["lai_m2m2"]).copy()
    lai_clean["t"] = (lai_clean["date"] - lai_clean["date"].min()).dt.days.astype(float)

    t0 = lai_clean["date"].min()
    s2_t = np.array([(d - t0).days for d in s2_dates], dtype=float)

    interp_fn = interp1d(
        lai_clean["t"].values,
        lai_clean["lai_m2m2"].values,
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, 0.0),
    )
    return interp_fn(s2_t)


# ---------------------------------------------------------------------------
# 6. Load harmonic CSV and merge
# ---------------------------------------------------------------------------

def load_harmonic(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    # Expected columns: date, agbd_mean, leaf_mean, wood_mean, agbd_fit,
    #                   leaf_harmonic, wood_harmonic  (names may vary slightly)
    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Correlation & phase analysis
# ---------------------------------------------------------------------------

def doy_of_peak(series: pd.Series, dates: pd.Series) -> int:
    """Return DOY of the maximum value in series."""
    idx = series.idxmax()
    return dates.iloc[idx].day_of_year


def harmonic_peak_doy(a, b, phi) -> int:
    """
    For f(t) = a + b*sin(2π*doy/365 + φ), peak at doy where derivative=0.
    doy_peak = (π/2 - φ) * 365 / (2π)  mod 365
    """
    doy = (np.pi / 2 - phi) * 365 / (2 * np.pi)
    return int(doy % 365)


def compute_metrics(x, y, label_x="LAI", label_y="leaf_harmonic"):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {}
    r, p_r = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)
    # Normalize both to [0,1] for shape comparison
    xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-9)
    rmse_norm = np.sqrt(np.mean((xn - yn) ** 2))
    return {
        "n": int(len(x)),
        "pearson_r": round(r, 4),
        "pearson_p": round(p_r, 4),
        "spearman_rho": round(rho, 4),
        "spearman_p": round(p_rho, 4),
        "normalized_rmse": round(rmse_norm, 4),
    }


# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------

def plot_lai_timeseries(lai_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(lai_df["date"], lai_df["lai_m2m2"], "o-", color="#2d6a4f", ms=3, lw=1.2,
            label="MODIS LAI (MCD15A2H)")
    ax.set_ylabel("LAI (m² / m²)")
    ax.set_title("MODIS MCD15A2H LAI — Michigan AOI (2020–2026)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = f"{OUT_DIR}/plots/lai_timeseries.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


def plot_comparison(merged: pd.DataFrame, metrics: dict):
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    # --- Panel 1: timeseries overlay (normalized) ---
    ax = axes[0]
    lai_n  = (merged["lai_interp"] - merged["lai_interp"].min()) / \
             (merged["lai_interp"].max() - merged["lai_interp"].min() + 1e-9)

    # Try both possible column names for harmonic leaf
    leaf_col = "leaf_harmonic" if "leaf_harmonic" in merged.columns else "leaf_mean"
    leaf_n = (merged[leaf_col] - merged[leaf_col].min()) / \
             (merged[leaf_col].max() - merged[leaf_col].min() + 1e-9)

    ax.plot(merged["date"], lai_n,  color="#2d6a4f", lw=1.5, label="MODIS LAI (norm.)")
    ax.plot(merged["date"], leaf_n, color="#e07b39", lw=1.5, linestyle="--",
            label="Harmonic leaf component (norm.)")
    ax.set_ylabel("Normalized value [0–1]")
    ax.set_title(
        f"LAI vs Harmonic Leaf Phenology  "
        f"(r={metrics.get('pearson_r','?')}, ρ={metrics.get('spearman_rho','?')})"
    )
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Panel 2: scatter LAI vs leaf ---
    ax = axes[1]
    ax.scatter(merged["lai_interp"], merged[leaf_col], alpha=0.5, s=15, color="#5e3a8c")
    # Regression line
    mask = np.isfinite(merged["lai_interp"]) & np.isfinite(merged[leaf_col])
    if mask.sum() > 2:
        coeffs = np.polyfit(merged.loc[mask, "lai_interp"], merged.loc[mask, leaf_col], 1)
        x_line = np.linspace(merged["lai_interp"].min(), merged["lai_interp"].max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "r-", lw=1.5,
                label=f"y={coeffs[0]:.2f}x+{coeffs[1]:.2f}")
    ax.set_xlabel("MODIS LAI (m²/m²)")
    ax.set_ylabel("Harmonic leaf biomass (Mg/ha)")
    ax.set_title("LAI vs Leaf Biomass — Scatter")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: DOY seasonal climatology ---
    ax = axes[2]
    merged["doy"] = merged["date"].dt.day_of_year
    # Group by DOY (bin to ~8-day windows)
    merged["doy_bin"] = (merged["doy"] // 8) * 8 + 4
    clim = merged.groupby("doy_bin").agg(
        lai_mean=("lai_interp", "mean"),
        leaf_mean=(leaf_col, "mean"),
    ).reset_index()

    ax2 = ax.twinx()
    ax.fill_between(clim["doy_bin"], clim["lai_mean"], alpha=0.3, color="#2d6a4f", label="LAI")
    ax.plot(clim["doy_bin"], clim["lai_mean"], color="#2d6a4f", lw=1.5)
    ax2.plot(clim["doy_bin"], clim["leaf_mean"], color="#e07b39", lw=1.5,
             linestyle="--", label="Leaf biomass")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Mean MODIS LAI (m²/m²)", color="#2d6a4f")
    ax2.set_ylabel("Mean leaf biomass (Mg/ha)", color="#e07b39")
    ax.set_title("Seasonal Climatology (DOY)")
    ax.set_xlim(1, 365)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{OUT_DIR}/plots/lai_validation_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


def plot_per_year(merged: pd.DataFrame):
    """One subplot per year showing LAI and leaf biomass together."""
    leaf_col = "leaf_harmonic" if "leaf_harmonic" in merged.columns else "leaf_mean"
    merged["year"] = merged["date"].dt.year
    years = sorted(merged["year"].unique())

    fig, axes = plt.subplots(len(years), 1, figsize=(12, 3 * len(years)), sharex=False)
    if len(years) == 1:
        axes = [axes]

    for ax, yr in zip(axes, years):
        sub = merged[merged["year"] == yr]
        ax2 = ax.twinx()
        ax.plot(sub["date"].dt.day_of_year, sub["lai_interp"],
                "o-", color="#2d6a4f", ms=3, lw=1.2, label="LAI")
        ax2.plot(sub["date"].dt.day_of_year, sub[leaf_col],
                 "s--", color="#e07b39", ms=3, lw=1.2, label="Leaf biomass")
        ax.set_title(str(yr))
        ax.set_ylabel("LAI", color="#2d6a4f", fontsize=8)
        ax2.set_ylabel("Leaf (Mg/ha)", color="#e07b39", fontsize=8)
        ax.set_xlim(1, 365)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Year LAI vs Leaf Biomass", y=1.01)
    plt.tight_layout()
    out = f"{OUT_DIR}/plots/lai_per_year.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODIS LAI Validation vs Harmonic Leaf Decomposition")
    print("=" * 60)

    # --- Step 1: Discover and process LAI files ---
    file_df = discover_files(LAI_DIR)
    print("\nExtracting LAI over AOI...")
    lai_df = build_lai_timeseries(file_df)

    # --- Step 2: Load harmonic timeseries ---
    if not os.path.exists(HARMONIC_CSV):
        raise FileNotFoundError(f"Harmonic CSV not found: {HARMONIC_CSV}")
    harm_df = load_harmonic(HARMONIC_CSV)
    print(f"\nLoaded harmonic CSV: {len(harm_df)} S2 dates")
    print(f"  Columns: {list(harm_df.columns)}")

    # --- Step 3: Interpolate LAI to S2 dates ---
    harm_df["lai_interp"] = interpolate_lai_to_s2(lai_df, harm_df["date"])

    # --- Step 4: Metrics ---
    leaf_col = "leaf_harmonic" if "leaf_harmonic" in harm_df.columns else "leaf_mean"
    metrics = compute_metrics(
        harm_df["lai_interp"].values,
        harm_df[leaf_col].values,
    )
    print("\n--- Correlation metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Phase check
    try:
        params_df = pd.read_csv("harmonic_params.csv")
        phi = float(params_df["phi"].iloc[0])
        b   = float(params_df["b"].iloc[0])
        a   = float(params_df["a"].iloc[0])
        peak_doy_harmonic = harmonic_peak_doy(a, b, phi)
        print(f"\n  Harmonic leaf peak DOY : {peak_doy_harmonic}")
    except Exception:
        peak_doy_harmonic = None
        print("\n  (harmonic_params.csv not found — skipping phase check)")

    lai_nonan = lai_df.dropna(subset=["lai_m2m2"])
    if len(lai_nonan) > 0:
        # Multi-year peak: average DOY of annual maxima
        lai_nonan = lai_nonan.copy()
        lai_nonan["year"] = lai_nonan["date"].dt.year
        lai_nonan["doy"]  = lai_nonan["date"].dt.day_of_year
        annual_peaks = lai_nonan.groupby("year").apply(
            lambda g: g.loc[g["lai_m2m2"].idxmax(), "doy"]
        )
        mean_lai_peak_doy = int(annual_peaks.mean())
        print(f"  MODIS LAI peak DOY    : {mean_lai_peak_doy} "
              f"(mean of annual maxima: {annual_peaks.to_dict()})")
        if peak_doy_harmonic is not None:
            print(f"  Phase offset          : {peak_doy_harmonic - mean_lai_peak_doy:+d} days")

    # --- Step 5: Save merged CSV ---
    out_csv = os.path.join(OUT_DIR, "lai_vs_leaf_comparison.csv")
    harm_df.to_csv(out_csv, index=False)
    print(f"\nSaved comparison CSV → {out_csv}")

    # --- Step 6: Plots ---
    print("\nGenerating plots...")
    plot_lai_timeseries(lai_df)
    plot_comparison(harm_df, metrics)
    plot_per_year(harm_df)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  LAI composites processed : {len(file_df)}")
    print(f"  Valid LAI observations   : {lai_df['lai_m2m2'].notna().sum()}")
    print(f"  S2 dates in harmonic CSV : {len(harm_df)}")
    print(f"  Pearson r (LAI ~ leaf)   : {metrics.get('pearson_r', 'N/A')}")
    print(f"  Spearman ρ (LAI ~ leaf)  : {metrics.get('spearman_rho', 'N/A')}")
    print(f"  Normalized RMSE          : {metrics.get('normalized_rmse', 'N/A')}")
    print()
    print("Next steps:")
    print("  1. If r > 0.85 → harmonic leaf phase is physically valid")
    print("  2. If peak DOY offset > 15 days → refit harmonic with φ initialized")
    print("     closer to the MODIS LAI peak")
    print("  3. Use LAI as a direct supervised signal for a 3-phase S2 model")
    print("     (winter + summer + target) → wood/leaf end-to-end prediction")


if __name__ == "__main__":
    main()
