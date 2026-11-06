"""
prepare_leaf_prior.py
=====================
Computes a spatially explicit leaf fraction map for Michigan by combining:

  1. NLCD 2021        — forest mask (classes 41, 42, 43) + fallback grouping
  2. TreeMap 2016     — species composition per plot (SPCD, SCIENTIFIC_NAME,
                        DIA, TPA_UNADJ), aligned to NLCD grid
  3. BAAD             — PRIMARY leaf fraction source: direct measurements
                        per species (m.lf / (m.lf + m.st + m.br))
  4. Jenkins (2003)   — FALLBACK leaf fraction source: allometric equations
                        for species not in BAAD (hardwood / softwood x DBH)

Leaf fraction priority per tree:
  1. BAAD match on SCIENTIFIC_NAME   -> direct measurement  (~32% TPA)
  2. Jenkins Table 6 (hw/sw + DBH)   -> allometric equation (~68% TPA)

Coverage:
  - 91% of forest pixels: species-weighted leaf fraction via TreeMap
  - 9%  of forest pixels: spatial interpolation from nearby covered pixels

Outputs:
  data/processed/leaf_fraction_michigan.tif   -- per-pixel leaf fraction map
  data/processed/source_map_michigan.tif      -- source per pixel
                                                 0=non-forest 1=baad
                                                 2=jenkins 3=spatial_fallback
  data/processed/leaf_prior_michigan.csv      -- LEAF_PRIOR per NLCD class
  (printed) ready-to-paste LEAF_PRIOR dict for train_2_FCN.py

Usage:
  pip install rasterio numpy pandas scipy
  python prepare_leaf_prior.py
"""

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from scipy.ndimage import uniform_filter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NLCD_PATH     = "nlcd_4326_mi.tif"
TREEMAP_PATH  = "treemap_aligned.tif"
TREEMAP_CSV   = "TreeMap2022_CONUS_Tree_Table.csv"
BAAD_CSV      = "baad_data.csv"
OUT_MAP       = "leaf_fraction_michigan_1.tif"
OUT_SRC_MAP   = "source_map_michigan_1.tif"
OUT_PRIOR_CSV = "leaf_prior_michigan_1.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TM_NODATA      = 4294967296
FOREST_CODES   = [41, 42, 43]
NLCD_TO_FOREST = {
    41: "temperate_deciduous",
    42: "boreal_conifer",
    43: "mixed",
}

# Source codes written into source_map output raster
SRC_NODATA   = 0
SRC_BAAD     = 1
SRC_JENKINS  = 2
SRC_FALLBACK = 3

# ---------------------------------------------------------------------------
# Jenkins et al. (2003) Table 6
# Foliage fraction = exp(d0 + d1 / dbh_cm)
# Two classes only: hardwood and softwood
# ---------------------------------------------------------------------------
JENKINS_FOLIAGE = {
    "hardwood": {"d0": -4.0813, "d1": 5.8816},
    "softwood": {"d0": -2.9584, "d1": 4.4766},
}

# FIA SPCD values classified as softwood
# Everything else is treated as hardwood
SOFTWOOD_SPCDS = {
    # True firs
    12, 15, 17, 19, 20, 21,
    # Spruces
    91, 93, 94, 96, 98,
    # Pines
    95, 97, 100, 101, 102, 103, 105, 106, 108, 109,
    110, 111, 113, 114, 115, 116, 120, 121, 122, 123,
    124, 125, 126, 127, 128, 129,
    # Cedars / cypress / juniper
    42, 43, 44, 46, 47, 50, 51, 54, 58, 60, 61,
    62, 63, 64, 65, 66, 67, 68,
    # Hemlock
    260, 261, 262, 263, 264, 265,
    # Larch / tamarack
    70, 71, 72, 73, 74, 75,
    # Douglas fir
    202, 203,
    # Redwood / sequoia
    211, 212,
}


def jenkins_leaf_fraction(spcd: int, dia_inches: float):
    """
    Compute foliage fraction using Jenkins et al. (2003) Table 6.

    Args:
        spcd:       FIA species code
        dia_inches: DBH in inches (TreeMap DIA column)

    Returns:
        float leaf fraction, or None if DBH below minimum (2.5 cm)
    """
    dia_cm = float(dia_inches) * 2.54
    if dia_cm < 2.5:
        return None
    group = "softwood" if int(spcd) in SOFTWOOD_SPCDS else "hardwood"
    d0    = JENKINS_FOLIAGE[group]["d0"]
    d1    = JENKINS_FOLIAGE[group]["d1"]
    return float(np.exp(d0 + d1 / dia_cm))


# ===========================================================================
# Step 1 — Load rasters
# ===========================================================================
print("=" * 65)
print("Step 1 — Loading rasters")
print("=" * 65)

with rasterio.open(NLCD_PATH) as src:
    nlcd           = src.read(1)
    nlcd_meta      = src.meta.copy()
    nlcd_transform = src.transform
    nlcd_crs       = src.crs

with rasterio.open(TREEMAP_PATH) as src:
    tmid_raw = src.read(1).astype(np.float64)

# Mask out TreeMap nodata (4294967295 = max uint32)
tmid_aligned = tmid_raw.copy()
tmid_aligned[tmid_raw == TM_NODATA] = 0
tmid_aligned[tmid_raw < 0]          = 0
tmid_aligned = tmid_aligned.astype(np.int64)

forest_mask = np.isin(nlcd, FOREST_CODES)
tm_valid    = tmid_aligned > 0

print(f"  Forest pixels (NLCD):          {forest_mask.sum():,}")
print(f"  Forest + TreeMap covered:      {(forest_mask & tm_valid).sum():,}")
print(f"  Forest but NO TreeMap:         {(forest_mask & ~tm_valid).sum():,}")
print(f"  TreeMap coverage of forest:    "
      f"{100*(forest_mask & tm_valid).sum() / forest_mask.sum():.1f}%")


# ===========================================================================
# Step 2 — Load and process BAAD (primary source)
# ===========================================================================
print("\n" + "=" * 65)
print("Step 2 — Processing BAAD (primary source)")
print("=" * 65)

baad = pd.read_csv(BAAD_CSV, low_memory=False)
baad = baad.dropna(subset=["m.lf", "m.st", "m.br", "speciesMatched"])
baad = baad[(baad["m.lf"] > 0) & (baad["m.st"] > 0)].copy()

baad["m.wood"]        = baad["m.st"] + baad["m.br"]
baad["m.total"]       = baad["m.lf"] + baad["m.wood"]
baad["leaf_fraction"] = baad["m.lf"] / baad["m.total"]
baad = baad[(baad["leaf_fraction"] > 0) & (baad["leaf_fraction"] < 1)].copy()
baad["speciesMatched"] = baad["speciesMatched"].str.strip().str.title()

# Species-level mean leaf fraction
baad_species = (
    baad.groupby("speciesMatched")["leaf_fraction"]
    .agg(baad_mean="mean", baad_std="std", baad_n="count")
    .reset_index()
)

# Fast lookup: species name -> mean leaf fraction
baad_lookup = dict(zip(baad_species["speciesMatched"],
                       baad_species["baad_mean"]))

print(f"  BAAD species with leaf fraction: {len(baad_species):,}")
print(f"  Leaf fraction range:             "
      f"{baad_species['baad_mean'].min():.3f} - "
      f"{baad_species['baad_mean'].max():.3f}")
print(f"  Leaf fraction mean:              "
      f"{baad_species['baad_mean'].mean():.3f}")


# ===========================================================================
# Step 3 — Load TreeMap CSV and assign leaf fraction per tree
#          Priority: 1) BAAD direct measurement  2) Jenkins allometric
# ===========================================================================
print("\n" + "=" * 65)
print("Step 3 — Processing TreeMap CSV")
print("         Priority: BAAD (direct) -> Jenkins (allometric fallback)")
print("=" * 65)

tree = pd.read_csv(TREEMAP_CSV, low_memory=False)
tree = tree[tree["STATUSCD"] == 1].copy()
tree = tree[tree["TPA_UNADJ"] > 0].copy()
tree["speciesMatched"] = tree["SCIENTIFIC_NAME"].str.strip().str.title()

print(f"  Live trees in CSV:             {len(tree):,}")
print(f"  Unique species:                {tree['speciesMatched'].nunique():,}")

# Assign leaf fraction with BAAD -> Jenkins priority
leaf_fractions = []
sources        = []

for _, row in tree.iterrows():
    species = row["speciesMatched"]
    spcd    = row["SPCD"]
    dia     = row["DIA"]

    # 1. BAAD (primary)
    baad_val = baad_lookup.get(species)
    if baad_val is not None:
        leaf_fractions.append(baad_val)
        sources.append("baad")
        continue

    # 2. Jenkins (fallback)
    if pd.notna(dia):
        jval = jenkins_leaf_fraction(spcd, dia)
        if jval is not None:
            leaf_fractions.append(jval)
            sources.append("jenkins")
            continue

    # Should not reach here given 100% DIA coverage
    leaf_fractions.append(np.nan)
    sources.append("none")

tree["leaf_fraction"] = leaf_fractions
tree["source"]        = sources

# Coverage report by TPA weight
source_tpa = tree.groupby("source")["TPA_UNADJ"].sum()
total_tpa  = tree["TPA_UNADJ"].sum()
print(f"\n  Leaf fraction source breakdown (TPA-weighted):")
for src, tpa in source_tpa.items():
    print(f"    {src:<10}: {tpa:>15,.0f} TPA  ({100*tpa/total_tpa:.1f}%)")

# Validation: BAAD vs Jenkins for species that have both
print(f"\n  BAAD vs Jenkins validation (matched species only):")
tree_val = tree[tree["source"] == "baad"].copy()
tree_val["jenkins_val"] = tree_val.apply(
    lambda r: jenkins_leaf_fraction(r["SPCD"], r["DIA"])
    if pd.notna(r["DIA"]) else None,
    axis=1,
)
tree_val = tree_val.dropna(subset=["jenkins_val"])
if len(tree_val) > 0:
    diff = tree_val["leaf_fraction"] - tree_val["jenkins_val"]
    print(f"    Mean difference (BAAD - Jenkins): {diff.mean():+.4f}")
    print(f"    Std of difference:                {diff.std():.4f}")
    print(f"    Max absolute difference:          {diff.abs().max():.4f}")
    # Per-species breakdown for top BAAD species
    per_species = (
        tree_val.groupby("speciesMatched")
        .apply(lambda g: pd.Series({
            "baad_mean":    g["leaf_fraction"].mean(),
            "jenkins_mean": g["jenkins_val"].mean(),
            "diff":         (g["leaf_fraction"] - g["jenkins_val"]).mean(),
            "tpa":          g["TPA_UNADJ"].sum(),
        }))
        .sort_values("tpa", ascending=False)
        .head(10)
    )
    print(f"\n    Top 10 species by TPA (BAAD vs Jenkins):")
    print(per_species.to_string())
else:
    print("    No overlap found for validation.")


# ===========================================================================
# Step 4 — Compute per TM_ID weighted leaf fraction
#          Weight by TPA_UNADJ across all species in each plot
# ===========================================================================
print("\n" + "=" * 65)
print("Step 4 — Computing per-plot weighted leaf fractions")
print("=" * 65)

tree_valid = tree.dropna(subset=["leaf_fraction"]).copy()

def plot_summary(g):
    lf  = np.average(g["leaf_fraction"], weights=g["TPA_UNADJ"])
    # dominant source = source of the highest-TPA tree in plot
    dom = g.loc[g["TPA_UNADJ"].idxmax(), "source"]
    return pd.Series({"leaf_fraction": lf, "dominant_source": dom})

tm_leaf = (
    tree_valid
    .groupby("TM_ID")
    .apply(plot_summary)
    .reset_index()
)

SOURCE_TO_CODE = {"baad": SRC_BAAD, "jenkins": SRC_JENKINS, "none": SRC_NODATA}
tm_leaf["source_code"] = tm_leaf["dominant_source"].map(SOURCE_TO_CODE)

print(f"  Plots with leaf fraction:      {len(tm_leaf):,}")
print(f"  Leaf fraction range:           "
      f"{tm_leaf['leaf_fraction'].min():.4f} - "
      f"{tm_leaf['leaf_fraction'].max():.4f}")
print(f"  Leaf fraction mean:            {tm_leaf['leaf_fraction'].mean():.4f}")
print(f"  Leaf fraction std:             {tm_leaf['leaf_fraction'].std():.4f}")

# Fast lookup dicts
tmid_to_leaf   = dict(zip(tm_leaf["TM_ID"].astype(np.int64),
                          tm_leaf["leaf_fraction"]))
tmid_to_source = dict(zip(tm_leaf["TM_ID"].astype(np.int64),
                          tm_leaf["source_code"]))


# ===========================================================================
# Step 5 — Build pixel-wise leaf fraction map
# ===========================================================================
print("\n" + "=" * 65)
print("Step 5 — Building pixel-wise leaf fraction map")
print("=" * 65)

temp_map   = np.full(nlcd.shape, np.nan, dtype=np.float64)
source_map = np.zeros(nlcd.shape,        dtype=np.uint8)

# --- TreeMap-covered forest pixels ---
covered_forest = forest_mask & tm_valid
unique_ids     = np.unique(tmid_aligned[covered_forest])
max_id         = int(unique_ids.max()) + 1

id_to_leaf   = np.full(max_id, np.nan, dtype=np.float64)
id_to_source = np.zeros(max_id,         dtype=np.uint8)

n_matched  = 0
n_no_match = 0
for tid in unique_ids:
    if tid in tmid_to_leaf:
        id_to_leaf[tid]   = tmid_to_leaf[tid]
        id_to_source[tid] = tmid_to_source.get(tid, SRC_JENKINS)
        n_matched += 1
    else:
        n_no_match += 1

print(f"  Unique TM_IDs in forest:       {len(unique_ids):,}")
print(f"  TM_IDs with leaf fraction:     {n_matched:,}")
print(f"  TM_IDs without match:          {n_no_match:,}")

# Vectorised pixel assignment
tm_ids_flat = tmid_aligned[covered_forest]
valid_range = tm_ids_flat < max_id

leaf_vals   = np.where(
    valid_range,
    id_to_leaf[np.clip(tm_ids_flat, 0, max_id - 1)],
    np.nan,
)
source_vals = np.where(
    valid_range,
    id_to_source[np.clip(tm_ids_flat, 0, max_id - 1)],
    SRC_NODATA,
)

temp_map[covered_forest]   = leaf_vals
source_map[covered_forest] = source_vals

# --- Spatial interpolation fallback for uncovered forest pixels (9%) ---
# Use local neighbourhood mean from covered pixels of the same NLCD class
# so the fallback reflects nearby species composition, not a global mean
print(f"\n  Spatial interpolation fallback for uncovered pixels...")

WINDOW_SIZE = 51   # ~1.5 km at 30m resolution

for nlcd_code, forest_type in NLCD_TO_FOREST.items():
    needs_fallback = forest_mask & (nlcd == nlcd_code) & np.isnan(temp_map)

    if needs_fallback.sum() == 0:
        print(f"  NLCD {nlcd_code} ({forest_type}): no fallback needed")
        continue

    # Calibration source: covered pixels of same NLCD class
    cal_mask = (nlcd == nlcd_code) & covered_forest & ~np.isnan(temp_map)

    covered_vals  = np.where(cal_mask, temp_map, 0.0)
    covered_count = cal_mask.astype(np.float64)

    smoothed_sum   = uniform_filter(covered_vals,  size=WINDOW_SIZE)
    smoothed_count = uniform_filter(covered_count, size=WINDOW_SIZE)

    global_mean = float(np.mean(temp_map[cal_mask])) if cal_mask.sum() > 0 else 0.06

    local_mean = np.where(
        smoothed_count > 1e-6,
        smoothed_sum / (smoothed_count + 1e-9),
        global_mean,
    )

    temp_map[needs_fallback]   = local_mean[needs_fallback]
    source_map[needs_fallback] = SRC_FALLBACK

    print(f"  NLCD {nlcd_code} ({forest_type:<24}): "
          f"filled {needs_fallback.sum():,} pixels  "
          f"global_mean={global_mean:.4f}")

# Zero out non-forest pixels
temp_map[~forest_mask]   = 0.0
source_map[~forest_mask] = SRC_NODATA
leaf_map = temp_map.astype(np.float32)

# Summary stats
forest_vals = leaf_map[forest_mask]
print(f"\n  Leaf fraction map stats (forest pixels):")
print(f"    Min:    {forest_vals.min():.4f}")
print(f"    Max:    {forest_vals.max():.4f}")
print(f"    Mean:   {forest_vals.mean():.4f}")
print(f"    Median: {np.median(forest_vals):.4f}")
print(f"    Std:    {forest_vals.std():.4f}")

print(f"\n  Pixel source breakdown (forest pixels only):")
for code, name in [(SRC_BAAD,     "baad"),
                   (SRC_JENKINS,  "jenkins"),
                   (SRC_FALLBACK, "spatial_fallback")]:
    n = ((source_map == code) & forest_mask).sum()
    print(f"    {name:<20}: {n:>12,}  ({100*n/forest_mask.sum():.1f}%)")


# ===========================================================================
# Step 6 — Save outputs
# ===========================================================================
print("\n" + "=" * 65)
print("Step 6 — Saving outputs")
print("=" * 65)

# Leaf fraction map
out_meta = nlcd_meta.copy()
out_meta.update({"dtype": "float32", "count": 1, "nodata": -1.0})
with rasterio.open(OUT_MAP, "w", **out_meta) as dst:
    dst.write(leaf_map, 1)
print(f"  Saved: {OUT_MAP}")

# Source map
src_meta = nlcd_meta.copy()
src_meta.update({"dtype": "uint8", "count": 1, "nodata": 0})
with rasterio.open(OUT_SRC_MAP, "w", **src_meta) as dst:
    dst.write(source_map, 1)
print(f"  Saved: {OUT_SRC_MAP}")
print(f"         (0=non-forest  1=baad  2=jenkins  3=spatial_fallback)")


# ===========================================================================
# Step 7 — Compute LEAF_PRIOR per NLCD class
#          Summary of leaf fraction map grouped by forest type
#          These values go into train_2_FCN.py
# ===========================================================================
print("\n" + "=" * 65)
print("Step 7 — Computing LEAF_PRIOR per NLCD forest class")
print("=" * 65)

prior_rows = []
for nlcd_code, forest_type in NLCD_TO_FOREST.items():
    mask = forest_mask & (nlcd == nlcd_code) & (leaf_map > 0)
    vals = leaf_map[mask]

    if len(vals) == 0:
        print(f"  WARNING: No pixels for {forest_type}, skipping.")
        continue

    mean   = round(float(np.mean(vals)),   4)
    std    = round(float(np.std(vals)),    4)
    std    = max(std, 0.005)
    median = round(float(np.median(vals)), 4)

    baad_n     = ((source_map == SRC_BAAD)     & mask).sum()
    jenkins_n  = ((source_map == SRC_JENKINS)  & mask).sum()
    fallback_n = ((source_map == SRC_FALLBACK) & mask).sum()
    total_n    = len(vals)

    prior_rows.append({
        "forest_type":  forest_type,
        "nlcd_code":    nlcd_code,
        "mean":         mean,
        "std":          std,
        "median":       median,
        "n_pixels":     total_n,
        "pct_baad":     round(100 * baad_n     / total_n, 1),
        "pct_jenkins":  round(100 * jenkins_n  / total_n, 1),
        "pct_fallback": round(100 * fallback_n / total_n, 1),
    })

    print(f"  {forest_type:<26} "
          f"mean={mean:.4f}  std={std:.4f}  median={median:.4f}  "
          f"n={total_n:,}  "
          f"(baad={baad_n/total_n*100:.0f}%  "
          f"jenkins={jenkins_n/total_n*100:.0f}%  "
          f"fallback={fallback_n/total_n*100:.0f}%)")

prior_df = pd.DataFrame(prior_rows)
prior_df.to_csv(OUT_PRIOR_CSV, index=False)
print(f"\n  Saved: {OUT_PRIOR_CSV}")


# ===========================================================================
# Step 8 — Print ready-to-paste LEAF_PRIOR for train_2_FCN.py
# ===========================================================================
print("\n" + "=" * 65)
print("Step 8 — Paste this into train_2_FCN.py")
print("=" * 65)
print("\nLEAF_PRIOR = {")
for _, row in prior_df.iterrows():
    print(f'    "{row.forest_type}": '
          f'{{"mean": {row["mean"]}, "std": {row["std"]}}},   '
          f'# n={row["n_pixels"]:,}  '
          f'baad={row["pct_baad"]}%  '
          f'jenkins={row["pct_jenkins"]}%  '
          f'fallback={row["pct_fallback"]}%')
print("}")
