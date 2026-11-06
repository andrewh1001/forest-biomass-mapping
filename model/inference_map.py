import os
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import shape
from pyproj import Transformer
from network import AGBDNet, S2EncoderFCN_ResFiLM
import tifffile as tif

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
S2_DIR        = "stacks/s2_6933"
SCL_DIR       = None
ALPHA_TIF     = "stacks/SCL/s2_6933"
CKPT_PATH     = "../weights/singlehead.pt"
OUT_CSV       = "agbd_timeseries.csv"

# Leaf fraction map from prepare_leaf_prior.py
LEAF_FRAC_TIF = "data/processed/leaf_fraction_michigan.tif"

# Output directories
OUT_DIR_AGBD  = "outputs/agbd"       # total AGBD per date
OUT_DIR_LEAF  = "outputs/leaf"       # leaf biomass per date
OUT_DIR_WOOD  = "outputs/wood"       # wood biomass per date

NLCD_TIF       = None
FOREST_CLASSES = {41, 42, 43}

CLOUD_CODES       = {3, 8, 9, 10}
SNOW_CODE         = 11
CLOUD_FRAC_THRESH = 0.496

AOI_GEOJSON = {
    "type": "Polygon",
    "coordinates": [[
        [-85.44179,  42.623034],
        [-85.44179,  42.637305],
        [-85.407286, 42.637305],
        [-85.407286, 42.623034],
        [-85.44179,  42.623034],
    ]]
}

TILE         = 256
OVERLAP      = 64
PATCH        = 21
BATCH_PIXELS = 8192
DEVICE       = "cuda"
S2_SCALE     = 1/10000.0
EPS          = 3e-5
NODATA       = -9999.0

# Create output dirs
for d in [OUT_DIR_AGBD, OUT_DIR_LEAF, OUT_DIR_WOOD]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# HELPERS — unchanged from original
# ─────────────────────────────────────────────

def load_model(ckpt_path, device):
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    stats      = np.load(ckpt_path.replace(".pt", ".npz"))
    alpha_mean = stats["alpha_mean"].astype(np.float32).reshape(-1)
    alpha_std  = stats["alpha_std"].astype(np.float32).reshape(-1)

    encoder = S2EncoderFCN_ResFiLM(in_channels=10, embed_dim=128, ae_dim=66)
    model   = AGBDNet(s2_encoder=encoder, embed_dim=128)
    sd      = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(dev).eval()
    return model, dev, alpha_mean, alpha_std


def read_clipped(tif_path, aoi_geom, bands=None):
    with rasterio.open(tif_path) as src:
        tif_crs = src.crs
        transformer = Transformer.from_crs("EPSG:4326", tif_crs, always_xy=True)
        coords = AOI_GEOJSON["coordinates"][0]
        projected_coords = [transformer.transform(x, y) for x, y in coords]
        projected_aoi    = {"type": "Polygon", "coordinates": [projected_coords]}
        out_img, out_transform = mask(src, [projected_aoi], crop=True)
        out_img = out_img.astype(np.float32)
        crs     = src.crs
        nodata  = src.nodata
    if bands is not None:
        out_img = out_img[bands]
    print(out_img.shape)
    return out_img, out_transform, crs, nodata


def read_window_with_halo(data, row0, col0, h, w, pad, fill_value=0.0):
    _, H, W = data.shape
    r0, c0  = row0 - pad, col0 - pad
    r1, c1  = row0 + h + pad, col0 + w + pad
    rr0, cc0 = max(0, r0), max(0, c0)
    rr1, cc1 = min(H, r1), min(W, c1)
    out = np.full((data.shape[0], h + 2*pad, w + 2*pad), fill_value, dtype=np.float32)
    top  = rr0 - r0
    left = cc0 - c0
    out[:, top:top + (rr1-rr0), left:left + (cc1-cc0)] = data[:, rr0:rr1, cc0:cc1]
    return out


def make_blend_weights(h, w, overlap):
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)
    oy = int(min(overlap, h // 2))
    ox = int(min(overlap, w // 2))
    if oy > 0:
        ramp = np.linspace(0., 1., oy, dtype=np.float32)
        wy[:oy] *= ramp; wy[-oy:] *= ramp[::-1]
    if ox > 0:
        ramp = np.linspace(0., 1., ox, dtype=np.float32)
        wx[:ox] *= ramp; wx[-ox:] *= ramp[::-1]
    return np.clip(wy[:, None] * wx[None, :], 1e-6, 1.0).astype(np.float32)


@torch.no_grad()
def predict_tile(model, device, s2_halo, alpha_tile, patch, batch_pixels):
    p = patch // 2
    _, Hh, Wh = s2_halo.shape
    h, w = Hh - 2*p, Wh - 2*p

    s2      = torch.from_numpy(s2_halo).unsqueeze(0).to(device)
    patches = F.unfold(s2, kernel_size=patch, stride=1)
    patches = patches.squeeze(0).T.contiguous().view(h*w, 10, patch, patch)

    a = torch.from_numpy(alpha_tile).to(device)
    a = a.view(a.shape[0], h*w).T.contiguous()

    preds = torch.empty(h*w, device=device, dtype=torch.float32)
    for i in range(0, h*w, batch_pixels):
        j = min(h*w, i + batch_pixels)
        preds[i:j] = model(patches[i:j], a[i:j]).view(-1)

    return preds.view(h, w).cpu().numpy()


def predict_full_map(model, dev, alpha_mean, alpha_std, s2_data, alpha_data,
                     scl_data=None, tile=256, overlap=64, patch=21,
                     batch_pixels=8192, s2_scale=None):
    pad    = patch // 2
    stride = tile - overlap
    _, H, W = s2_data.shape

    if s2_scale is not None:
        s2_data = s2_data * np.float32(s2_scale)

    alpha_data[:64] = (alpha_data[:64] - alpha_mean[:, None, None]) / \
                      (alpha_std[:, None, None] + 1e-6)

    acc  = np.zeros((H, W), dtype=np.float32)
    wacc = np.zeros((H, W), dtype=np.float32)

    if scl_data is not None:
        scl        = scl_data[0]
        cloud_mask = np.isin(scl, list(CLOUD_CODES)) | (scl == SNOW_CODE)
    else:
        cloud_mask = np.zeros((H, W), dtype=bool)

    for row0 in range(0, H, stride):
        h = min(tile, H - row0)
        for col0 in range(0, W, stride):
            w = min(tile, W - col0)

            s2_halo    = read_window_with_halo(s2_data, row0, col0, h, w, pad)
            alpha_tile = alpha_data[:, row0:row0+h, col0:col0+w]

            pred = predict_tile(model, dev, s2_halo, alpha_tile, patch, batch_pixels)
            pred = (np.exp(pred) - EPS).astype(np.float32)

            ww = make_blend_weights(h, w, overlap)
            center_s2 = s2_halo[:, pad:pad+h, pad:pad+w]
            invalid   = np.all(center_s2 == 0, axis=0)
            ww        = ww.copy()
            ww[invalid] = 0.0

            acc [row0:row0+h, col0:col0+w] += pred * ww
            wacc[row0:row0+h, col0:col0+w] += ww

    pred_map          = np.full((H, W), NODATA, dtype=np.float32)
    valid             = wacc > 0
    pred_map[valid]   = acc[valid] / wacc[valid]
    return pred_map, cloud_mask


def parse_date_from_filename(fname):
    match = re.search(r"(\d{4}[-_]\d{2}[-_]\d{2})", fname)
    if match:
        return match.group(1).replace("_", "-")
    match = re.search(r"(\d{8})", fname)
    if match:
        d = match.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return os.path.splitext(fname)[0]


# ─────────────────────────────────────────────
# NEW — Save GeoTIFF helper
# ─────────────────────────────────────────────

def save_geotiff(data: np.ndarray, out_path: str, transform, crs,
                 nodata: float = NODATA):
    """Save a [H, W] float32 array as a single-band GeoTIFF."""
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",     # lossless compression, keeps file size small
        predictor=3,            # floating-point predictor for better compression
    ) as dst:
        dst.write(data, 1)


# ─────────────────────────────────────────────
# NEW — Clip leaf fraction map to AOI + reproject to prediction grid
# ─────────────────────────────────────────────

def load_leaf_fraction(pred_transform, pred_crs, pred_shape):
    """
    Load leaf fraction map, clip to AOI, and reproject to match
    the prediction map grid exactly (same transform, shape, CRS).

    Returns [H, W] float32 array aligned to prediction map.
    """
    H, W = pred_shape

    with rasterio.open(LEAF_FRAC_TIF) as lf_src:
        # Project AOI to leaf fraction CRS for clipping
        transformer  = Transformer.from_crs("EPSG:4326", lf_src.crs, always_xy=True)
        coords       = AOI_GEOJSON["coordinates"][0]
        proj_coords  = [transformer.transform(x, y) for x, y in coords]
        proj_aoi     = {"type": "Polygon", "coordinates": [proj_coords]}

        lf_clipped, lf_transform = mask(lf_src, [proj_aoi], crop=True)
        lf_clipped = lf_clipped[0].astype(np.float32)   # [H_lf, W_lf]
        lf_crs     = lf_src.crs

    # Reproject leaf fraction to exactly match prediction map grid
    lf_aligned = np.zeros((H, W), dtype=np.float32)
    reproject(
        source=lf_clipped,
        destination=lf_aligned,
        src_transform=lf_transform,
        src_crs=lf_crs,
        dst_transform=pred_transform,
        dst_crs=pred_crs,
        resampling=Resampling.bilinear,   # bilinear ok — leaf fraction is continuous
    )

    # Clip to valid range [0, 1] after reproject
    lf_aligned = np.clip(lf_aligned, 0.0, 1.0)

    print(f"  Leaf fraction map aligned to prediction grid: {lf_aligned.shape}")
    print(f"  Leaf fraction range: {lf_aligned.min():.4f} – {lf_aligned.max():.4f}")
    print(f"  Leaf fraction mean:  {lf_aligned[lf_aligned > 0].mean():.4f}")

    return lf_aligned


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    aoi_shape = shape(AOI_GEOJSON)
    print(aoi_shape)
    model, dev, alpha_mean, alpha_std = load_model(CKPT_PATH, DEVICE)

    nlcd_data = None
    if NLCD_TIF is not None:
        print("Loading NLCD raster...")
        nlcd_clipped, _, _, _ = read_clipped(NLCD_TIF, aoi_shape)
        nlcd_data = nlcd_clipped[0].astype(np.int32)

    s2_files = sorted([
        f for f in os.listdir(S2_DIR)
        if f.endswith(".tif") or f.endswith(".tiff")
    ])
    print(f"Found {len(s2_files)} S2 files")

    # Leaf fraction loaded once after first prediction to get the grid
    leaf_frac_map = None

    results = []

    for fname in s2_files:
        date     = parse_date_from_filename(fname)
        s2_path  = os.path.join(S2_DIR, fname)
        alpha_path = os.path.join(ALPHA_TIF, fname)
        print(f"\nProcessing {date}...")

        s2_data, out_transform, out_crs, _ = read_clipped(s2_path, aoi_shape)
        alpha_data, _, _, _                = read_clipped(alpha_path, aoi_shape)

        pred_map, cloud_mask = predict_full_map(
            model, dev, alpha_mean, alpha_std,
            s2_data, alpha_data.copy(),
            scl_data=None,
            tile=TILE, overlap=OVERLAP, patch=PATCH,
            batch_pixels=BATCH_PIXELS, s2_scale=S2_SCALE,
        )

        H_p, W_p = pred_map.shape

        # Load and align leaf fraction map on first iteration
        if leaf_frac_map is None:
            print("  Loading and aligning leaf fraction map...")
            leaf_frac_map = load_leaf_fraction(
                pred_transform=out_transform,
                pred_crs=out_crs,
                pred_shape=(H_p, W_p),
            )

        # ── Compute leaf and wood biomass maps ──────────────────────────────
        valid_mask = pred_map != NODATA

        # Leaf biomass = total AGBD × leaf fraction
        leaf_map = np.full_like(pred_map, NODATA)
        leaf_map[valid_mask] = pred_map[valid_mask] * leaf_frac_map[valid_mask]

        # Wood biomass = total AGBD − leaf biomass
        # Theoretically equals winter AGBD (no leaves)
        wood_map = np.full_like(pred_map, NODATA)
        wood_map[valid_mask] = pred_map[valid_mask] - leaf_map[valid_mask]

        # ── Save GeoTIFFs ───────────────────────────────────────────────────
        stem = os.path.splitext(fname)[0]

        agbd_path = os.path.join(OUT_DIR_AGBD, f"{stem}_agbd.tif")
        leaf_path = os.path.join(OUT_DIR_LEAF, f"{stem}_leaf.tif")
        wood_path = os.path.join(OUT_DIR_WOOD, f"{stem}_wood.tif")

        save_geotiff(pred_map, agbd_path, out_transform, out_crs)
        save_geotiff(leaf_map, leaf_path, out_transform, out_crs)
        save_geotiff(wood_map, wood_path, out_transform, out_crs)

        print(f"  Saved: {agbd_path}")
        print(f"  Saved: {leaf_path}")
        print(f"  Saved: {wood_path}")

        # ── Statistics for CSV ──────────────────────────────────────────────
        def _crop(arr): return arr[:H_p, :W_p]

        clear_mask = valid_mask & ~_crop(cloud_mask)

        if nlcd_data is not None:
            forest_mask_arr = np.isin(_crop(nlcd_data), list(FOREST_CLASSES))
            mean_mask       = clear_mask & forest_mask_arr
        else:
            mean_mask = clear_mask

        valid_pixels   = pred_map[mean_mask]
        total_pixels   = int(valid_mask.sum())
        cloud_coverage = float(_crop(cloud_mask)[valid_mask].mean()) \
                         if total_pixels > 0 else 1.0

        if len(valid_pixels) == 0:
            print(f"  [{date}] No valid clear pixels")
            results.append({
                "date":           date,
                "mean_agbd":      np.nan,
                "mean_leaf_agbd": np.nan,
                "mean_wood_agbd": np.nan,
                "leaf_fraction":  np.nan,
                "valid_pixels":   0,
                "cloud_coverage": cloud_coverage,
                "filename":       fname,
            })
            continue

        mean_agbd = float(pred_map[mean_mask].mean())
        mean_leaf = float(leaf_map[mean_mask].mean())
        mean_wood = float(wood_map[mean_mask].mean())
        mean_lf   = float(leaf_frac_map[mean_mask].mean())

        print(f"  [{date}] "
              f"AGBD={mean_agbd:.2f}  "
              f"leaf={mean_leaf:.2f}  "
              f"wood={mean_wood:.2f}  "
              f"leaf_frac={mean_lf:.4f}  "
              f"valid={len(valid_pixels)}/{total_pixels}px  "
              f"cloud={cloud_coverage:.2f}")

        results.append({
            "date":           date,
            "mean_agbd":      round(mean_agbd, 4),
            "mean_leaf_agbd": round(mean_leaf, 4),
            "mean_wood_agbd": round(mean_wood, 4),
            "leaf_fraction":  round(mean_lf,   4),
            "valid_pixels":   len(valid_pixels),
            "cloud_coverage": cloud_coverage,
            "filename":       fname,
        })

    # ── Save CSV ────────────────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values("date")
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")
    print(df.to_string(index=False))

    # ── Print seasonal summary ───────────────────────────────────────────────
    df["month"] = pd.to_datetime(df["date"]).dt.month
    summer = df[df["month"].isin([6, 7, 8])].dropna(subset=["mean_agbd"])
    winter = df[df["month"].isin([12, 1, 2])].dropna(subset=["mean_agbd"])

    if len(summer) > 0 and len(winter) > 0:
        print(f"\n{'='*55}")
        print(f"Seasonal comparison")
        print(f"{'='*55}")
        print(f"  Summer mean AGBD:      {summer['mean_agbd'].mean():.2f} Mg/ha")
        print(f"  Summer mean leaf AGBD: {summer['mean_leaf_agbd'].mean():.2f} Mg/ha")
        print(f"  Summer mean wood AGBD: {summer['mean_wood_agbd'].mean():.2f} Mg/ha")
        print(f"  Winter mean AGBD:      {winter['mean_agbd'].mean():.2f} Mg/ha")
        print(f"  Wood vs Winter AGBD diff: "
              f"{summer['mean_wood_agbd'].mean() - winter['mean_agbd'].mean():+.2f} Mg/ha")
        print(f"  (close to 0 = leaf fraction map is well calibrated)")


if __name__ == "__main__":
    main()
