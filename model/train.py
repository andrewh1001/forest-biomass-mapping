from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os as _os

# ---------------------------------------------------------------------------
# BAAD-derived leaf fraction priors — fallback only
# Used when a sample's acquisition date has no harmonic entry.
# ---------------------------------------------------------------------------
LEAF_PRIOR = {
    "temperate_deciduous": {"mean": 0.084,  "std": 0.048},
    "boreal_conifer":      {"mean": 0.1241, "std": 0.0717},
    "mixed":               {"mean": 0.1022, "std": 0.06},
}


# ---------------------------------------------------------------------------
# Harmonic leaf fraction table
# ---------------------------------------------------------------------------

def load_harmonic_leaf_fractions(
    csv_path: str = "agbd_timeseries_harmonic.csv",
) -> dict:
    """
    Load the harmonic-derived leaf fraction per acquisition date from
    agbd_timeseries_harmonic.csv (output of harmonic_decomposition.py).

    Returns
    -------
    dict : {"YYYY-MM-DD": float}
        AOI-mean harmonic leaf fraction per S2 date.
        Values are clipped to [0, 1]; winter dates where the non-negativity
        constraint applies will have fraction = 0.0.

    The LUT is intentionally AOI-level (not per-pixel) because the harmonic
    fit is computed from the spatial mean AGBD timeseries.  Per-GEDI-shot
    fractions inherit the AOI fraction scaled by that shot's total AGBD:
        leaf_label = gedi_agbd * harmonic_leaf_frac(date)
        wood_label = gedi_agbd * (1 - harmonic_leaf_frac(date))
    This is better than BAAD because the fraction varies by date (0 in winter,
    ~0.20 at peak summer) rather than being a fixed species-level constant.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    if "leaf_fraction_harmonic" in df.columns:
        frac_col = "leaf_fraction_harmonic"
    elif "leaf_harmonic" in df.columns and "agbd_fitted" in df.columns:
        df["leaf_fraction_harmonic"] = (
            df["leaf_harmonic"] / (df["agbd_fitted"] + 1e-6)
        ).clip(0.0, 1.0)
        frac_col = "leaf_fraction_harmonic"
    else:
        raise ValueError(
            "agbd_timeseries_harmonic.csv must contain 'leaf_fraction_harmonic' "
            "or both 'leaf_harmonic' and 'agbd_fitted'."
        )

    lut = dict(zip(df["date_str"], df[frac_col].clip(0.0, 1.0)))
    print(f"[harmonic] Loaded leaf fractions for {len(lut)} dates "
          f"({min(lut.keys())} → {max(lut.keys())})")
    print(f"[harmonic] Fraction range: "
          f"{min(lut.values()):.4f} – {max(lut.values()):.4f}  "
          f"(0.0 = winter, ~0.20 = peak summer)")
    return lut


# Load at import time if CSV exists; train_model() reloads with correct path.
_DEFAULT_CSV = "agbd_timeseries_harmonic.csv"
HARMONIC_LEAF_LUT: dict = (
    load_harmonic_leaf_fractions(_DEFAULT_CSV)
    if _os.path.exists(_DEFAULT_CSV)
    else {}
)
if not HARMONIC_LEAF_LUT:
    print("[harmonic] WARNING: agbd_timeseries_harmonic.csv not found. "
          "Pass harmonic_csv to train_model() or place it in the working directory.")


def get_harmonic_leaf_fraction(
    acq_dates: list,
    forest_types: list,
    device: torch.device,
) -> tuple:
    """
    Return per-sample (leaf_fraction, fraction_std) tensors.

    Lookup priority:
      1. Harmonic LUT for the exact YYYY-MM-DD date        ← primary
      2. BAAD prior for the sample's forest type            ← fallback

    Parameters
    ----------
    acq_dates    : list of "YYYY-MM-DD" strings (len B)
    forest_types : list of forest type strings  (len B)
    device       : torch device

    Returns
    -------
    fracs : FloatTensor [B]  leaf fraction targets
    stds  : FloatTensor [B]  uncertainty (0.0 for harmonic hits, BAAD std for fallbacks)
    """
    fracs = []
    stds  = []
    for date_str, ft in zip(acq_dates, forest_types):
        if date_str in HARMONIC_LEAF_LUT:
            fracs.append(HARMONIC_LEAF_LUT[date_str])
            stds.append(0.0)        # harmonic is deterministic — no std needed
        else:
            # Fallback: BAAD prior for this forest type
            prior = LEAF_PRIOR.get(ft, LEAF_PRIOR["mixed"])
            fracs.append(prior["mean"])
            stds.append(prior["std"])

    return (
        torch.tensor(fracs, dtype=torch.float32, device=device),
        torch.tensor(stds,  dtype=torch.float32, device=device),
    )


# ---------------------------------------------------------------------------
# Dual loss
# ---------------------------------------------------------------------------

def dual_loss(
    preds:        dict,
    y:            torch.Tensor,
    forest_types: list,
    acq_dates:    list,
    device:       torch.device,
    eps:          float = 3e-5,
    alpha:        float = 0.5,   # weight: total AGBD loss (log-space weighted MSE)
    beta:         float = 0.3,   # weight: ratio regularization loss
    gamma:        float = 0.2,   # weight: leaf + wood supervised loss
) -> tuple:
    """
    Composite loss for dual-head leaf / wood AGBD prediction.

    Leaf/wood supervision now uses harmonic-derived date-specific fractions
    instead of BAAD fixed priors. BAAD is retained as a fallback for dates
    outside the harmonic LUT.

    Args
    ----
    preds        : dict with keys 'leaf', 'wood', 'total'  (linear space, [B])
    y            : true total AGBD from GEDI L4A            (linear space, [B])
    forest_types : list of forest-type strings              (len B)
    acq_dates    : list of "YYYY-MM-DD" strings             (len B)
    device       : torch device
    eps/alpha/beta/gamma : loss hyperparameters

    Returns
    -------
    (total_loss, loss_dict)
    """
    # ------------------------------------------------------------------
    # 1. Total AGBD loss — preserves original weighted log-space MSE
    # ------------------------------------------------------------------
    y_log    = torch.log(y + eps)
    pred_log = torch.log(preds["total"] + eps)
    w        = 1.0 + 0.5 * (y / y.mean())
    loss_total = torch.mean(w * (pred_log - y_log) ** 2)

    # ------------------------------------------------------------------
    # 2. Harmonic leaf fraction targets (with BAAD fallback)
    # ------------------------------------------------------------------
    target_fractions, target_stds = get_harmonic_leaf_fraction(
        acq_dates, forest_types, device
    )

    # Per-sample leaf / wood targets: GEDI total × harmonic fraction
    leaf_target = y * target_fractions
    wood_target = y * (1.0 - target_fractions)

    loss_leaf = F.huber_loss(preds["leaf"], leaf_target)
    loss_wood = F.huber_loss(preds["wood"], wood_target)

    # ------------------------------------------------------------------
    # 3. Ratio regularization — penalise departures from target fraction
    #    For harmonic dates: std=0 → use a small fixed floor (0.03) so
    #    the regularizer doesn't blow up; BAAD dates use their own std.
    # ------------------------------------------------------------------
    reg_std         = torch.clamp(target_stds, min=0.03)
    pred_fraction   = preds["leaf"] / (preds["total"] + 1e-6)
    ratio_deviation = (pred_fraction - target_fractions) / reg_std
    loss_ratio      = (ratio_deviation ** 2).mean()

    # ------------------------------------------------------------------
    # 4. Composite
    # ------------------------------------------------------------------
    total_loss = (
        alpha * loss_total
        + beta  * loss_ratio
        + gamma * (loss_leaf + loss_wood)
    )

    loss_dict = {
        "loss_total": loss_total.item(),
        "loss_leaf":  loss_leaf.item(),
        "loss_wood":  loss_wood.item(),
        "loss_ratio": loss_ratio.item(),
        "n_harmonic": sum(1 for d in acq_dates if d in HARMONIC_LEAF_LUT),
        "n_fallback": sum(1 for d in acq_dates if d not in HARMONIC_LEAF_LUT),
    }

    return total_loss, loss_dict


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,                        # kept for API compatibility; unused when dual_head=True
    optimizer,
    epochs,
    device=None,
    scheduler=None,
    cloud_frac_thresh=None,
    dual_head: bool = False,
    harmonic_csv: str = "agbd_timeseries_harmonic.csv",   # NEW: path override
):
    global HARMONIC_LEAF_LUT

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reload LUT with explicit path in case working directory differs
    if dual_head and harmonic_csv != _DEFAULT_CSV and _os.path.exists(harmonic_csv):
        HARMONIC_LEAF_LUT = load_harmonic_leaf_fractions(harmonic_csv)

    model.to(device)
    print(f"Using device: {device}  |  dual_head={dual_head}")
    if dual_head:
        n_harmonic = len(HARMONIC_LEAF_LUT)
        print(f"[harmonic] {n_harmonic} dates in LUT "
              f"({'OK' if n_harmonic > 0 else 'EMPTY — check CSV path'})")

    train_losses = []
    val_losses   = []
    eps = 3e-5

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}/{epochs}")

        # ----------------------------------------------------------------
        # TRAIN
        # ----------------------------------------------------------------
        model.train()
        running_train_loss = 0.0
        n_train            = 0
        epoch_harmonic     = 0
        epoch_fallback     = 0

        for batch in tqdm(train_loader, desc="Train"):
            s2           = batch["s2"].to(device)
            ae           = batch["ae"].to(device)
            y            = batch["target"].to(device).float()
            cloud_frac   = batch["cloud_frac"].to(device)
            center_cloud = batch["center_cloud"].to(device)
            forest_types = batch["forest_type"]   # list[str], stays on CPU
            acq_dates    = batch["acq_date"]       # list[str], stays on CPU

            # Cloud filtering — identical to original
            keep = (~center_cloud) & (cloud_frac <= cloud_frac_thresh)
            if keep.sum() == 0:
                continue
            keep_idx     = keep.nonzero(as_tuple=True)[0].tolist()
            s2           = s2[keep]
            ae           = ae[keep]
            y            = y[keep]
            forest_types = [forest_types[i] for i in keep_idx]
            acq_dates    = [acq_dates[i]    for i in keep_idx]

            if dual_head:
                preds = model(s2, ae)
                loss, loss_dict = dual_loss(
                    preds, y, forest_types, acq_dates, device, eps
                )
                epoch_harmonic += loss_dict["n_harmonic"]
                epoch_fallback += loss_dict["n_fallback"]
            else:
                # Original single-head path — completely unchanged
                y_log    = torch.log(y + eps)
                pred_log = model(s2, ae)
                w        = 1.0 + 0.5 * (y / y.mean())
                loss     = torch.mean(w * (pred_log - y_log) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = y.size(0)
            running_train_loss += loss.item() * bs
            n_train            += bs

        avg_train_loss = running_train_loss / max(n_train, 1)
        train_losses.append(avg_train_loss)
        print(f"  Train loss: {avg_train_loss:.4f}", end="")
        if dual_head:
            print(f"  [harmonic={epoch_harmonic} | fallback={epoch_fallback}]", end="")
        print()

        # ----------------------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------------------
        model.eval()
        running_val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val  "):
                s2           = batch["s2"].to(device)
                ae           = batch["ae"].to(device)
                y            = batch["target"].to(device).float()
                cloud_frac   = batch["cloud_frac"].to(device)
                center_cloud = batch["center_cloud"].to(device)

                keep = (~center_cloud) & (cloud_frac <= cloud_frac_thresh)
                if keep.sum() == 0:
                    continue
                s2 = s2[keep]
                ae = ae[keep]
                y  = y[keep]

                if dual_head:
                    preds = model(s2, ae)
                    loss  = torch.mean((preds["total"] - y) ** 2)
                else:
                    pred_log = model(s2, ae)
                    preds_   = torch.exp(pred_log) - eps
                    loss     = torch.mean((preds_ - y) ** 2)

                bs = y.size(0)
                running_val_loss += loss.item() * bs
                n_val            += bs

        avg_val_loss = running_val_loss / max(n_val, 1)
        val_losses.append(avg_val_loss)
        print(f"  Val   loss: {avg_val_loss:.4f}")

        if scheduler is not None:
            scheduler.step(avg_val_loss)

    return train_losses, val_losses
