"""
harmonic_decomposition.py
=========================
Decomposes the AGBD timeseries into:
  - Wood biomass  : constant structural component (season-invariant)
  - Leaf biomass  : seasonal oscillation component (phenological)

Using harmonic regression:
  AGBD(t) = a + b*sin(2π*doy/365 + φ) + c*sin(4π*doy/365 + ψ)
              ↑              ↑                      ↑
           wood mean     annual cycle          semi-annual cycle
                         (leaf flush/drop)     (secondary peak)

The constant term 'a' is the wood biomass estimate.
The oscillating terms capture leaf phenology.

Inputs:
  agbd_timeseries.csv   — from inference_map.py

Outputs:
  outputs/histograms/harmonic_decomposition.png
  outputs/histograms/wood_leaf_decomposition.png
  harmonic_params.csv   — fitted parameters per pixel (if pixel-level run)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_PATH  = "agbd_timeseries.csv"
OUT_DIR   = "outputs/histograms"
NODATA    = -9999.0

import os
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# HARMONIC MODELS
# ─────────────────────────────────────────────

def harmonic_1(doy, a, b, phi):
    """
    Single harmonic: AGBD = a + b*sin(2π*doy/365 + φ)
    a   = wood biomass (constant, season-invariant)
    b   = leaf amplitude (half the summer-winter swing)
    phi = phase offset (controls timing of peak)
    """
    return a + b * np.sin(2 * np.pi * doy / 365.0 + phi)


def harmonic_2(doy, a, b, phi, c, psi):
    """
    Double harmonic: adds semi-annual term for asymmetric phenology.
    AGBD = a + b*sin(2π*doy/365 + φ) + c*sin(4π*doy/365 + ψ)
    Captures spring green-up / fall senescence asymmetry.
    """
    return (a
            + b * np.sin(2 * np.pi * doy / 365.0 + phi)
            + c * np.sin(4 * np.pi * doy / 365.0 + psi))


def fit_harmonic(doy, agbd, model="double"):
    """
    Fit harmonic model to AGBD timeseries.

    Returns:
        popt   — fitted parameters
        pcov   — parameter covariance
        r2     — R² of fit
        model  — which model was used
    """
    # Initial guesses
    a0   = agbd.mean()
    b0   = (agbd.max() - agbd.min()) / 2
    phi0 = -np.pi / 2   # peak around summer solstice (DOY ~172)

    try:
        if model == "double":
            p0    = [a0, b0, phi0, b0*0.2, 0.0]
            popt, pcov = curve_fit(harmonic_2, doy, agbd, p0=p0, maxfev=10000)
            agbd_fit   = harmonic_2(doy, *popt)
        else:
            p0    = [a0, b0, phi0]
            popt, pcov = curve_fit(harmonic_1, doy, agbd, p0=p0, maxfev=10000)
            agbd_fit   = harmonic_1(doy, *popt)

        ss_res = np.sum((agbd - agbd_fit) ** 2)
        ss_tot = np.sum((agbd - agbd.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return popt, pcov, r2, model

    except Exception as e:
        print(f"  Fit failed ({e}) — falling back to single harmonic")
        try:
            p0    = [a0, b0, phi0]
            popt, pcov = curve_fit(harmonic_1, doy, agbd, p0=p0, maxfev=10000)
            agbd_fit   = harmonic_1(doy, *popt)
            ss_res = np.sum((agbd - agbd_fit) ** 2)
            ss_tot = np.sum((agbd - agbd.mean()) ** 2)
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            return popt, pcov, r2, "single"
        except Exception:
            return None, None, 0.0, None


# ─────────────────────────────────────────────
# LOAD AND PREPARE DATA
# ─────────────────────────────────────────────

print("Loading timeseries...")
df = pd.read_csv(CSV_PATH)
df["date"]  = pd.to_datetime(df["date"])
df["doy"]   = df["date"].dt.dayofyear
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month
df = df.dropna(subset=["mean_agbd"]).sort_values("date").reset_index(drop=True)

print(f"  Dates: {len(df)}  ({df['date'].min().date()} → {df['date'].max().date()})")
print(f"  AGBD range: {df['mean_agbd'].min():.1f} – {df['mean_agbd'].max():.1f} Mg/ha")
print(f"  AGBD mean:  {df['mean_agbd'].mean():.1f} Mg/ha")

doy  = df["doy"].values.astype(float)
agbd = df["mean_agbd"].values.astype(float)


# ─────────────────────────────────────────────
# FIT HARMONIC MODELS
# ─────────────────────────────────────────────

print("\nFitting harmonic models...")

# Single harmonic
popt1, pcov1, r2_1, _ = fit_harmonic(doy, agbd, model="single")
# Double harmonic
popt2, pcov2, r2_2, _ = fit_harmonic(doy, agbd, model="double")

print(f"\n  Single harmonic R²: {r2_1:.4f}")
print(f"  Double harmonic R²: {r2_2:.4f}")

# Always prefer double harmonic if fit succeeded; fallback to single only on failure
use_double = popt2 is not None
popt       = popt2 if use_double else popt1
r2         = r2_2  if use_double else r2_1
model_used = "double" if use_double else "single"
r2_improvement = r2_2 - r2_1 if popt2 is not None else 0.0
print(f"  Using: {model_used} harmonic (R²={r2:.4f}, ΔR²={r2_improvement:+.4f} vs single)")

# Extract components
if model_used == "double":
    a, b, phi, c, psi = popt
else:
    a, b, phi = popt
    c, psi    = 0.0, 0.0

print(f"\n=== Harmonic Decomposition Results ===")
print(f"  Wood biomass (constant term a):  {a:.2f} Mg/ha")
print(f"  Leaf amplitude (b):              {abs(b):.2f} Mg/ha")
print(f"  Phase offset (phi):              {phi:.3f} rad")
if model_used == "double":
    print(f"  Semi-annual amplitude (c):       {abs(c):.2f} Mg/ha")
    print(f"  Semi-annual phase (psi):         {psi:.3f} rad")
    print(f"  [Double harmonic captures spring/autumn asymmetry in leaf phenology]")

# Peak DOY from phase
peak_doy = int((np.pi/2 - phi) * 365 / (2*np.pi)) % 365
trough_doy = int((3*np.pi/2 - phi) * 365 / (2*np.pi)) % 365
print(f"\n  Estimated peak DOY:   {peak_doy}  (~{pd.Timestamp('2021-01-01') + pd.Timedelta(days=peak_doy-1):%b %d})")
print(f"  Estimated trough DOY: {trough_doy}  (~{pd.Timestamp('2021-01-01') + pd.Timedelta(days=trough_doy-1):%b %d})")

# Compute fitted curves — smooth daily resolution
doy_smooth = np.linspace(1, 365, 365)
if model_used == "double":
    agbd_fit_smooth = harmonic_2(doy_smooth, *popt)
else:
    agbd_fit_smooth = harmonic_1(doy_smooth, *popt)

# Fitted values at observed dates
if model_used == "double":
    agbd_fitted = harmonic_2(doy, *popt)
else:
    agbd_fitted = harmonic_1(doy, *popt)

# ── Non-negativity constraint on leaf biomass ──────────────────────────────
# Raw harmonic oscillation around constant term 'a' can go negative in winter.
# Physical constraint: leaf biomass >= 0.
#
# Constrained decomposition:
#   leaf(t) = max(agbd_fit(t) - a, 0)   always >= 0
#   wood(t) = agbd_fit(t) - leaf(t)     = min(agbd_fit(t), a)
#
# Wood is no longer exactly constant but stays <= a and is always positive.
# In summer: leaf > 0, wood ≈ a (when agbd_fit > a)
# In winter: leaf = 0, wood = agbd_fit (all predicted biomass is wood)

# Smooth curves
leaf_raw_smooth  = agbd_fit_smooth - a
leaf_smooth      = np.maximum(leaf_raw_smooth, 0.0)
wood_smooth      = agbd_fit_smooth - leaf_smooth    # = min(agbd_fit, a)

# At observed dates
leaf_raw_fitted  = agbd_fitted - a
leaf_fitted      = np.maximum(leaf_raw_fitted, 0.0)
wood_fitted      = agbd_fitted - leaf_fitted

print(f"\n  Non-negativity constraint applied:")
print(f"  Dates with leaf < 0 (corrected): "
      f"{(leaf_raw_fitted < 0).sum()} / {len(leaf_raw_fitted)}")
print(f"  Constrained leaf range: "
      f"{leaf_fitted.min():.2f} – {leaf_fitted.max():.2f} Mg/ha")
print(f"  Constrained wood range: "
      f"{wood_fitted.min():.2f} – {wood_fitted.max():.2f} Mg/ha")

# Add to dataframe
df["agbd_fitted"]            = agbd_fitted
df["wood_harmonic"]          = wood_fitted
df["leaf_harmonic"]          = leaf_fitted
df["leaf_raw_harmonic"]      = leaf_raw_fitted   # unconstrained, for reference
df["leaf_fraction_harmonic"] = df["leaf_harmonic"] / (df["agbd_fitted"] + 1e-6)


# ─────────────────────────────────────────────
# SEASONAL STATISTICS
# ─────────────────────────────────────────────

summer = df[df["month"].isin([6, 7, 8])]
winter = df[df["month"].isin([12, 1, 2])]

print(f"\n=== Seasonal Statistics (Harmonic Decomposition) ===")
print(f"  {'':20} {'Summer':>10} {'Winter':>10} {'Diff':>10}")
print(f"  {'-'*52}")

for label, col in [
    ("Total AGBD",   "mean_agbd"),
    ("Wood (harmonic)", "wood_harmonic"),
    ("Leaf (harmonic)", "leaf_harmonic"),
]:
    s_mean = summer[col].mean() if col in summer else np.nan
    w_mean = winter[col].mean() if col in winter else np.nan
    diff   = s_mean - w_mean
    print(f"  {label:20} {s_mean:>10.2f} {w_mean:>10.2f} {diff:>+10.2f}")

# CV comparison
wood_cv_harmonic = df["wood_harmonic"].std() / df["wood_harmonic"].mean()
leaf_cv_harmonic = df["leaf_harmonic"].std()  / (df["leaf_harmonic"].mean() + 1e-6)
total_cv         = df["mean_agbd"].std()       / df["mean_agbd"].mean()

print(f"\n=== Coefficient of Variation ===")
print(f"  Total AGBD CV:                  {total_cv:.4f}")
print(f"  Wood (harmonic, constrained) CV:{wood_cv_harmonic:.4f}  "
      f"(low = seasonally stable)")
print(f"  Leaf (harmonic, constrained) CV:{leaf_cv_harmonic:.4f}  "
      f"(high = phenologically active)")

print(f"\n=== Implied Leaf Fraction ===")
print(f"  Summer leaf fraction:  {summer['leaf_fraction_harmonic'].mean():.4f}")
print(f"  Winter leaf fraction:  {winter['leaf_fraction_harmonic'].mean():.4f}")
print(f"  Annual mean:           {df['leaf_fraction_harmonic'].mean():.4f}")
print(f"  Max (peak summer):     {df['leaf_fraction_harmonic'].max():.4f}")


# ─────────────────────────────────────────────
# PLOT 1 — Timeseries decomposition
# ─────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle(
    f"Harmonic Decomposition of AGBD Timeseries (deciduous pixels)\n"
    f"{model_used.capitalize()} harmonic fit  |  R²={r2:.4f}",
    fontsize=13
)

def shade_seasons(ax, df):
    years = df["year"].unique()
    for year in years:
        ax.axvspan(pd.Timestamp(f"{year}-06-01"),
                   pd.Timestamp(f"{year}-08-31"),
                   alpha=0.08, color="green")
        try:
            ax.axvspan(pd.Timestamp(f"{year}-12-01"),
                       pd.Timestamp(f"{year+1}-02-28"),
                       alpha=0.08, color="blue")
        except Exception:
            pass

# Panel 1: Observed + fitted total AGBD
ax = axes[0]
ax.scatter(df["date"], df["mean_agbd"], s=30, color="steelblue",
           zorder=3, label="Observed AGBD")
ax.plot(df["date"], df["agbd_fitted"], "-", color="red",
        linewidth=2, label=f"Harmonic fit (R²={r2:.3f})")
ax.axhline(a, color="saddlebrown", linestyle="--", linewidth=1.5,
           label=f"Wood component: {a:.1f} Mg/ha")
shade_seasons(ax, df)
ax.set_ylabel("AGBD (Mg/ha)")
ax.set_title("Total AGBD: Observed vs Harmonic Fit")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Wood component (constrained)
ax = axes[1]
ax.plot(df["date"], df["wood_harmonic"], "o-", color="saddlebrown",
        markersize=5, linewidth=1.5, label="Wood AGBD (constrained)")
ax.axhline(a, color="saddlebrown", linestyle="--", linewidth=1.5,
           label=f"Harmonic constant a = {a:.2f} Mg/ha")
ax.fill_between(df["date"], df["wood_harmonic"], a,
                where=(df["wood_harmonic"] < a),
                alpha=0.15, color="saddlebrown",
                label="Winter dip below constant (leaf=0 region)")
shade_seasons(ax, df)
ax.set_ylabel("Wood AGBD (Mg/ha)")
ax.set_title(f"Wood Biomass — Constrained Component  (max={a:.2f} Mg/ha)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Leaf component (seasonal, constrained)
ax = axes[2]
ax.plot(df["date"], df["leaf_harmonic"], "o-", color="forestgreen",
        markersize=5, linewidth=1.5, label="Leaf AGBD (constrained ≥ 0)")
ax.plot(df["date"], df["leaf_raw_harmonic"], "--", color="lightgreen",
        linewidth=1.0, alpha=0.7, label="Leaf AGBD (raw, unconstrained)")
ax.axhline(0, color="black", linestyle="--", linewidth=0.8, label="Zero baseline")
ax.axhline(df["leaf_harmonic"].max(), color="forestgreen",
           linestyle=":", linewidth=1, alpha=0.6,
           label=f"Peak: {df['leaf_harmonic'].max():.2f} Mg/ha")
ax.fill_between(df["date"], 0, df["leaf_harmonic"],
                alpha=0.2, color="forestgreen")
shade_seasons(ax, df)
ax.set_ylabel("Leaf AGBD (Mg/ha)")
ax.set_title("Leaf Biomass — Seasonal Component (non-negative constrained)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "harmonic_decomposition.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {out1}")


# ─────────────────────────────────────────────
# PLOT 2 — Annual cycle (DOY)
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Annual Phenological Cycle — Harmonic Decomposition", fontsize=13)

# Left: DOY scatter + smooth curve
ax = axes[0]
sc = ax.scatter(df["doy"], df["mean_agbd"], c=df["year"],
                cmap="viridis", s=30, alpha=0.7, zorder=3, label="Observed")
ax.plot(doy_smooth, agbd_fit_smooth, "r-", linewidth=2.5,
        label=f"{model_used.capitalize()} harmonic fit (R²={r2:.3f})", zorder=4)
# Also plot single harmonic for comparison if double was used
if model_used == "double" and popt1 is not None:
    agbd_fit_single_smooth = harmonic_1(doy_smooth, *popt1)
    ax.plot(doy_smooth, agbd_fit_single_smooth, "r--", linewidth=1.2, alpha=0.5,
            label=f"Single harmonic (R²={r2_1:.3f})", zorder=3)
ax.axhline(a, color="saddlebrown", linestyle="--", linewidth=1.5,
           label=f"Wood: {a:.1f} Mg/ha")
plt.colorbar(sc, ax=ax, label="Year")
ax.set_xlabel("Day of Year")
ax.set_ylabel("AGBD (Mg/ha)")
ax.set_title("AGBD vs Day of Year (coloured by year)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: Decomposition area plot over DOY (constrained)
ax = axes[1]
ax.fill_between(doy_smooth, 0, wood_smooth,
                alpha=0.7, color="saddlebrown",
                label=f"Wood: max={wood_smooth.max():.1f} Mg/ha")
ax.fill_between(doy_smooth, wood_smooth, agbd_fit_smooth,
                alpha=0.6, color="forestgreen",
                label=f"Leaf: max={leaf_smooth.max():.1f} Mg/ha")
ax.axhline(a, color="saddlebrown", linestyle="--", linewidth=1,
           alpha=0.6, label=f"Constant a={a:.1f}")
ax.set_xlabel("Day of Year")
ax.set_ylabel("AGBD (Mg/ha)")
ax.set_title("Wood + Leaf Decomposition Over Annual Cycle")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Add month labels on x-axis
month_doys   = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
ax.set_xticks(month_doys)
ax.set_xticklabels(month_labels, rotation=45)
axes[0].set_xticks(month_doys)
axes[0].set_xticklabels(month_labels, rotation=45)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "annual_cycle_decomposition.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out2}")


# ─────────────────────────────────────────────
# PLOT 3 — Compare harmonic vs fixed ratio vs dual-head
# ─────────────────────────────────────────────

# Only runs if all three columns exist in CSV
has_dual   = "mean_wood_agbd" in df.columns and df["mean_wood_agbd"].notna().any()
has_fixed  = True   # leaf_harmonic always available after fitting

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Wood Biomass Estimation Comparison", fontsize=13)

# Left: Wood timeseries comparison
ax = axes[0]
ax.plot(df["date"], df["wood_harmonic"], "o-", color="saddlebrown",
        markersize=4, linewidth=1.5, label=f"Harmonic (constant={a:.1f})")

if has_dual:
    ax.plot(df["date"], df["mean_wood_agbd"], "s--", color="steelblue",
            markersize=4, linewidth=1.2, label="Dual-head model")

# Fixed ratio wood: total × (1 - leaf_frac)
fixed_leaf_frac = 0.147   # from earlier diagnostic
df["wood_fixed"] = df["mean_agbd"] * (1 - fixed_leaf_frac)
ax.plot(df["date"], df["wood_fixed"], "^:", color="gray",
        markersize=4, linewidth=1.0, label=f"Fixed ratio (×{1-fixed_leaf_frac:.3f})")

shade_seasons(ax, df)
ax.set_ylabel("Wood AGBD (Mg/ha)")
ax.set_title("Wood Biomass Over Time")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

# Right: Seasonal variability comparison (CV)
methods = ["Fixed ratio", "Harmonic\n(constrained)"]
cvs     = [
    df["wood_fixed"].std()    / df["wood_fixed"].mean(),
    wood_cv_harmonic,
]
colors  = ["gray", "saddlebrown"]

if has_dual:
    methods.append("Dual-head")
    cvs.append(df["mean_wood_agbd"].std() / df["mean_wood_agbd"].mean())
    colors.append("steelblue")

ax = axes[1]
bars = ax.bar(methods, cvs, color=colors, alpha=0.8, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Coefficient of Variation (lower = more stable)")
ax.set_title("Wood Biomass Seasonal Stability\n(CV = std/mean, ideal = 0)")
ax.grid(True, alpha=0.3, axis="y")

for bar, cv in zip(bars, cvs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{cv:.4f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "wood_comparison.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out3}")


# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────

results = {
    "model":               model_used,
    "r2":                  round(r2, 4),
    "r2_single":           round(r2_1, 4),
    "r2_double":           round(r2_2, 4) if popt2 is not None else None,
    "r2_improvement":      round(r2_2 - r2_1, 4) if popt2 is not None else None,
    "wood_biomass_mgha":   round(a, 2),
    "leaf_amplitude_mgha": round(abs(b), 2),
    "phi":                 round(phi, 4),
    "c":                   round(abs(c), 4),   # semi-annual amplitude
    "psi":                 round(psi, 4),       # semi-annual phase
    "peak_doy":            peak_doy,
    "trough_doy":          trough_doy,
    "summer_leaf_frac":    round(summer["leaf_fraction_harmonic"].mean(), 4),
    "winter_leaf_frac":    round(winter["leaf_fraction_harmonic"].mean(), 4),
    "annual_leaf_frac":    round(df["leaf_fraction_harmonic"].mean(), 4),
}

pd.DataFrame([results]).to_csv("harmonic_params.csv", index=False)
print(f"\nSaved: harmonic_params.csv")
print(pd.DataFrame([results]).to_string(index=False))

# Save updated timeseries with harmonic components
df.to_csv("agbd_timeseries_harmonic.csv", index=False)
print(f"Saved: agbd_timeseries_harmonic.csv")