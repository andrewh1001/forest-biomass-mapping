from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch
from dataset import MultiBandAGBDDataset
from train import train_model
from network import AGBDNet_DualHead, S2EncoderFCN_ResFiLM
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from helper import get_invalid_img
from time import time
import datetime
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

season      = "all"
name        = "21x21_new"
patch_dir_s2  = f"data/s2_np/{name}/{season}/np_s2"
patch_dir_scl = f"data/s2_np/{name}/{season}/np_scl"
image_path    = f"data/s2_np/{name}/{season}/img"
csv_path      = f"data/outputs/gedi_valid_all_layers_from_h5_6_{season}.csv"
alpha_csv     = f"data/outputs/alphaearth_vec_2024_{season}.csv"

NLCD_PATH = "nlcd_4326_mi.tif"


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(batch_size=128, num_workers=0, seed=42, test=True):
    invalid_img_ids = np.load("invalid_img_ids.npy")
    invalid_idx     = list(map(int, invalid_img_ids))

    alpha_df = pd.read_csv(alpha_csv)
    df       = pd.read_csv(csv_path)
    df       = df.drop(invalid_idx).reset_index(drop=True)

    target_col = "agbd"

    # Build un-normalised dataset first (to compute alpha stats from train split)
    dataset_full = MultiBandAGBDDataset(
        df=df,
        alpha_df=alpha_df,
        patch_dir_s2=patch_dir_s2,
        patch_dir_scl=patch_dir_scl,
        target_col=target_col,
        alpha_mean=None,
        alpha_std=None,
        scale_to_reflectance=False,
        nlcd_path=NLCD_PATH if os.path.exists(NLCD_PATH) else None,
    )

    n = len(dataset_full)
    print("Total samples:", n)

    if not test:
        n_train = int(0.8 * n)
        n_val   = n - n_train
        n_test  = 0
    else:
        n_train = int(0.7 * n)
        n_val   = int(0.7 * (n - n_train))
        n_test  = n - n_train - n_val

    print(f"Train: {n_train}  Val: {n_val}  Test: {n_test}")

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset_full,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    # Compute normalisation stats from train split only
    train_ids   = set(df.iloc[train_ds.indices]["id"].values)
    alpha_train = alpha_df[alpha_df["id"].isin(train_ids)].iloc[:, 1:-2].values.astype("float32")
    alpha_mean  = alpha_train.mean(axis=0)
    alpha_std   = np.maximum(alpha_train.std(axis=0), 1e-6)

    # Re-build with normalisation applied
    dataset_full_norm = MultiBandAGBDDataset(
        df=df,
        alpha_df=alpha_df,
        patch_dir_s2=patch_dir_s2,
        patch_dir_scl=patch_dir_scl,
        target_col=target_col,
        alpha_mean=alpha_mean,
        alpha_std=alpha_std,
        scale_to_reflectance=False,
        nlcd_path=NLCD_PATH if os.path.exists(NLCD_PATH) else None,
    )

    train_ds = Subset(dataset_full_norm, train_ds.indices)
    val_ds   = Subset(dataset_full_norm, val_ds.indices)
    test_ds  = Subset(dataset_full_norm, test_ds.indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, df, alpha_mean, alpha_std


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(targets, preds, leaf_preds=None, wood_preds=None):
    targets = np.array(targets)
    preds   = np.array(preds)
    errors  = preds - targets

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # True vs Predicted
    ax = axes[0]
    ax.scatter(targets, preds, s=8, alpha=0.4)
    mv = max(targets.max(), preds.max())
    ax.plot([0, mv], [0, mv], "r--")
    ax.set_xlabel("True AGBD (Mg/ha)")
    ax.set_ylabel("Predicted AGBD (Mg/ha)")
    ax.set_title("Total AGBD: True vs Predicted")
    ax.grid(True)

    # Residuals
    ax = axes[1]
    ax.scatter(targets, errors, s=8, alpha=0.4)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("True AGBD")
    ax.set_ylabel("Residual (pred − true)")
    ax.set_title("Residual Plot")
    ax.grid(True)

    # Error histogram
    ax = axes[2]
    ax.hist(errors, bins=40, alpha=0.7)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Error (Mg/ha)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Error Distribution")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("results_total.png", dpi=150)
    plt.show()

    # Optional leaf / wood breakdown
    if leaf_preds is not None and wood_preds is not None:
        leaf_preds = np.array(leaf_preds)
        wood_preds = np.array(wood_preds)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        ax = axes[0]
        ax.hist(leaf_preds / (preds + 1e-6), bins=40, alpha=0.7, label="predicted")
        ax.set_xlabel("Leaf fraction")
        ax.set_ylabel("Count")
        ax.set_title("Predicted Leaf Fraction Distribution")
        ax.grid(True)

        ax = axes[1]
        ax.scatter(leaf_preds, wood_preds, s=6, alpha=0.3)
        ax.set_xlabel("Leaf AGBD (Mg/ha)")
        ax.set_ylabel("Wood AGBD (Mg/ha)")
        ax.set_title("Leaf vs Wood AGBD")
        ax.grid(True)

        plt.tight_layout()
        plt.savefig("results_leaf_wood.png", dpi=150)
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    eval_only: bool = False,
    model_path: str = None,
    epochs: int = 50,
    seed: int = 42,
    dual_head: bool = True,     # NEW flag — set False to revert to original model
    test: bool = True,
):
    start_time = time()
    train_loader, val_loader, test_loader, df, alpha_mean, alpha_std = \
        build_dataloaders(seed=seed, test=test)

    # Save normalisation stats alongside the weights
    np.savez(
        f"{model_path.removesuffix('.pt')}.npz",
        alpha_mean=np.float32(alpha_mean),
        alpha_std=np.float32(alpha_std),
        dual_head=np.bool_(dual_head),
    )

    # ---- Build model ----
    encoder = S2EncoderFCN_ResFiLM(
        in_channels=10,
        embed_dim=128,
        ae_dim=66,
    )

    if dual_head:
        model = AGBDNet_DualHead(
            s2_encoder=encoder,
            embed_dim=128,
            ae_dim=66,
        ).to(device)
        print("Model: AGBDNet_DualHead (leaf + wood → total)")
    else:
        from network import AGBDNet
        model = AGBDNet(
            s2_encoder=encoder,
            embed_dim=128,
        ).to(device)
        print("Model: AGBDNet (original single-head)")

    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6,
    )

    cloud_frac_thresh = 0.2

    # ---- Train ----
    if not eval_only:
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            epochs=epochs,
            device=device,
            scheduler=scheduler,
            cloud_frac_thresh=cloud_frac_thresh,
            dual_head=dual_head,            # NEW
        )

        torch.save(model.state_dict(), model_path)

        with open("training_log.txt", "a") as f:
            now = datetime.datetime.now()
            f.write(f"\nDate: {now}\n")
            f.write(f"Model: {model_path}  dual_head={dual_head}\n")
            f.write("Training / Val Loss:\n")
            for i, (tr, va) in enumerate(zip(train_losses, val_losses)):
                f.write(f"  Epoch {i+1:>3}: train={tr:.5f}  val={va:.5f}\n")

    # ---- Eval ----
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    eps = 3e-5

    all_preds       = []
    all_targets     = []
    all_leaf_preds  = []
    all_wood_preds  = []
    running_mse     = 0.0
    n_samples       = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test "):
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
                out   = model(s2, ae)
                preds = out["total"]                # use total for evaluation
                all_leaf_preds.append(out["leaf"].cpu())
                all_wood_preds.append(out["wood"].cpu())
            else:
                pred_log = model(s2, ae)
                preds    = torch.exp(pred_log) - eps

            mse_batch  = torch.mean((preds - y) ** 2)
            bs         = y.size(0)
            running_mse += mse_batch.item() * bs
            n_samples   += bs

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    mse  = running_mse / n_samples
    rmse = mse ** 0.5

    all_preds   = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    leaf_arr = torch.cat(all_leaf_preds).numpy() if all_leaf_preds else None
    wood_arr = torch.cat(all_wood_preds).numpy() if all_wood_preds else None

    r2 = r2_score(all_targets, all_preds) * 100
    mae = np.mean(np.abs(all_preds - all_targets))

    print(f"\nTotal time : {time() - start_time:.1f}s")
    print(f"MSE        : {mse:.4f}")
    print(f"RMSE       : {rmse:.4f}")
    print(f"MAE        : {mae:.4f}")
    print(f"R²         : {r2:.2f}%")

    if leaf_arr is not None:
        mean_leaf_frac = (leaf_arr / (all_preds + 1e-6)).mean()
        print(f"Mean leaf fraction : {mean_leaf_frac:.4f}  "
              f"(expected ~0.05–0.08 for Michigan)")

    print("\nSample predictions (first 10):")
    for i in range(min(10, len(all_preds))):
        row = f"  True: {all_targets[i]:7.2f}  Pred: {all_preds[i]:7.2f}"
        if leaf_arr is not None:
            row += f"  Leaf: {leaf_arr[i]:6.2f}  Wood: {wood_arr[i]:6.2f}"
        print(row)

    # Baseline constant-mean model
    y_all          = df["agbd"].values.astype("float32")
    baseline_rmse  = ((y_all - y_all.mean()) ** 2).mean() ** 0.5
    print(f"\nBaseline (mean) RMSE: {baseline_rmse:.4f}")

    plot_results(all_targets, all_preds, leaf_arr, wood_arr)

    if not eval_only:
        with open("training_log.txt", "a") as f:
            f.write(f"MSE={mse:.4f}  RMSE={rmse:.4f}  R2={r2:.2f}\n")
            f.write("-" * 60 + "\n")


if __name__ == "__main__":
    epochs     = 50
    seed       = 42
    dual_head  = True   

    model_path = "../weights/dualhead.pt"

    main(
        eval_only=True,
        model_path=model_path,
        epochs=epochs,
        seed=seed,
        dual_head=dual_head,
        test=True,
    )
