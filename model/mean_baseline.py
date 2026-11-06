import pandas as pd
import numpy as np

df = pd.read_csv("data/outputs/gedi_valid_all_layers_from_h5_6_all.csv")
y  = df["agbd"].dropna().values.astype("float32")

baseline_rmse = ((y - y.mean()) ** 2).mean() ** 0.5
baseline_mae  = np.mean(np.abs(y - y.mean()))
baseline_mse  = ((y - y.mean()) ** 2).mean()

print(f"N samples:      {len(y):,}")
print(f"Mean AGBD:      {y.mean():.2f} Mg/ha")
print(f"Std AGBD:       {y.std():.2f} Mg/ha")
print(f"Min AGBD:       {y.min():.2f} Mg/ha")
print(f"Max AGBD:       {y.max():.2f} Mg/ha")
print(f"")
print(f"Baseline MSE:   {baseline_mse:.4f}")
print(f"Baseline RMSE:  {baseline_rmse:.4f} Mg/ha")
print(f"Baseline MAE:   {baseline_mae:.4f} Mg/ha")
print(f"")
print(f"--- Improvement over baseline ---")
model_rmse = 37.2112
model_mae  = 18.7158
print(f"RMSE: {baseline_rmse:.2f} → {model_rmse:.2f}  "
      f"({(1 - model_rmse/baseline_rmse)*100:.1f}% improvement)")
print(f"MAE:  {baseline_mae:.2f} → {model_mae:.2f}  "
      f"({(1 - model_mae/baseline_mae)*100:.1f}% improvement)")