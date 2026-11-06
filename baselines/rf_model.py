import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("outputs/gedi_valid_all_layers.csv")

print("Columns:", df.columns.tolist())

feature_cols = [
    "canopy",
    "srtm",
    "daymet",
    "nlcd",
]
target_col = "agbd"

data = df[feature_cols + [target_col]].dropna()

X = data[feature_cols].astype(float)
y = data[target_col].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

param_grid = {
    "n_estimators": [100, 200, 400, 800],
    "max_depth": [10, 20, 40, None],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=40,
    scoring="r2",
    cv=4,
    random_state=40,
    n_jobs=-1,
    verbose=1
)
search.fit(X_train, y_train)

print("\nBest Parameters:")
print(search.best_params_)

best_rf = search.best_estimator_
y_pred = best_rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nPerformance on test set:")
print(f"R²   = {r2:.3f}")
print(f"MAE  = {mae:.3f}")
print(f"RMSE = {rmse:.3f}")

importances = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=importances.index, palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, s=10, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual AGBD")
plt.ylabel("Predicted AGBD")
plt.title(f"Random Forest Prediction (R² = {r2:.2f})")
plt.tight_layout()
plt.show()
