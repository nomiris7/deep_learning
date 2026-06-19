from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import numpy as np


def get_xgboost_surrogate():
    return XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )


def get_mlp_surrogate():
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )


def evaluate_surrogate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Spearman": spearman_corr}
