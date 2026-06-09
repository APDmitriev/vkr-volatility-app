from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from app.services.forecast_future_utils import build_future_preview, infer_future_timestamps
from app.services.neural_network_utils import (
    append_prediction_to_history,
    build_supervised_matrix,
    calculate_metrics,
    load_processed_dataframe,
)


def _hidden_layers(layer_1: int, layer_2: int = 0) -> tuple[int, ...]:
    layers = []
    if int(layer_1) > 0:
        layers.append(int(layer_1))
    if int(layer_2) > 0:
        layers.append(int(layer_2))
    if not layers:
        layers = [32]
    return tuple(layers)


def run_mlp_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    window_size: int = 7,
    hidden_layer_1: int = 64,
    hidden_layer_2: int = 32,
    max_iter: int = 500,
    random_state: int = 42,
    include_exogenous: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = load_processed_dataframe(processed_file_path)
    data = build_supervised_matrix(
        df=df,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        include_exogenous=include_exogenous,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(data["x_train_flat"])
    y_train = y_scaler.fit_transform(data["y_train"].reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=_hidden_layers(hidden_layer_1, hidden_layer_2),
        activation="relu",
        solver="adam",
        max_iter=max(100, int(max_iter)),
        random_state=int(random_state),
        early_stopping=True,
        n_iter_no_change=20,
    )
    model.fit(x_train, y_train)

    history = data["history_matrix"].copy()
    predictions: list[float] = []
    for _ in range(int(forecast_horizon)):
        x = history[-int(window_size):].reshape(1, -1)
        pred_scaled = float(model.predict(x_scaler.transform(x))[0])
        pred = float(y_scaler.inverse_transform([[pred_scaled]])[0, 0])
        predictions.append(pred)
        history = append_prediction_to_history(history, pred, data["last_exog"])

    result_df = pd.DataFrame(
        {
            "timestamp": data["test_df"]["timestamp"],
            "actual": data["actual"],
            "predicted": predictions,
        }
    )
    result_df["error"] = result_df["actual"] - result_df["predicted"]
    metrics = calculate_metrics(result_df["actual"], result_df["predicted"])



    future_history = data["matrix"].copy()
    future_last_exog = future_history[-1, 1:].copy() if future_history.shape[1] > 1 else np.asarray([], dtype=float)
    future_predictions: list[float] = []
    for _ in range(int(forecast_horizon)):
        x = future_history[-int(window_size):].reshape(1, -1)
        pred_scaled = float(model.predict(x_scaler.transform(x))[0])
        pred = float(y_scaler.inverse_transform([[pred_scaled]])[0, 0])
        future_predictions.append(pred)
        future_history = append_prediction_to_history(future_history, pred, future_last_exog)

    summary = {
        "model": "MLP Neural Network",
        "window_size": int(window_size),
        "hidden_layers": list(_hidden_layers(hidden_layer_1, hidden_layer_2)),
        "max_iter": max(100, int(max_iter)),
        "random_state": int(random_state),
        "include_exogenous": bool(include_exogenous),
        "feature_columns": data["feature_columns"],
        "train_size": int(data["train_end"]),
        "test_size": int(len(result_df)),
        "metrics": metrics,
        "future_preview": build_future_preview(
            infer_future_timestamps(data["aligned_df"], int(forecast_horizon)),
            future_predictions,
        ),
    }
    return result_df, summary
