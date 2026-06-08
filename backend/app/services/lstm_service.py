from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.forecast_future_utils import build_future_preview, infer_future_timestamps
from app.services.neural_network_utils import (
    append_prediction_to_history,
    build_supervised_matrix,
    calculate_metrics,
    load_processed_dataframe,
)


def run_lstm_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    window_size: int = 7,
    hidden_size: int = 32,
    num_layers: int = 1,
    epochs: int = 60,
    learning_rate: float = 0.01,
    random_state: int = 42,
    include_exogenous: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Для LSTM требуется PyTorch. Установите зависимость: pip install torch"
        ) from exc

    df = load_processed_dataframe(processed_file_path)
    data = build_supervised_matrix(
        df=df,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        include_exogenous=include_exogenous,
    )

    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))

    x_train = data["x_train_seq"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32).reshape(-1, 1)

    feature_mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    feature_std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) if float(y_train.std()) != 0 else 1.0

    x_scaled = (x_train - feature_mean) / feature_std
    y_scaled = (y_train - y_mean) / y_std

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    class _LSTMRegressor(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):  # type: ignore[no-untyped-def]
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = _LSTMRegressor(
        input_size=x_train.shape[-1],
        hidden_size=max(4, int(hidden_size)),
        num_layers=max(1, int(num_layers)),
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    model.train()
    for _ in range(max(5, int(epochs))):
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    history = data["history_matrix"].astype(np.float32).copy()
    predictions: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(int(forecast_horizon)):
            window = history[-int(window_size):]
            x_window = ((window - feature_mean) / feature_std).reshape(1, int(window_size), -1)
            pred_scaled = model(torch.tensor(x_window, dtype=torch.float32)).numpy()[0, 0]
            pred = float(pred_scaled * y_std + y_mean)
            predictions.append(pred)
            history = append_prediction_to_history(history, pred, data["last_exog"].astype(np.float32)).astype(np.float32)

    result_df = pd.DataFrame(
        {
            "timestamp": data["test_df"]["timestamp"],
            "actual": data["actual"],
            "predicted": predictions,
        }
    )
    result_df["error"] = result_df["actual"] - result_df["predicted"]
    metrics = calculate_metrics(result_df["actual"], result_df["predicted"])

    future_history = data["matrix"].astype(np.float32).copy()
    future_last_exog = future_history[-1, 1:].copy() if future_history.shape[1] > 1 else np.asarray([], dtype=np.float32)
    future_predictions: list[float] = []
    with torch.no_grad():
        for _ in range(int(forecast_horizon)):
            window = future_history[-int(window_size):]
            x_window = ((window - feature_mean) / feature_std).reshape(1, int(window_size), -1)
            pred_scaled = model(torch.tensor(x_window, dtype=torch.float32)).numpy()[0, 0]
            pred = float(pred_scaled * y_std + y_mean)
            future_predictions.append(pred)
            future_history = append_prediction_to_history(
                future_history,
                pred,
                future_last_exog.astype(np.float32),
            ).astype(np.float32)

    summary = {
        "model": "LSTM Neural Network",
        "window_size": int(window_size),
        "hidden_size": max(4, int(hidden_size)),
        "num_layers": max(1, int(num_layers)),
        "epochs": max(5, int(epochs)),
        "learning_rate": float(learning_rate),
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
