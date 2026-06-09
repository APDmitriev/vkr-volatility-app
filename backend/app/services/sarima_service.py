from pathlib import Path

import numpy as np
import pandas as pd

from app.services.forecast_future_utils import build_future_preview, infer_future_timestamps
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")

    valid_mask = actual.notna() & predicted.notna()
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]

    if actual.empty:
        raise ValueError("Недостаточно данных для расчёта метрик")

    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    non_zero_mask = actual != 0
    if non_zero_mask.any():
        mape = float(
            np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
        )
    else:
        mape = None

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


def run_sarima_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    p: int = 1,
    d: int = 1,
    q: int = 1,
    seasonal_p: int = 1,
    seasonal_d: int = 0,
    seasonal_q: int = 1,
    seasonal_period: int = 7,
) -> tuple[pd.DataFrame, dict]:
    file_path = Path(processed_file_path)
    if not file_path.exists():
        raise FileNotFoundError("Файл обработанного датасета не найден")

    if forecast_horizon < 1:
        raise ValueError("Горизонт прогноза должен быть не меньше 1")
    if seasonal_period < 2:
        raise ValueError("Длина сезона s должна быть не меньше 2")

    df = pd.read_csv(file_path)
    required_columns = {"timestamp", "value"}
    missing_required = required_columns - set(df.columns)
    if missing_required:
        raise ValueError(f"В файле отсутствуют обязательные колонки: {', '.join(sorted(missing_required))}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) <= forecast_horizon:
        raise ValueError("Горизонт прогноза слишком велик для текущего ряда")

    train_df = df.iloc[:-forecast_horizon].copy()
    test_df = df.iloc[-forecast_horizon:].copy().reset_index(drop=True)

    min_required = max(
        20,
        p + d + q + seasonal_p * seasonal_period + seasonal_d * seasonal_period + seasonal_q * seasonal_period + 2,
        seasonal_period * 2,
    )
    if len(train_df) < min_required:
        raise ValueError(
            "Недостаточно данных для обучения SARIMA-модели с выбранной сезонностью. "
            f"Нужно минимум {min_required} наблюдений в обучающей части."
        )

    model = SARIMAX(
        train_df["value"],
        order=(p, d, q),
        seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = model.fit(disp=False)

    forecast_values = fitted_model.forecast(steps=forecast_horizon)
    forecast_values = pd.Series(forecast_values).reset_index(drop=True)

    result_df = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"],
            "actual": test_df["value"],
            "predicted": forecast_values,
        }
    )

    metrics = _calculate_metrics(result_df["actual"], result_df["predicted"])

    full_model = SARIMAX(
        df["value"],
        order=(p, d, q),
        seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    full_fitted_model = full_model.fit(disp=False)
    future_values = pd.Series(full_fitted_model.forecast(steps=forecast_horizon)).reset_index(drop=True)

    summary = {
        "model": "SARIMA",
        "order": [p, d, q],
        "seasonal_order": [seasonal_p, seasonal_d, seasonal_q, seasonal_period],
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "metrics": metrics,
        "future_preview": build_future_preview(
            infer_future_timestamps(df, forecast_horizon),
            future_values.tolist(),
        ),
    }
    return result_df, summary
