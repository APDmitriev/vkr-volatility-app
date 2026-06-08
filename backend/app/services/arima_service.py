from pathlib import Path

import numpy as np
import pandas as pd

from app.services.forecast_future_utils import build_future_preview, infer_future_timestamps
from statsmodels.tsa.arima.model import ARIMA


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


def run_arima_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    p: int = 1,
    d: int = 1,
    q: int = 1,
) -> tuple[pd.DataFrame, dict]:
    file_path = Path(processed_file_path)

    if not file_path.exists():
        raise FileNotFoundError("Файл обработанного датасета не найден")

    df = pd.read_csv(file_path)

    required_columns = {"timestamp", "value"}
    missing_required = required_columns - set(df.columns)
    if missing_required:
        raise ValueError(
            f"В файле отсутствуют обязательные колонки: {', '.join(sorted(missing_required))}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) <= forecast_horizon:
        raise ValueError("Горизонт прогноза слишком велик для текущего ряда")

    train_df = df.iloc[:-forecast_horizon].copy()
    test_df = df.iloc[-forecast_horizon:].copy()

    if len(train_df) < max(10, p + d + q + 1):
        raise ValueError("Недостаточно данных для обучения ARIMA-модели")

    model = ARIMA(train_df["value"], order=(p, d, q))
    fitted_model = model.fit()

    forecast_values = fitted_model.forecast(steps=forecast_horizon)
    forecast_values = pd.Series(forecast_values).reset_index(drop=True)

    result_df = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"].reset_index(drop=True),
            "actual": test_df["value"].reset_index(drop=True),
            "predicted": forecast_values,
        }
    )

    metrics = _calculate_metrics(
        actual=result_df["actual"],
        predicted=result_df["predicted"],
    )

    # Метрики считаются на holdout-сегменте, а пользовательский прогноз
    # строится от конца всего ряда, чтобы не подменять будущее последними
    # уже известными наблюдениями.
    full_model = ARIMA(df["value"], order=(p, d, q))
    full_fitted_model = full_model.fit()
    future_values = pd.Series(full_fitted_model.forecast(steps=forecast_horizon)).reset_index(drop=True)

    summary = {
        "model": "ARIMA",
        "order": [p, d, q],
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "metrics": metrics,
        "future_preview": build_future_preview(
            infer_future_timestamps(df, forecast_horizon),
            future_values.tolist(),
        ),
    }

    return result_df, summary