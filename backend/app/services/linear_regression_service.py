from pathlib import Path

import numpy as np
import pandas as pd

from app.services.forecast_future_utils import build_future_preview, infer_future_timestamps


def _calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")

    mask = actual.notna() & predicted.notna()
    actual = actual[mask]
    predicted = predicted[mask]

    if actual.empty:
        raise ValueError("Недостаточно данных для расчёта метрик")

    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(np.square(errors)))
    rmse = float(np.sqrt(mse))

    non_zero_mask = actual != 0
    if non_zero_mask.any():
        mape = float(np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100)
    else:
        mape = None

    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}


def _make_lag_matrix(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for i in range(window_size, len(values)):
        x_rows.append(values[i - window_size:i])
        y_rows.append(values[i])
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)


def run_linear_regression_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    window_size: int = 5,
) -> tuple[pd.DataFrame, dict]:
    file_path = Path(processed_file_path)
    if not file_path.exists():
        raise FileNotFoundError("Файл обработанного датасета не найден")

    df = pd.read_csv(file_path)
    required_columns = {"timestamp", "value"}
    missing_required = required_columns - set(df.columns)
    if missing_required:
        raise ValueError(f"В файле отсутствуют обязательные колонки: {', '.join(sorted(missing_required))}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)

    if forecast_horizon >= len(df):
        raise ValueError("Горизонт прогноза слишком велик для текущего ряда")

    window_size = max(1, int(window_size))
    if len(df) < window_size + forecast_horizon + 5:
        raise ValueError("Недостаточно данных для линейной регрессии с выбранным окном")

    values = df["value"].to_numpy(dtype=float)
    train_values = values[:-forecast_horizon]
    test_df = df.iloc[-forecast_horizon:].copy().reset_index(drop=True)

    x_train, y_train = _make_lag_matrix(train_values, window_size)
    if len(y_train) < 3:
        raise ValueError("Недостаточно обучающих примеров для линейной регрессии")

    # Добавляем свободный член и решаем задачу МНК без внешних зависимостей.
    x_design = np.column_stack([np.ones(len(x_train)), x_train])
    coef, *_ = np.linalg.lstsq(x_design, y_train, rcond=None)

    history = list(train_values[-window_size:])
    predictions = []
    for _ in range(forecast_horizon):
        x = np.asarray([1.0] + history[-window_size:], dtype=float)
        pred = float(x @ coef)
        predictions.append(pred)
        history.append(pred)

    result_df = pd.DataFrame({
        "timestamp": test_df["timestamp"],
        "actual": test_df["value"],
        "predicted": predictions,
    })

    metrics = _calculate_metrics(result_df["actual"], result_df["predicted"])

    # Итоговый прогноз строим отдельно: обучаем модель на всём доступном ряде
    # и продолжаем значения после последней фактической даты. Backtest выше
    # нужен только для расчёта метрик качества.
    full_x_train, full_y_train = _make_lag_matrix(values, window_size)
    if len(full_y_train) < 3:
        raise ValueError("Недостаточно обучающих примеров для итогового прогноза")
    full_x_design = np.column_stack([np.ones(len(full_x_train)), full_x_train])
    full_coef, *_ = np.linalg.lstsq(full_x_design, full_y_train, rcond=None)
    future_history = list(values[-window_size:])
    future_predictions = []
    for _ in range(forecast_horizon):
        x = np.asarray([1.0] + future_history[-window_size:], dtype=float)
        pred = float(x @ full_coef)
        future_predictions.append(pred)
        future_history.append(pred)

    summary = {
        "model": "Linear Regression",
        "window_size": window_size,
        "train_size": int(len(train_values)),
        "test_size": int(len(test_df)),
        "metrics": metrics,
        "future_preview": build_future_preview(
            infer_future_timestamps(df, forecast_horizon),
            future_predictions,
        ),
    }
    return result_df, summary
