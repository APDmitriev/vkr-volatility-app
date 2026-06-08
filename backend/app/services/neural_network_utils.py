from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def calculate_metrics(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> dict[str, float | None]:
    actual_arr = pd.to_numeric(pd.Series(actual), errors="coerce")
    pred_arr = pd.to_numeric(pd.Series(predicted), errors="coerce")
    mask = actual_arr.notna() & pred_arr.notna()
    actual_arr = actual_arr[mask].astype(float)
    pred_arr = pred_arr[mask].astype(float)
    if actual_arr.empty:
        raise ValueError("Недостаточно данных для расчёта метрик")

    errors = actual_arr.to_numpy() - pred_arr.to_numpy()
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(np.square(errors)))
    rmse = float(np.sqrt(mse))
    non_zero_mask = actual_arr.to_numpy() != 0
    if np.any(non_zero_mask):
        mape = float(np.mean(np.abs(errors[non_zero_mask] / actual_arr.to_numpy()[non_zero_mask])) * 100)
    else:
        mape = None
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}


def load_processed_dataframe(processed_file_path: str) -> pd.DataFrame:
    path = Path(processed_file_path)
    if not path.exists():
        raise FileNotFoundError("Файл обработанного датасета не найден")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Файл обработанного датасета пуст")
    required = {"timestamp", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В файле отсутствуют обязательные колонки: {', '.join(sorted(missing))}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) < 12:
        raise ValueError("Недостаточно наблюдений для нейросетевой модели")
    return df


def select_feature_columns(df: pd.DataFrame, include_exogenous: bool = True) -> list[str]:
    columns = ["value"]
    if include_exogenous:
        for column in df.columns:
            if column in {"timestamp", "value"}:
                continue
            converted = pd.to_numeric(df[column], errors="coerce")
            usable_count = int(converted.notna().sum())
            if usable_count >= max(5, int(len(df) * 0.5)):
                df[column] = converted
                columns.append(str(column))
    return columns


def build_supervised_matrix(
    df: pd.DataFrame,
    window_size: int,
    forecast_horizon: int,
    include_exogenous: bool = True,
) -> dict[str, Any]:
    window_size = max(1, int(window_size))
    forecast_horizon = max(1, int(forecast_horizon))
    if forecast_horizon >= len(df):
        raise ValueError("Горизонт прогноза слишком велик для текущего ряда")
    if len(df) < window_size + forecast_horizon + 5:
        raise ValueError("Недостаточно данных для выбранного окна и горизонта прогноза")

    work_df = df.copy()
    feature_columns = select_feature_columns(work_df, include_exogenous=include_exogenous)
    feature_df = work_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.ffill().bfill().dropna()
    if len(feature_df) < window_size + forecast_horizon + 5:
        raise ValueError("После обработки признаков осталось недостаточно наблюдений")

    # Синхронизируем timestamps/value после удаления NaN из признаков.
    idx = feature_df.index
    aligned_df = work_df.loc[idx].reset_index(drop=True)
    feature_df = feature_df.reset_index(drop=True)
    matrix = feature_df.to_numpy(dtype=float)
    values = pd.to_numeric(aligned_df["value"], errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_datetime(aligned_df["timestamp"], errors="coerce")

    train_end = len(aligned_df) - forecast_horizon
    if train_end <= window_size + 3:
        raise ValueError("Недостаточно обучающих примеров")

    x_rows = []
    y_rows = []
    for i in range(window_size, train_end):
        x_rows.append(matrix[i - window_size:i])
        y_rows.append(values[i])

    x_train_seq = np.asarray(x_rows, dtype=float)
    y_train = np.asarray(y_rows, dtype=float)
    x_train_flat = x_train_seq.reshape((x_train_seq.shape[0], -1))

    test_df = aligned_df.iloc[train_end:].copy().reset_index(drop=True)
    actual = values[train_end:]
    history_matrix = matrix[:train_end].copy()
    last_exog = history_matrix[-1, 1:].copy() if history_matrix.shape[1] > 1 else np.asarray([], dtype=float)

    return {
        "aligned_df": aligned_df,
        "feature_columns": feature_columns,
        "matrix": matrix,
        "values": values,
        "timestamps": timestamps,
        "train_end": int(train_end),
        "x_train_seq": x_train_seq,
        "x_train_flat": x_train_flat,
        "y_train": y_train,
        "test_df": test_df,
        "actual": actual,
        "history_matrix": history_matrix,
        "last_exog": last_exog,
    }


def append_prediction_to_history(history_matrix: np.ndarray, prediction: float, last_exog: np.ndarray) -> np.ndarray:
    if history_matrix.shape[1] > 1:
        next_row = np.concatenate([[float(prediction)], last_exog.astype(float)])
    else:
        next_row = np.asarray([float(prediction)], dtype=float)
    return np.vstack([history_matrix, next_row])
