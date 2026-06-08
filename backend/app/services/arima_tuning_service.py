from pathlib import Path

import numpy as np
import pandas as pd
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
            np.mean(
                np.abs(
                    (actual[non_zero_mask] - predicted[non_zero_mask])
                    / actual[non_zero_mask]
                )
            )
            * 100
        )
    else:
        mape = None

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


def tune_arima_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    p_values: list[int],
    d_values: list[int],
    q_values: list[int],
    optimize_by: str = "rmse",
) -> tuple[pd.DataFrame, dict]:
    file_path = Path(processed_file_path)

    if not file_path.exists():
        raise FileNotFoundError("Файл обработанного датасета не найден")

    if optimize_by not in {"mae", "rmse", "mape"}:
        raise ValueError("optimize_by должен быть одним из: mae, rmse, mape")

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

    candidates: list[dict] = []
    best_result = None
    best_forecast_df = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)

                try:
                    if len(train_df) < max(10, p + d + q + 1):
                        continue

                    model = ARIMA(train_df["value"], order=order)
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

                    candidate = {
                        "order": [p, d, q],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "mape": metrics["mape"],
                    }
                    candidates.append(candidate)

                    candidate_score = candidate[optimize_by]
                    if candidate_score is None:
                        continue

                    if best_result is None or candidate_score < best_result[optimize_by]:
                        best_result = candidate
                        best_forecast_df = result_df

                except Exception:
                    continue

    if best_result is None or best_forecast_df is None:
        raise ValueError("Не удалось подобрать рабочую конфигурацию ARIMA")

    candidates_sorted = sorted(
        candidates,
        key=lambda x: float("inf") if x[optimize_by] is None else x[optimize_by],
    )

    summary = {
        "best_model": "ARIMA",
        "best_order": best_result["order"],
        "optimize_by": optimize_by,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "metrics": {
            "mae": best_result["mae"],
            "rmse": best_result["rmse"],
            "mape": best_result["mape"],
        },
        "candidates": candidates_sorted,
    }

    return best_forecast_df, summary