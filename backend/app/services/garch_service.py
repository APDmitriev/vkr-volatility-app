from pathlib import Path
import math

import numpy as np
import pandas as pd

from app.services.forecast_future_utils import build_future_preview, infer_future_timestamps
from arch import arch_model


def run_garch_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    p: int = 1,
    q: int = 1,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> tuple[pd.DataFrame, dict]:
    file_path = Path(processed_file_path)

    if not file_path.exists():
        raise FileNotFoundError("Файл обработанного датасета не найден")

    df = pd.read_csv(file_path)

    required_columns = {"timestamp", "returns"}
    missing_required = required_columns - set(df.columns)
    if missing_required:
        raise ValueError(
            f"В файле отсутствуют обязательные колонки: {', '.join(sorted(missing_required))}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["returns"] = pd.to_numeric(df["returns"], errors="coerce")

    df = df.dropna(subset=["timestamp", "returns"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) <= forecast_horizon:
        raise ValueError("Горизонт прогноза слишком велик для текущего ряда")

    train_df = df.iloc[:-forecast_horizon].copy()
    test_df = df.iloc[-forecast_horizon:].copy()

    if len(train_df) < max(30, p + q + 5):
        raise ValueError("Недостаточно данных для обучения GARCH-модели")


    train_returns = train_df["returns"] * 100.0

    model = arch_model(
        train_returns,
        vol="GARCH",
        p=p,
        q=q,
        mean="Zero",
        rescale=False,
    )
    fitted_model = model.fit(disp="off")

    forecast = fitted_model.forecast(horizon=forecast_horizon)
    variance_values = forecast.variance.iloc[-1].values
    volatility_values = np.sqrt(variance_values) / 100.0

    if annualize:
        volatility_values = volatility_values * math.sqrt(periods_per_year)

    result_df = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"].reset_index(drop=True),
            "actual_returns": test_df["returns"].reset_index(drop=True),
            "forecast_volatility": volatility_values,
        }
    )

    full_returns = df["returns"] * 100.0
    full_model = arch_model(
        full_returns,
        vol="GARCH",
        p=p,
        q=q,
        mean="Zero",
        rescale=False,
    )
    full_fitted_model = full_model.fit(disp="off")
    full_forecast = full_fitted_model.forecast(horizon=forecast_horizon)
    future_variance_values = full_forecast.variance.iloc[-1].values
    future_volatility_values = np.sqrt(future_variance_values) / 100.0
    if annualize:
        future_volatility_values = future_volatility_values * math.sqrt(periods_per_year)

    summary = {
        "train_size": int(len(train_df)),
        "forecast_horizon": int(forecast_horizon),
        "min_forecast_volatility": float(np.min(future_volatility_values)) if len(future_volatility_values) else None,
        "max_forecast_volatility": float(np.max(future_volatility_values)) if len(future_volatility_values) else None,
        "mean_forecast_volatility": float(np.mean(future_volatility_values)) if len(future_volatility_values) else None,
        "future_preview": build_future_preview(
            infer_future_timestamps(df, forecast_horizon),
            future_volatility_values.tolist(),
            value_column="forecast_volatility",
        ),
    }

    return result_df, summary