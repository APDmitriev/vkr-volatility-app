from pathlib import Path
import math

import numpy as np
import pandas as pd


def calculate_rolling_volatility(
    processed_file_path: str,
    window_size: int,
    annualize: bool = False,
    periods_per_year: int = 252,
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

    if "returns" not in df.columns:
        df["returns"] = pd.to_numeric(df["value"], errors="coerce").pct_change()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["returns"] = pd.to_numeric(df["returns"], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["volatility"] = df["returns"].rolling(window=window_size).std()

    if annualize:
        df["volatility"] = df["volatility"] * math.sqrt(periods_per_year)

    df["volatility"] = df["volatility"].replace([np.inf, -np.inf], np.nan)

    volatility_series = df["volatility"].dropna()

    summary = {
        "rows_count": int(len(df)),
        "returns_count": int(df["returns"].notna().sum()),
        "volatility_count": int(volatility_series.shape[0]),
        "min_volatility": (
            float(volatility_series.min()) if not volatility_series.empty else None
        ),
        "max_volatility": (
            float(volatility_series.max()) if not volatility_series.empty else None
        ),
        "mean_volatility": (
            float(volatility_series.mean()) if not volatility_series.empty else None
        ),
    }

    return df, summary