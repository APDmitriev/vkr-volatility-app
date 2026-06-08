from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def infer_future_timestamps(df: pd.DataFrame, horizon: int, timestamp_column: str = "timestamp") -> list[pd.Timestamp]:
    """Return timestamps immediately after the last point of a prepared time series."""
    horizon = max(1, int(horizon))
    if timestamp_column not in df.columns:
        return [pd.Timestamp(i) for i in range(horizon)]

    timestamps = pd.to_datetime(df[timestamp_column], errors="coerce").dropna().sort_values().reset_index(drop=True)
    if timestamps.empty:
        start = pd.Timestamp.now().normalize()
        return list(pd.date_range(start=start, periods=horizon, freq="D"))

    last_timestamp = timestamps.iloc[-1]
    deltas = timestamps.diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    step = deltas.median() if not deltas.empty else pd.Timedelta(days=1)
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(days=1)

    return [last_timestamp + step * i for i in range(1, horizon + 1)]


def build_future_preview(timestamps: list[Any], predicted: list[Any], value_column: str = "predicted") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timestamp, value in zip(timestamps, predicted):
        if isinstance(timestamp, pd.Timestamp):
            timestamp_value: Any = timestamp.isoformat()
        else:
            timestamp_value = timestamp
        try:
            predicted_value: Any = float(value)
            if np.isnan(predicted_value) or np.isinf(predicted_value):
                predicted_value = None
        except (TypeError, ValueError):
            predicted_value = value
        rows.append({"timestamp": timestamp_value, value_column: predicted_value})
    return rows
