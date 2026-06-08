from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd

from app.services.forecast_future_utils import build_future_preview, infer_future_timestamps


def _safe_float(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def _trapezoid_membership(x: float, a1: float, a2: float, a3: float, a4: float) -> float:
    if x <= a1 or x >= a4:
        return 0.0
    if a2 <= x <= a3:
        return 1.0
    if a1 < x < a2:
        denom = a2 - a1
        return (x - a1) / denom if denom != 0 else 0.0
    if a3 < x < a4:
        denom = a4 - a3
        return (a4 - x) / denom if denom != 0 else 0.0
    return 0.0


def _build_fuzzy_sets(values: np.ndarray, min_sets: int, max_sets: int):
    sorted_values = np.sort(values.astype(float))

    if len(sorted_values) < 2:
        raise ValueError("Для нечёткого прогнозирования нужно минимум 2 значения ряда")

    distances = np.abs(np.diff(sorted_values))
    positive_distances = distances[distances > 0]

    if len(positive_distances) == 0:
        raise ValueError("Ряд содержит одинаковые значения, метод не применим")

    avg_distance_raw = float(np.mean(positive_distances))
    sigma_distance = float(np.std(positive_distances))

    lower = max(0.0, avg_distance_raw - sigma_distance)
    upper = avg_distance_raw + sigma_distance

    filtered = positive_distances[(positive_distances >= lower) & (positive_distances <= upper)]
    avg_distance = float(np.mean(filtered)) if len(filtered) > 0 else avg_distance_raw

    d_min = float(np.min(values))
    d_max = float(np.max(values))

    universe_min = d_min - avg_distance
    universe_max = d_max + avg_distance

    auto_sets = int(round((universe_max - universe_min - avg_distance) / (2 * avg_distance)))
    fuzzy_sets_count = max(min_sets, min(max_sets, auto_sets if auto_sets > 0 else min_sets))

    delta = (universe_max - universe_min) / (2 * fuzzy_sets_count + 1)

    fuzzy_sets = []
    for i in range(fuzzy_sets_count):
        a1 = universe_min + (2 * i) * delta
        a2 = a1 + delta
        a3 = a2 + delta
        a4 = a3 + delta
        midpoint = (a2 + a3) / 2

        fuzzy_sets.append(
            {
                "label": f"A{i + 1}",
                "a1": float(a1),
                "a2": float(a2),
                "a3": float(a3),
                "a4": float(a4),
                "midpoint": float(midpoint),
            }
        )

    return {
        "fuzzy_sets": fuzzy_sets,
        "avg_distance": avg_distance,
        "sigma_distance": sigma_distance,
        "universe_min": float(universe_min),
        "universe_max": float(universe_max),
        "fuzzy_sets_count": fuzzy_sets_count,
    }


def _fuzzify_value(x: float, fuzzy_sets: list[dict]) -> dict:
    best_set = None
    best_mu = -1.0

    for fuzzy_set in fuzzy_sets:
        mu = _trapezoid_membership(
            x,
            fuzzy_set["a1"],
            fuzzy_set["a2"],
            fuzzy_set["a3"],
            fuzzy_set["a4"],
        )
        if mu > best_mu:
            best_mu = mu
            best_set = fuzzy_set

    if best_set is None:
        raise ValueError("Не удалось фаззифицировать значение")

    return best_set


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

    non_zero_mask = actual != 0
    if non_zero_mask.any():
        mape = float(
            np.mean(
                np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])
            ) * 100
        )
    else:
        mape = None

    return {
        "mae": mae,
        "mse": mse,
        "mape": mape,
    }


def run_fuzzy_first_order_forecast(
    processed_file_path: str,
    min_sets: int = 7,
    max_sets: int = 30,
    forecast_horizon: int = 1,
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

    if len(df) < 3:
        raise ValueError("Для метода нужно минимум 3 точки ряда")

    values = df["value"].to_numpy(dtype=float)

    fuzzy_info = _build_fuzzy_sets(
        values=values,
        min_sets=min_sets,
        max_sets=max_sets,
    )
    fuzzy_sets = fuzzy_info["fuzzy_sets"]

    fuzzified_labels = []
    fuzzified_midpoints = []

    for value in values:
        fuzzy_set = _fuzzify_value(value, fuzzy_sets)
        fuzzified_labels.append(fuzzy_set["label"])
        fuzzified_midpoints.append(fuzzy_set["midpoint"])

    transition_groups: dict[str, OrderedDict[str, float]] = {}

    for i in range(len(fuzzified_labels) - 1):
        current_label = fuzzified_labels[i]
        next_label = fuzzified_labels[i + 1]
        next_midpoint = fuzzified_midpoints[i + 1]

        if current_label not in transition_groups:
            transition_groups[current_label] = OrderedDict()

        if next_label not in transition_groups[current_label]:
            transition_groups[current_label][next_label] = next_midpoint

    rows = []

    for i in range(len(df) - 1):
        current_label = fuzzified_labels[i]
        next_actual = float(values[i + 1])

        consequents = transition_groups.get(current_label, OrderedDict())
        if len(consequents) == 0:
            predicted_value = fuzzified_midpoints[i]
            forecast_labels = current_label
        else:
            predicted_value = float(np.mean(list(consequents.values())))
            forecast_labels = ", ".join(consequents.keys())

        rows.append(
            {
                "source_timestamp": df.loc[i, "timestamp"],
                "target_timestamp": df.loc[i + 1, "timestamp"],
                "actual": next_actual,
                "fuzzy_state": current_label,
                "forecast_labels": forecast_labels,
                "predicted": predicted_value,
            }
        )

    result_df = pd.DataFrame(rows)

    metrics = _calculate_metrics(
        actual=result_df["actual"],
        predicted=result_df["predicted"],
    )

    future_predictions = []
    current_label = fuzzified_labels[-1]
    current_midpoint = fuzzified_midpoints[-1]
    for _ in range(max(1, int(forecast_horizon))):
        group = transition_groups.get(current_label, OrderedDict())
        if len(group) == 0:
            predicted_value = float(current_midpoint)
        else:
            predicted_value = float(np.mean(list(group.values())))
        future_predictions.append(predicted_value)
        next_set = _fuzzify_value(predicted_value, fuzzy_sets)
        current_label = next_set["label"]
        current_midpoint = next_set["midpoint"]

    next_forecast = future_predictions[0]

    summary = {
        "model": "FuzzyTimeSeriesFirstOrder",
        "fuzzy_sets_count": fuzzy_info["fuzzy_sets_count"],
        "groups_count": len(transition_groups),
        "universe_min": fuzzy_info["universe_min"],
        "universe_max": fuzzy_info["universe_max"],
        "avg_distance": fuzzy_info["avg_distance"],
        "sigma_distance": fuzzy_info["sigma_distance"],
        "next_forecast": next_forecast,
        "metrics": metrics,
        "future_preview": build_future_preview(
            infer_future_timestamps(df, max(1, int(forecast_horizon))),
            future_predictions,
        ),
    }

    return result_df, summary