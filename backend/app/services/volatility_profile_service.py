from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_NAMES = [
    "Linear Regression",
    "ARIMA",
    "SARIMA",
    "GARCH",
    "Fuzzy First Order",
    "MLP Neural Network",
    "LSTM Neural Network",
]


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _autocorr(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(values) <= lag + 2:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    corr = np.corrcoef(left, right)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _detect_value_column(df: pd.DataFrame) -> str:
    if "value" in df.columns:
        return "value"
    numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_columns:
        return str(numeric_columns[0])
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() >= max(3, int(len(df) * 0.5)):
            return str(column)
    raise ValueError("Не удалось определить числовой столбец временного ряда")


def _prepare_values(processed_file_path: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    path = Path(processed_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл обработанного датасета не найден: {processed_file_path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Файл обработанного датасета пуст")

    value_column = _detect_value_column(df)
    values = pd.to_numeric(df[value_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 8:
        raise ValueError("Недостаточно наблюдений для оценки профиля волатильности")

    if "returns" in df.columns:
        returns = pd.to_numeric(df["returns"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    else:
        aligned_values = values.reset_index(drop=True)
        returns = aligned_values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) < 5:
        returns = values.diff().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 5:
        raise ValueError("Недостаточно наблюдений для расчёта доходностей и волатильности")

    return df, values.reset_index(drop=True), returns.reset_index(drop=True)


def calculate_volatility_profile(
    processed_file_path: str,
    window_size: int = 7,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    df, values, returns = _prepare_values(processed_file_path)

    y = values.to_numpy(dtype=float)
    r = returns.to_numpy(dtype=float)
    abs_r = np.abs(r)

    rel_volatility = float(np.std(y) / np.mean(np.abs(y))) if np.mean(np.abs(y)) != 0 else 0.0
    std_returns = float(np.std(r))
    mean_abs_return = float(np.mean(abs_r))

    effective_window = max(2, min(int(window_size), len(returns)))
    rolling_vol = returns.rolling(window=effective_window, min_periods=2).std().dropna()
    mean_rolling_volatility = _safe_float(rolling_vol.mean()) or 0.0
    max_rolling_volatility = _safe_float(rolling_vol.max()) or 0.0
    volatility_cv = float(rolling_vol.std() / rolling_vol.mean()) if len(rolling_vol) > 2 and rolling_vol.mean() not in (0, 0.0) else 0.0

    median_abs = float(np.median(abs_r)) if len(abs_r) else 0.0
    mad = float(np.median(np.abs(abs_r - median_abs))) if len(abs_r) else 0.0
    robust_threshold = median_abs + 3.0 * (mad if mad > 0 else std_returns)
    std_threshold = 3.0 * std_returns
    spike_threshold = max(robust_threshold, std_threshold, 1e-12)
    spike_ratio = float(np.mean(abs_r > spike_threshold)) if len(abs_r) else 0.0

    cumulative_max = np.maximum.accumulate(y)
    drawdowns = np.where(cumulative_max != 0, (y - cumulative_max) / cumulative_max, 0.0)
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) else 0.0

    acf_lag_1 = _autocorr(y, 1)
    autocorrelation_detected = abs(acf_lag_1) >= 0.25

    abs_returns_acf_1 = _autocorr(abs_r, 1)
    abs_returns_acf_5 = _autocorr(abs_r, 5)
    volatility_clustering_detected = bool(
        abs_returns_acf_1 >= 0.20
        or abs_returns_acf_5 >= 0.20
        or (volatility_cv >= 0.65 and mean_rolling_volatility > 0)
    )

    candidate_periods = [7, 12, 24, 30, 52, 90, 365]
    best_period = None
    best_period_corr = 0.0
    for period in candidate_periods:
        if len(y) >= period * 3:
            corr = abs(_autocorr(y, period))
            if corr > best_period_corr:
                best_period_corr = corr
                best_period = period
    seasonality_detected = bool(best_period is not None and best_period_corr >= 0.35)

    if std_returns >= 0.05 or spike_ratio >= 0.08 or rel_volatility >= 0.35:
        volatility_level = "high"
        volatility_level_ru = "высокая"
    elif std_returns >= 0.02 or spike_ratio >= 0.03 or rel_volatility >= 0.12:
        volatility_level = "medium"
        volatility_level_ru = "средняя"
    else:
        volatility_level = "low"
        volatility_level_ru = "низкая"

    extra_numeric_features = [
        c for c in df.columns
        if c not in {"timestamp", "value"} and pd.to_numeric(df[c], errors="coerce").notna().sum() >= max(5, int(len(df) * 0.5))
    ]

    scores = {
        "Linear Regression": 0.38,
        "ARIMA": 0.45,
        "SARIMA": 0.40,
        "GARCH": 0.35,
        "Fuzzy First Order": 0.42,
        "MLP Neural Network": 0.36,
        "LSTM Neural Network": 0.34,
    }

    if volatility_level == "low":
        scores["Linear Regression"] += 0.25
        scores["ARIMA"] += 0.10
        scores["MLP Neural Network"] += 0.08
    elif volatility_level == "medium":
        scores["ARIMA"] += 0.18
        scores["SARIMA"] += 0.08
        scores["MLP Neural Network"] += 0.12
    else:
        scores["Fuzzy First Order"] += 0.25
        scores["GARCH"] += 0.20
        scores["MLP Neural Network"] += 0.20
        scores["LSTM Neural Network"] += 0.18
        scores["Linear Regression"] -= 0.18

    if autocorrelation_detected:
        scores["ARIMA"] += 0.25
        scores["SARIMA"] += 0.10
    if seasonality_detected:
        scores["SARIMA"] += 0.35
        scores["LSTM Neural Network"] += 0.12
        scores["ARIMA"] -= 0.05
    if volatility_clustering_detected:
        scores["GARCH"] += 0.35
    if spike_ratio >= 0.05:
        scores["Fuzzy First Order"] += 0.25
        scores["MLP Neural Network"] += 0.12
        scores["LSTM Neural Network"] += 0.10
        scores["Linear Regression"] -= 0.12
        scores["ARIMA"] -= 0.05
    if extra_numeric_features:
        scores["MLP Neural Network"] += 0.18
        scores["LSTM Neural Network"] += 0.15
    if std_returns < 0.015 and not seasonality_detected and not volatility_clustering_detected:
        scores["Linear Regression"] += 0.12

    ranked_models = []
    for model in MODEL_NAMES:
        score = max(0.05, min(0.98, scores[model]))
        reasons: list[str] = []
        if model == "Linear Regression":
            reasons.append("подходит как простая базовая модель при спокойной динамике")
            if volatility_level == "high":
                reasons.append("может сглаживать резкие скачки")
        elif model == "ARIMA":
            reasons.append("учитывает зависимость текущих значений от прошлых")
            if not autocorrelation_detected:
                reasons.append("автокорреляция выражена слабо")
        elif model == "SARIMA":
            reasons.append("учитывает сезонную компоненту")
            if not seasonality_detected:
                reasons.append("явная сезонность не обнаружена")
        elif model == "GARCH":
            reasons.append("предназначена для прогнозирования условной волатильности")
            if volatility_clustering_detected:
                reasons.append("обнаружены признаки кластеризации волатильности")
        elif model == "Fuzzy First Order":
            reasons.append("устойчива к скачкообразной и слабо стационарной динамике")
            if spike_ratio >= 0.05:
                reasons.append("обнаружена повышенная доля резких изменений")
        elif model == "MLP Neural Network":
            reasons.append("использует лаговые значения и дополнительные числовые признаки")
            if extra_numeric_features:
                reasons.append(f"доступно дополнительных признаков: {len(extra_numeric_features)}")
        elif model == "LSTM Neural Network":
            reasons.append("использует последовательности наблюдений и может учитывать нелинейную динамику")
            if extra_numeric_features:
                reasons.append(f"доступно дополнительных признаков: {len(extra_numeric_features)}")
        ranked_models.append(
            {
                "model": model,
                "score": round(score, 3),
                "reason": "; ".join(reasons),
            }
        )

    ranked_models.sort(key=lambda item: item["score"], reverse=True)
    recommended_model = ranked_models[0]["model"]
    alternative_models = [item["model"] for item in ranked_models[1:3]]

    explanation_parts = [
        f"Уровень волатильности ряда оценивается как {volatility_level_ru}.",
    ]
    if seasonality_detected:
        explanation_parts.append(f"Обнаружена сезонная зависимость с предполагаемым периодом {best_period}.")
    if autocorrelation_detected:
        explanation_parts.append("Обнаружена автокорреляционная зависимость значений.")
    if volatility_clustering_detected:
        explanation_parts.append("Обнаружены признаки кластеризации волатильности.")
    if spike_ratio >= 0.05:
        explanation_parts.append("В ряде присутствует повышенная доля резких скачков.")
    if extra_numeric_features:
        explanation_parts.append(f"После предобработки сохранены дополнительные числовые признаки: {len(extra_numeric_features)}.")
    explanation_parts.append(f"По совокупности признаков рекомендуется модель {recommended_model}.")

    return {
        "rows_count": int(len(y)),
        "returns_count": int(len(r)),
        "volatility_level": volatility_level,
        "volatility_level_ru": volatility_level_ru,
        "std_returns": std_returns,
        "mean_abs_return": mean_abs_return,
        "relative_volatility": rel_volatility,
        "mean_rolling_volatility": mean_rolling_volatility,
        "max_rolling_volatility": max_rolling_volatility,
        "volatility_cv": volatility_cv,
        "spike_ratio": spike_ratio,
        "max_drawdown": max_drawdown,
        "acf_lag_1": acf_lag_1,
        "abs_returns_acf_lag_1": abs_returns_acf_1,
        "abs_returns_acf_lag_5": abs_returns_acf_5,
        "autocorrelation_detected": autocorrelation_detected,
        "seasonality_detected": seasonality_detected,
        "seasonal_period": best_period,
        "seasonal_strength": best_period_corr,
        "volatility_clustering_detected": volatility_clustering_detected,
        "recommended_model": recommended_model,
        "alternative_models": alternative_models,
        "model_ranking": ranked_models,
        "explanation": " ".join(explanation_parts),
    }
