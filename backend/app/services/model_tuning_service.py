from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

from app.services.linear_regression_service import run_linear_regression_forecast
from app.services.arima_service import run_arima_forecast
from app.services.sarima_service import run_sarima_forecast
from app.services.garch_service import run_garch_forecast
from app.services.fuzzy_time_series_service import run_fuzzy_first_order_forecast


_ALLOWED_OPTIMIZERS = {"mae", "mse", "rmse", "mape"}


def _unique_ints(values: list[int], *, min_value: int = 0, max_value: int | None = None) -> list[int]:
    cleaned: list[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item < min_value:
            continue
        if max_value is not None and item > max_value:
            continue
        if item not in cleaned:
            cleaned.append(item)
    return cleaned


def _normalize_optimizer(optimize_by: str, *, default: str = "rmse", allowed: set[str] | None = None) -> str:
    allowed_values = allowed or _ALLOWED_OPTIMIZERS
    value = (optimize_by or default).lower().strip()
    if value not in allowed_values:
        raise ValueError(f"Неподдерживаемая метрика оптимизации: {optimize_by}")
    return value


def _score_from_metrics(metrics: dict[str, Any], optimize_by: str) -> float | None:
    value = metrics.get(optimize_by)
    if value is None and optimize_by == "rmse" and metrics.get("mse") is not None:
        value = math.sqrt(float(metrics["mse"]))
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(score) or np.isinf(score):
        return None
    return score


def _garch_proxy_metrics(result_df: pd.DataFrame) -> dict[str, float | None]:
    actual_proxy = pd.to_numeric(result_df["actual_returns"], errors="coerce").abs()
    predicted = pd.to_numeric(result_df["forecast_volatility"], errors="coerce")
    mask = actual_proxy.notna() & predicted.notna()
    actual_proxy = actual_proxy[mask]
    predicted = predicted[mask]

    if actual_proxy.empty:
        return {"mae": None, "mse": None, "rmse": None, "mape": None}

    errors = actual_proxy - predicted
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(np.square(errors)))
    rmse = float(math.sqrt(mse))

    non_zero_mask = actual_proxy != 0
    if non_zero_mask.any():
        mape = float(np.mean(np.abs((actual_proxy[non_zero_mask] - predicted[non_zero_mask]) / actual_proxy[non_zero_mask])) * 100)
    else:
        mape = None

    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}


def tune_linear_regression_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    window_values: list[int],
    optimize_by: str = "rmse",
) -> tuple[pd.DataFrame, dict]:
    optimize_by = _normalize_optimizer(optimize_by)
    window_values = _unique_ints(window_values, min_value=1, max_value=365)
    if not window_values:
        raise ValueError("Не задан диапазон окон для линейной регрессии")

    candidates: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_df: pd.DataFrame | None = None

    for window_size in window_values:
        try:
            result_df, summary = run_linear_regression_forecast(
                processed_file_path=processed_file_path,
                forecast_horizon=forecast_horizon,
                window_size=window_size,
            )
            metrics = dict(summary.get("metrics", {}))
            score = _score_from_metrics(metrics, optimize_by)
            candidate = {
                "parameters": {"window_size": window_size},
                "window_size": window_size,
                "mae": metrics.get("mae"),
                "mse": metrics.get("mse"),
                "rmse": metrics.get("rmse"),
                "mape": metrics.get("mape"),
            }
            candidates.append(candidate)
            if score is not None and (best_result is None or score < float(best_result["score"])):
                best_result = {**candidate, "score": score, "summary": summary}
                best_df = result_df
        except Exception:
            continue

    if best_result is None or best_df is None:
        raise ValueError("Не удалось подобрать рабочую конфигурацию Linear Regression")

    candidates_sorted = sorted(candidates, key=lambda item: float("inf") if _score_from_metrics(item, optimize_by) is None else _score_from_metrics(item, optimize_by))
    best_metrics = {key: best_result.get(key) for key in ("mae", "mse", "rmse", "mape")}
    summary = {
        "best_model": "Linear Regression",
        "best_parameters": {"window_size": int(best_result["window_size"])},
        "optimize_by": optimize_by,
        "train_size": int(best_result["summary"].get("train_size", 0)),
        "test_size": int(best_result["summary"].get("test_size", 0)),
        "metrics": best_metrics,
        "candidates": candidates_sorted,
    }
    return best_df, summary


def tune_arima_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    p_values: list[int],
    d_values: list[int],
    q_values: list[int],
    optimize_by: str = "rmse",
) -> tuple[pd.DataFrame, dict]:
    optimize_by = _normalize_optimizer(optimize_by, allowed={"mae", "rmse", "mape"})
    p_values = _unique_ints(p_values, min_value=0, max_value=10)
    d_values = _unique_ints(d_values, min_value=0, max_value=5)
    q_values = _unique_ints(q_values, min_value=0, max_value=10)
    if not p_values or not d_values or not q_values:
        raise ValueError("Не задан диапазон параметров ARIMA")

    candidates: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_df: pd.DataFrame | None = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result_df, summary = run_arima_forecast(
                            processed_file_path=processed_file_path,
                            forecast_horizon=forecast_horizon,
                            p=p,
                            d=d,
                            q=q,
                        )
                    metrics = dict(summary.get("metrics", {}))
                    score = _score_from_metrics(metrics, optimize_by)
                    candidate = {
                        "parameters": {"p": p, "d": d, "q": q},
                        "order": [p, d, q],
                        "mae": metrics.get("mae"),
                        "rmse": metrics.get("rmse"),
                        "mape": metrics.get("mape"),
                    }
                    candidates.append(candidate)
                    if score is not None and (best_result is None or score < float(best_result["score"])):
                        best_result = {**candidate, "score": score, "summary": summary}
                        best_df = result_df
                except Exception:
                    continue

    if best_result is None or best_df is None:
        raise ValueError("Не удалось подобрать рабочую конфигурацию ARIMA")

    candidates_sorted = sorted(candidates, key=lambda item: float("inf") if _score_from_metrics(item, optimize_by) is None else _score_from_metrics(item, optimize_by))
    summary = {
        "best_model": "ARIMA",
        "best_order": best_result["order"],
        "best_parameters": best_result["parameters"],
        "optimize_by": optimize_by,
        "train_size": int(best_result["summary"].get("train_size", 0)),
        "test_size": int(best_result["summary"].get("test_size", 0)),
        "metrics": {
            "mae": best_result.get("mae"),
            "rmse": best_result.get("rmse"),
            "mape": best_result.get("mape"),
        },
        "candidates": candidates_sorted,
    }
    return best_df, summary


def tune_sarima_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    p_values: list[int],
    d_values: list[int],
    q_values: list[int],
    seasonal_p_values: list[int],
    seasonal_d_values: list[int],
    seasonal_q_values: list[int],
    seasonal_period_values: list[int],
    optimize_by: str = "rmse",
) -> tuple[pd.DataFrame, dict]:
    optimize_by = _normalize_optimizer(optimize_by, allowed={"mae", "rmse", "mape"})
    p_values = _unique_ints(p_values, min_value=0, max_value=5)
    d_values = _unique_ints(d_values, min_value=0, max_value=2)
    q_values = _unique_ints(q_values, min_value=0, max_value=5)
    seasonal_p_values = _unique_ints(seasonal_p_values, min_value=0, max_value=3)
    seasonal_d_values = _unique_ints(seasonal_d_values, min_value=0, max_value=2)
    seasonal_q_values = _unique_ints(seasonal_q_values, min_value=0, max_value=3)
    seasonal_period_values = _unique_ints(seasonal_period_values, min_value=2, max_value=365)

    if not all([p_values, d_values, q_values, seasonal_p_values, seasonal_d_values, seasonal_q_values, seasonal_period_values]):
        raise ValueError("Не задан полный диапазон параметров SARIMA")

    candidates: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_df: pd.DataFrame | None = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for sp in seasonal_p_values:
                    for sd in seasonal_d_values:
                        for sq in seasonal_q_values:
                            for s in seasonal_period_values:
                                if p == 0 and q == 0 and sp == 0 and sq == 0:
                                    continue
                                try:
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        result_df, summary = run_sarima_forecast(
                                            processed_file_path=processed_file_path,
                                            forecast_horizon=forecast_horizon,
                                            p=p,
                                            d=d,
                                            q=q,
                                            seasonal_p=sp,
                                            seasonal_d=sd,
                                            seasonal_q=sq,
                                            seasonal_period=s,
                                        )
                                    metrics = dict(summary.get("metrics", {}))
                                    score = _score_from_metrics(metrics, optimize_by)
                                    candidate = {
                                        "parameters": {
                                            "p": p,
                                            "d": d,
                                            "q": q,
                                            "seasonal_p": sp,
                                            "seasonal_d": sd,
                                            "seasonal_q": sq,
                                            "seasonal_period": s,
                                        },
                                        "order": [p, d, q],
                                        "seasonal_order": [sp, sd, sq, s],
                                        "mae": metrics.get("mae"),
                                        "rmse": metrics.get("rmse"),
                                        "mape": metrics.get("mape"),
                                    }
                                    candidates.append(candidate)
                                    if score is not None and (best_result is None or score < float(best_result["score"])):
                                        best_result = {**candidate, "score": score, "summary": summary}
                                        best_df = result_df
                                except Exception:
                                    continue

    if best_result is None or best_df is None:
        raise ValueError("Не удалось подобрать рабочую конфигурацию SARIMA")

    candidates_sorted = sorted(candidates, key=lambda item: float("inf") if _score_from_metrics(item, optimize_by) is None else _score_from_metrics(item, optimize_by))
    summary = {
        "best_model": "SARIMA",
        "best_order": best_result["order"],
        "best_seasonal_order": best_result["seasonal_order"],
        "best_parameters": best_result["parameters"],
        "optimize_by": optimize_by,
        "train_size": int(best_result["summary"].get("train_size", 0)),
        "test_size": int(best_result["summary"].get("test_size", 0)),
        "metrics": {
            "mae": best_result.get("mae"),
            "rmse": best_result.get("rmse"),
            "mape": best_result.get("mape"),
        },
        "candidates": candidates_sorted,
    }
    return best_df, summary


def tune_garch_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    p_values: list[int],
    q_values: list[int],
    annualize: bool = False,
    periods_per_year: int = 252,
    optimize_by: str = "mae",
) -> tuple[pd.DataFrame, dict]:
    optimize_by = _normalize_optimizer(optimize_by, default="mae")
    p_values = _unique_ints(p_values, min_value=1, max_value=5)
    q_values = _unique_ints(q_values, min_value=1, max_value=5)
    if not p_values or not q_values:
        raise ValueError("Не задан диапазон параметров GARCH")

    candidates: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_df: pd.DataFrame | None = None

    for p in p_values:
        for q in q_values:
            try:
                result_df, summary = run_garch_forecast(
                    processed_file_path=processed_file_path,
                    forecast_horizon=forecast_horizon,
                    p=p,
                    q=q,
                    annualize=annualize,
                    periods_per_year=periods_per_year,
                )
                metrics = _garch_proxy_metrics(result_df)
                score = _score_from_metrics(metrics, optimize_by)
                candidate = {
                    "parameters": {"p": p, "q": q},
                    "order": [p, q],
                    "mae": metrics.get("mae"),
                    "mse": metrics.get("mse"),
                    "rmse": metrics.get("rmse"),
                    "mape": metrics.get("mape"),
                    "mean_forecast_volatility": summary.get("mean_forecast_volatility"),
                }
                candidates.append(candidate)
                if score is not None and (best_result is None or score < float(best_result["score"])):
                    best_result = {**candidate, "score": score, "summary": summary, "metrics": metrics}
                    best_df = result_df
            except Exception:
                continue

    if best_result is None or best_df is None:
        raise ValueError("Не удалось подобрать рабочую конфигурацию GARCH")

    candidates_sorted = sorted(candidates, key=lambda item: float("inf") if _score_from_metrics(item, optimize_by) is None else _score_from_metrics(item, optimize_by))
    summary = {
        "best_model": "GARCH",
        "best_order": best_result["order"],
        "best_parameters": best_result["parameters"],
        "optimize_by": optimize_by,
        "train_size": int(best_result["summary"].get("train_size", 0)),
        "forecast_horizon": int(forecast_horizon),
        "metrics": best_result["metrics"],
        "summary": best_result["summary"],
        "candidates": candidates_sorted,
    }
    return best_df, summary


def tune_fuzzy_first_order_forecast(
    processed_file_path: str,
    min_sets_values: list[int],
    max_sets_values: list[int],
    optimize_by: str = "rmse",
) -> tuple[pd.DataFrame, dict]:
    optimize_by = _normalize_optimizer(optimize_by)
    min_sets_values = _unique_ints(min_sets_values, min_value=3, max_value=100)
    max_sets_values = _unique_ints(max_sets_values, min_value=3, max_value=150)
    if not min_sets_values or not max_sets_values:
        raise ValueError("Не задан диапазон числа нечётких множеств")

    candidates: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_df: pd.DataFrame | None = None

    for min_sets in min_sets_values:
        for max_sets in max_sets_values:
            if max_sets < min_sets:
                continue
            try:
                result_df, summary = run_fuzzy_first_order_forecast(
                    processed_file_path=processed_file_path,
                    min_sets=min_sets,
                    max_sets=max_sets,
                )
                metrics = dict(summary.get("metrics", {}))
                if metrics.get("rmse") is None and metrics.get("mse") is not None:
                    metrics["rmse"] = math.sqrt(float(metrics["mse"]))
                score = _score_from_metrics(metrics, optimize_by)
                candidate = {
                    "parameters": {"min_sets": min_sets, "max_sets": max_sets},
                    "fuzzy_sets_count": summary.get("fuzzy_sets_count"),
                    "mae": metrics.get("mae"),
                    "mse": metrics.get("mse"),
                    "rmse": metrics.get("rmse"),
                    "mape": metrics.get("mape"),
                }
                candidates.append(candidate)
                if score is not None and (best_result is None or score < float(best_result["score"])):
                    best_result = {**candidate, "score": score, "summary": summary, "metrics": metrics}
                    best_df = result_df
            except Exception:
                continue

    if best_result is None or best_df is None:
        raise ValueError("Не удалось подобрать рабочую конфигурацию Fuzzy Time Series")

    candidates_sorted = sorted(candidates, key=lambda item: float("inf") if _score_from_metrics(item, optimize_by) is None else _score_from_metrics(item, optimize_by))
    summary = {
        "best_model": "FuzzyTimeSeriesFirstOrder",
        "best_parameters": best_result["parameters"],
        "best_fuzzy_sets_count": best_result.get("fuzzy_sets_count"),
        "optimize_by": optimize_by,
        "metrics": best_result["metrics"],
        "summary": best_result["summary"],
        "candidates": candidates_sorted,
    }
    return best_df, summary


def tune_mlp_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    window_values: list[int],
    hidden_layer_1_values: list[int],
    hidden_layer_2_values: list[int],
    max_iter: int = 400,
    random_state: int = 42,
    include_exogenous: bool = True,
    optimize_by: str = "rmse",
) -> tuple[pd.DataFrame, dict]:
    from app.services.mlp_service import run_mlp_forecast

    optimize_by = _normalize_optimizer(optimize_by)
    window_values = _unique_ints(window_values, min_value=1, max_value=365)
    hidden_layer_1_values = _unique_ints(hidden_layer_1_values, min_value=1, max_value=512)
    hidden_layer_2_values = _unique_ints(hidden_layer_2_values, min_value=0, max_value=512)
    if not window_values or not hidden_layer_1_values or not hidden_layer_2_values:
        raise ValueError("Не задан диапазон параметров MLP")

    candidates: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_df: pd.DataFrame | None = None

    for window_size in window_values:
        for hidden_1 in hidden_layer_1_values:
            for hidden_2 in hidden_layer_2_values:
                try:
                    result_df, summary = run_mlp_forecast(
                        processed_file_path=processed_file_path,
                        forecast_horizon=forecast_horizon,
                        window_size=window_size,
                        hidden_layer_1=hidden_1,
                        hidden_layer_2=hidden_2,
                        max_iter=max_iter,
                        random_state=random_state,
                        include_exogenous=include_exogenous,
                    )
                    metrics = dict(summary.get("metrics", {}))
                    score = _score_from_metrics(metrics, optimize_by)
                    candidate = {
                        "parameters": {
                            "window_size": window_size,
                            "hidden_layer_1": hidden_1,
                            "hidden_layer_2": hidden_2,
                            "max_iter": max_iter,
                            "random_state": random_state,
                        },
                        "mae": metrics.get("mae"),
                        "mse": metrics.get("mse"),
                        "rmse": metrics.get("rmse"),
                        "mape": metrics.get("mape"),
                    }
                    candidates.append(candidate)
                    if score is not None and (best_result is None or score < float(best_result["score"])):
                        best_result = {**candidate, "score": score, "summary": summary}
                        best_df = result_df
                except Exception:
                    continue

    if best_result is None or best_df is None:
        raise ValueError("Не удалось подобрать рабочую конфигурацию MLP")

    candidates_sorted = sorted(candidates, key=lambda item: float("inf") if _score_from_metrics(item, optimize_by) is None else _score_from_metrics(item, optimize_by))
    summary = {
        "best_model": "MLP Neural Network",
        "best_parameters": best_result["parameters"],
        "optimize_by": optimize_by,
        "train_size": int(best_result["summary"].get("train_size", 0)),
        "test_size": int(best_result["summary"].get("test_size", 0)),
        "feature_columns": best_result["summary"].get("feature_columns", []),
        "metrics": {key: best_result.get(key) for key in ("mae", "mse", "rmse", "mape")},
        "candidates": candidates_sorted,
    }
    return best_df, summary


def tune_lstm_forecast(
    processed_file_path: str,
    forecast_horizon: int,
    window_values: list[int],
    hidden_size_values: list[int],
    num_layers_values: list[int],
    epochs: int = 40,
    learning_rate: float = 0.01,
    random_state: int = 42,
    include_exogenous: bool = True,
    optimize_by: str = "rmse",
) -> tuple[pd.DataFrame, dict]:
    from app.services.lstm_service import run_lstm_forecast

    optimize_by = _normalize_optimizer(optimize_by)
    window_values = _unique_ints(window_values, min_value=1, max_value=365)
    hidden_size_values = _unique_ints(hidden_size_values, min_value=4, max_value=512)
    num_layers_values = _unique_ints(num_layers_values, min_value=1, max_value=4)
    if not window_values or not hidden_size_values or not num_layers_values:
        raise ValueError("Не задан диапазон параметров LSTM")

    candidates: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_df: pd.DataFrame | None = None

    for window_size in window_values:
        for hidden_size in hidden_size_values:
            for num_layers in num_layers_values:
                try:
                    result_df, summary = run_lstm_forecast(
                        processed_file_path=processed_file_path,
                        forecast_horizon=forecast_horizon,
                        window_size=window_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        random_state=random_state,
                        include_exogenous=include_exogenous,
                    )
                    metrics = dict(summary.get("metrics", {}))
                    score = _score_from_metrics(metrics, optimize_by)
                    candidate = {
                        "parameters": {
                            "window_size": window_size,
                            "hidden_size": hidden_size,
                            "num_layers": num_layers,
                            "epochs": epochs,
                            "learning_rate": learning_rate,
                            "random_state": random_state,
                        },
                        "mae": metrics.get("mae"),
                        "mse": metrics.get("mse"),
                        "rmse": metrics.get("rmse"),
                        "mape": metrics.get("mape"),
                    }
                    candidates.append(candidate)
                    if score is not None and (best_result is None or score < float(best_result["score"])):
                        best_result = {**candidate, "score": score, "summary": summary}
                        best_df = result_df
                except Exception:
                    continue

    if best_result is None or best_df is None:
        raise ValueError("Не удалось подобрать рабочую конфигурацию LSTM")

    candidates_sorted = sorted(candidates, key=lambda item: float("inf") if _score_from_metrics(item, optimize_by) is None else _score_from_metrics(item, optimize_by))
    summary = {
        "best_model": "LSTM Neural Network",
        "best_parameters": best_result["parameters"],
        "optimize_by": optimize_by,
        "train_size": int(best_result["summary"].get("train_size", 0)),
        "test_size": int(best_result["summary"].get("test_size", 0)),
        "feature_columns": best_result["summary"].get("feature_columns", []),
        "metrics": {key: best_result.get(key) for key in ("mae", "mse", "rmse", "mape")},
        "candidates": candidates_sorted,
    }
    return best_df, summary
