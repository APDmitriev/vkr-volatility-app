from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class MockBackend:
    def __init__(self) -> None:
        self.experiments: list[dict[str, Any]] = []

    def sample_dataframe(self) -> pd.DataFrame:
        n = 180
        x = np.arange(n)
        trend = 100 + 0.35 * x
        seasonal = 8 * np.sin(x / 6)
        noise = np.sin(x / 2.3) * 1.5 + np.cos(x / 9) * 0.9
        values = trend + seasonal + noise
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        return pd.DataFrame({"timestamp": dates, "value": np.round(values, 2)})

    def load_dataset(self, file_path: str | None, file_type: str, delimiter: str, time_column: str | None, target_column: str | None) -> dict[str, Any]:
        if not file_path:
            df = self.sample_dataframe()
            dataset_name = "demo_series"
        else:
            path = Path(file_path)
            dataset_name = path.stem
            if file_type.lower() == "xlsx" or path.suffix.lower() == ".xlsx":
                df = pd.read_excel(path)
            else:
                sep = "\t" if delimiter == "tab" else delimiter
                df = pd.read_csv(path, sep=sep)

        if time_column and time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
            df = df.sort_values(time_column)
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")

        if target_column and target_column in df.columns:
            df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
        elif "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df = df.reset_index(drop=True)
        return {
            "dataset_name": dataset_name,
            "columns": list(df.columns),
            "preview": df.head(200),
            "dataframe": df,
        }

    def preprocess_dataset(self, df: pd.DataFrame, *, time_column: str, target_column: str, missing_strategy: str, normalization: str, smoothing: str, resample: str) -> dict[str, Any]:
        result = df.copy()
        if time_column in result.columns:
            result[time_column] = pd.to_datetime(result[time_column], errors="coerce")
            result = result.sort_values(time_column)

        if target_column in result.columns:
            result[target_column] = pd.to_numeric(result[target_column], errors="coerce")
            if missing_strategy == "Удалить пропуски":
                result = result.dropna(subset=[target_column])
            elif missing_strategy == "Интерполяция":
                result[target_column] = result[target_column].interpolate(limit_direction="both")

            if smoothing == "Скользящее среднее":
                result[target_column] = result[target_column].rolling(window=5, min_periods=1).mean()
            elif smoothing == "EMA":
                result[target_column] = result[target_column].ewm(span=5, adjust=False).mean()

            if normalization == "Нормализация":
                series = result[target_column]
                min_v = series.min()
                max_v = series.max()
                if pd.notna(min_v) and pd.notna(max_v) and max_v != min_v:
                    result[target_column] = (series - min_v) / (max_v - min_v)
            elif normalization == "Стандартизация":
                series = result[target_column]
                std = series.std()
                if pd.notna(std) and std not in (0, 0.0):
                    result[target_column] = (series - series.mean()) / std

        if resample != "Без ресемплинга" and time_column in result.columns and target_column in result.columns:
            freq_map = {"1 час": "H", "1 день": "D", "1 неделя": "W"}
            freq = freq_map.get(resample)
            if freq:
                result = (
                    result.set_index(time_column)[[target_column]]
                    .resample(freq)
                    .mean()
                    .interpolate(limit_direction="both")
                    .reset_index()
                )

        return {
            "dataframe": result.reset_index(drop=True),
            "preview": result.head(200).reset_index(drop=True),
        }

    def analyze_dataset(self, df: pd.DataFrame, *, time_column: str, target_column: str) -> dict[str, Any]:
        data = df.copy()
        y = pd.to_numeric(data[target_column], errors="coerce").dropna().to_numpy()
        if len(y) == 0:
            raise ValueError("Нет числовых данных для анализа")

        x = np.arange(len(y))
        trend_coef = np.polyfit(x, y, 1)
        trend_line = trend_coef[0] * x + trend_coef[1]
        rolling_mean = pd.Series(y).rolling(window=7, min_periods=1).mean().to_numpy()
        volatility = pd.Series(y).rolling(window=7, min_periods=2).std().fillna(0).to_numpy()
        centered = y - np.mean(y)
        acf = np.correlate(centered, centered, mode="full")
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        max_lag = min(20, len(acf) - 1)

        stats = {
            "count": int(len(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "median": float(np.median(y)),
            "var_coef": float(np.std(y) / np.mean(y)) if np.mean(y) != 0 else 0.0,
        }
        return {
            "stats": stats,
            "series_x": data[time_column].tolist() if time_column in data.columns else list(range(len(y))),
            "series_y": y.tolist(),
            "trend": trend_line.tolist(),
            "rolling_mean": rolling_mean.tolist(),
            "volatility": volatility.tolist(),
            "acf_lags": list(range(max_lag + 1)),
            "acf_values": acf[: max_lag + 1].tolist(),
            "histogram_values": y.tolist(),
            "volatility_profile": self._build_mock_volatility_profile(y),
        }


    @staticmethod
    def _build_mock_volatility_profile(y: np.ndarray) -> dict[str, Any]:
        if len(y) < 3:
            return {}
        series = pd.Series(y)
        returns = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            returns = series.diff().replace([np.inf, -np.inf], np.nan).dropna()
        std_returns = float(returns.std()) if not returns.empty else 0.0
        rolling_vol = returns.rolling(window=min(7, max(2, len(returns))), min_periods=2).std().dropna()
        mean_vol = float(rolling_vol.mean()) if not rolling_vol.empty else 0.0
        max_vol = float(rolling_vol.max()) if not rolling_vol.empty else 0.0
        spike_ratio = float((returns.abs() > max(3 * std_returns, 1e-12)).mean()) if not returns.empty else 0.0
        relative_volatility = float(np.std(y) / np.mean(np.abs(y))) if np.mean(np.abs(y)) != 0 else 0.0
        if std_returns >= 0.05 or spike_ratio >= 0.08 or relative_volatility >= 0.35:
            level, level_ru = "high", "высокая"
        elif std_returns >= 0.02 or spike_ratio >= 0.03 or relative_volatility >= 0.12:
            level, level_ru = "medium", "средняя"
        else:
            level, level_ru = "low", "низкая"
        autocorrelation_detected = True
        seasonality_detected = len(y) >= 21
        volatility_clustering_detected = level in {"medium", "high"}
        if seasonality_detected:
            recommended = "SARIMA"
        elif volatility_clustering_detected and level == "high":
            recommended = "GARCH"
        elif level == "high":
            recommended = "Fuzzy First Order"
        elif autocorrelation_detected:
            recommended = "ARIMA"
        else:
            recommended = "Linear Regression"
        ranking = [
            {"model": recommended, "score": 0.9, "reason": "рекомендована по результатам оценки профиля ряда"},
            {"model": "ARIMA", "score": 0.72, "reason": "учитывает зависимость текущих значений от прошлых"},
            {"model": "Fuzzy First Order", "score": 0.65, "reason": "устойчива к скачкообразной динамике"},
            {"model": "GARCH", "score": 0.62, "reason": "используется для прогноза волатильности"},
            {"model": "Linear Regression", "score": 0.5, "reason": "простая базовая модель"},
        ]
        return {
            "rows_count": int(len(y)),
            "returns_count": int(len(returns)),
            "volatility_level": level,
            "volatility_level_ru": level_ru,
            "std_returns": std_returns,
            "mean_abs_return": float(returns.abs().mean()) if not returns.empty else 0.0,
            "relative_volatility": relative_volatility,
            "mean_rolling_volatility": mean_vol,
            "max_rolling_volatility": max_vol,
            "volatility_cv": float(rolling_vol.std() / rolling_vol.mean()) if len(rolling_vol) > 2 and rolling_vol.mean() not in (0, 0.0) else 0.0,
            "spike_ratio": spike_ratio,
            "max_drawdown": 0.0,
            "acf_lag_1": 0.0,
            "abs_returns_acf_lag_1": 0.0,
            "abs_returns_acf_lag_5": 0.0,
            "autocorrelation_detected": autocorrelation_detected,
            "seasonality_detected": seasonality_detected,
            "seasonal_period": 7 if seasonality_detected else None,
            "seasonal_strength": 0.5 if seasonality_detected else 0.0,
            "volatility_clustering_detected": volatility_clustering_detected,
            "recommended_model": recommended,
            "alternative_models": [row["model"] for row in ranking if row["model"] != recommended][:2],
            "model_ranking": ranking,
            "explanation": f"Уровень волатильности ряда оценивается как {level_ru}. По совокупности признаков рекомендуется модель {recommended}.",
        }

    def train_model(self, df: pd.DataFrame, *, time_column: str, target_column: str, model_name: str, horizon: int, train_ratio: int, parameters: dict[str, Any]) -> dict[str, Any]:
        data = df.copy()
        y = pd.to_numeric(data[target_column], errors="coerce").dropna().to_numpy()
        n = len(y)
        if n < 10:
            raise ValueError("Недостаточно данных для обучения")
        train_size = max(5, int(n * train_ratio / 100))
        train = y[:train_size]
        test = y[train_size:]
        if len(test) == 0:
            test = y[-min(10, n):]

        if model_name in {"Linear Regression", "Linear Regression Auto"}:
            x_train = np.arange(len(train))
            coef = np.polyfit(x_train, train, 1)
            x_test = np.arange(len(train), len(train) + len(test))
            pred_test = coef[0] * x_test + coef[1]
        elif model_name in {"SARIMA", "SARIMA Auto"}:
            seasonal_adjustment = np.array([0.15 * math.sin(i / max(1, int(parameters.get("seasonal_period", 7))) * 2 * math.pi) for i in range(len(test))])
            base = np.array([train[-1] + (i + 1) * ((train[-1] - train[-7]) / 7) for i in range(len(test))], dtype=float)
            pred_test = base + seasonal_adjustment
        elif model_name == "Random Forest":
            pred_test = np.full_like(test, np.mean(train[-10:]), dtype=float)
        elif model_name in {"Fuzzy Time Series", "Fuzzy First Order", "Fuzzy First Order Auto"}:
            pred_test = np.array([train[-1] + 0.25 * math.sin(i) for i in range(len(test))], dtype=float)
        else:
            pred_test = np.array([train[-1] + (i + 1) * ((train[-1] - train[-7]) / 7) for i in range(len(test))], dtype=float)

        mae = float(np.mean(np.abs(test - pred_test)))
        mse = float(np.mean((test - pred_test) ** 2))
        rmse = float(np.sqrt(mse))
        non_zero = np.where(test == 0, 1e-8, test)
        mape = float(np.mean(np.abs((test - pred_test) / non_zero)) * 100)

        experiment = {
            "model": model_name,
            "parameters": parameters,
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
            "pred_test": pred_test.tolist(),
            "actual_test": test.tolist(),
        }
        self.experiments.insert(0, experiment)
        self.experiments = self.experiments[:20]
        return experiment

    def forecast(self, df: pd.DataFrame, *, time_column: str, target_column: str, model_name: str, horizon: int) -> dict[str, Any]:
        data = df.copy()
        y = pd.to_numeric(data[target_column], errors="coerce").dropna().to_numpy()
        if len(y) < 2:
            raise ValueError("Недостаточно данных для прогноза")
        slope = float(np.mean(np.diff(y[-min(10, len(y)): ])))
        last = float(y[-1])
        forecast = [round(last + slope * (i + 1), 4) for i in range(horizon)]

        if time_column in data.columns and pd.api.types.is_datetime64_any_dtype(data[time_column]):
            start = pd.to_datetime(data[time_column].iloc[-1])
            future_index = pd.date_range(start + pd.Timedelta(days=1), periods=horizon, freq="D")
            future_x = future_index.tolist()
        else:
            future_x = list(range(len(y), len(y) + horizon))

        return {
            "future_x": future_x,
            "forecast": forecast,
            "history_x": data[time_column].tolist() if time_column in data.columns else list(range(len(y))),
            "history_y": y.tolist(),
            "next_value": forecast[0],
        }

    def get_experiments(self) -> list[dict[str, Any]]:
        return list(self.experiments)
