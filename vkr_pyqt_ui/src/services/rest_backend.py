from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


class RestBackend:
    def __init__(self, base_url: str) -> None:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/api/v1"):
            base_url = f"{base_url}/api/v1"
        self.base_url = base_url
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _safe_json(response: requests.Response) -> Any:
        try:
            return response.json()
        except Exception:
            return response.text

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.ok:
            return
        payload = self._safe_json(response)
        if isinstance(payload, dict) and payload.get("detail"):
            raise ValueError(str(payload["detail"]))
        raise ValueError(f"HTTP {response.status_code}: {payload}")

    def health(self) -> dict[str, Any]:
        response = self.session.get(self._url("/health"), timeout=15)
        if response.ok:
            return response.json()
        # В backend из ВКР endpoint health может отсутствовать.
        # В таком случае проверяем доступность через существующий список проектов.
        if response.status_code == 404:
            projects_response = self.session.get(self._url("/projects/"), timeout=15)
            self._raise_for_status(projects_response)
            return {"status": "ok", "checked_by": "projects"}
        self._raise_for_status(response)
        return response.json()

    def list_projects(self) -> list[dict[str, Any]]:
        response = self.session.get(self._url("/projects/"), timeout=30)
        self._raise_for_status(response)
        items = response.json()
        return items if isinstance(items, list) else []

    def create_project(self, name: str, description: str = "") -> dict[str, Any]:
        response = self.session.post(
            self._url("/projects/"),
            json={"name": name, "description": description or None},
            timeout=30,
        )
        self._raise_for_status(response)
        return response.json()

    def delete_project(self, project_id: int) -> None:
        response = self.session.delete(self._url(f"/projects/{project_id}"), timeout=30)
        self._raise_for_status(response)


    def list_datasets(self, project_id: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if project_id is not None:
            params["project_id"] = project_id
        response = self.session.get(self._url("/datasets/"), params=params, timeout=30)
        self._raise_for_status(response)
        items = response.json()
        return items if isinstance(items, list) else []

    def open_existing_dataset(self, dataset: dict[str, Any], *, file_type: str = "CSV", delimiter: str = ",") -> dict[str, Any]:
        dataset_id = dataset.get("id")
        file_path = dataset.get("file_path")
        if not file_path:
            raise ValueError("У выбранного датасета нет file_path")
        path = Path(str(file_path))
        if not path.exists():
            raise ValueError(f"Файл датасета не найден на диске: {path}")

        if path.suffix.lower() in {".xlsx", ".xls"} or file_type.lower() == "xlsx":
            df = pd.read_excel(path)
        else:
            sep = "\t" if delimiter == "tab" else delimiter
            df = pd.read_csv(path, sep=sep)

        date_column = dataset.get("date_column")
        value_column = dataset.get("value_column")
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        if value_column and value_column in df.columns:
            df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

        return {
            "dataset_name": dataset.get("name", path.stem),
            "columns": [str(col) for col in df.columns.tolist()],
            "preview": df.head(200),
            "dataframe": df,
            "dataset_id": dataset_id,
            "file_path": str(path),
            "rows_count": dataset.get("rows_count"),
            "date_column": date_column,
            "value_column": value_column,
        }

    def load_dataset(
        self,
        *,
        file_path: str | None,
        file_type: str,
        delimiter: str,
        time_column: str | None,
        target_column: str | None,
        project_id: int | None,
    ) -> dict[str, Any]:
        if not file_path:
            raise ValueError("Для REST backend нужно выбрать файл")
        if project_id is None:
            raise ValueError("Не удалось определить project_id")

        path = Path(file_path)
        if not path.exists():
            raise ValueError("Файл не найден")

        with open(path, "rb") as fh:
            response = self.session.post(
                self._url("/datasets/upload"),
                files={"file": (path.name, fh)},
                data={"project_id": str(project_id)},
                timeout=120,
            )
        self._raise_for_status(response)
        payload = response.json()

        preview = pd.DataFrame(payload.get("preview", []))
        # для нормальной визуализации берём полный локальный файл, а не только preview с backend
        if path.suffix.lower() in {".xlsx", ".xls"} or file_type.lower() == "xlsx":
            df = pd.read_excel(path)
        else:
            sep = "\t" if delimiter == "tab" else delimiter
            df = pd.read_csv(path, sep=sep)

        if time_column and time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
        if target_column and target_column in df.columns:
            df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

        return {
            "dataset_name": payload.get("name", path.stem),
            "columns": payload.get("columns", list(df.columns)),
            "preview": preview if not preview.empty else df.head(200),
            "dataframe": df,
            "dataset_id": payload.get("id"),
            "file_path": payload.get("file_path"),
            "rows_count": payload.get("rows_count"),
        }

    def validate_dataset(self, dataset_id: int, *, date_column: str, value_column: str) -> dict[str, Any]:
        response = self.session.post(
            self._url(f"/datasets/{dataset_id}/validate"),
            json={"date_column": date_column, "value_column": value_column},
            timeout=60,
        )
        self._raise_for_status(response)
        return response.json()

    def preprocess_dataset(
        self,
        *,
        dataset_id: int,
        date_column: str,
        value_column: str,
        missing_strategy: str,
        returns_method: str,
        drop_duplicate_rows: bool,
        drop_duplicate_timestamps: bool,
        sort_by_date: bool,
    ) -> dict[str, Any]:
        self.validate_dataset(dataset_id, date_column=date_column, value_column=value_column)

        fill_method_map = {
            "Удалить пропуски": "drop",
            "Заполнить предыдущим": "ffill",
            "Заполнить следующим": "bfill",
            "Интерполяция": "ffill",
        }
        payload = {
            "date_column": date_column,
            "value_column": value_column,
            "drop_duplicate_rows": bool(drop_duplicate_rows),
            "drop_duplicate_timestamps": bool(drop_duplicate_timestamps),
            "sort_by_date": bool(sort_by_date),
            "fill_method": fill_method_map.get(missing_strategy, "drop"),
            "returns_method": returns_method,
        }
        response = self.session.post(
            self._url(f"/datasets/{dataset_id}/preprocess"),
            json=payload,
            timeout=120,
        )
        self._raise_for_status(response)
        data = response.json()
        processed_file_path = data.get("processed_file_path")
        preview = pd.DataFrame(data.get("preview", []))

        full_df = preview
        if processed_file_path and Path(processed_file_path).exists():
            full_df = pd.read_csv(processed_file_path)
        if "timestamp" in full_df.columns:
            full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], errors="coerce")
        if "timestamp" in preview.columns:
            preview["timestamp"] = pd.to_datetime(preview["timestamp"], errors="coerce")
        return {
            "dataframe": full_df,
            "preview": preview if not preview.empty else full_df.head(200),
            "processed_file_path": processed_file_path,
            "summary": data.get("summary", {}),
        }

    def analyze_dataset(
        self,
        *,
        processed_df: pd.DataFrame,
        processed_file_path: str | None,
        time_column: str,
        target_column: str,
    ) -> dict[str, Any]:
        if processed_df.empty:
            raise ValueError("После предобработки не осталось данных для анализа")

        df = processed_df.copy()
        if "timestamp" in df.columns:
            x = pd.to_datetime(df["timestamp"], errors="coerce")
        elif time_column in df.columns:
            x = pd.to_datetime(df[time_column], errors="coerce")
        else:
            x = pd.Series(range(len(df)))

        value_col = "value" if "value" in df.columns else target_column
        y_series = pd.to_numeric(df[value_col], errors="coerce").dropna().reset_index(drop=True)
        x = x.iloc[: len(y_series)] if hasattr(x, "iloc") else x[: len(y_series)]

        if y_series.empty:
            raise ValueError("Нет числовых данных для анализа")

        y = y_series.to_numpy(dtype=float)
        idx = np.arange(len(y), dtype=float)
        trend_coef = np.polyfit(idx, y, 1)
        trend_line = trend_coef[0] * idx + trend_coef[1]
        centered = y - np.mean(y)
        acf = np.correlate(centered, centered, mode="full")
        acf = acf[len(acf) // 2 :]
        acf = acf / acf[0] if acf[0] != 0 else acf
        max_lag = min(20, len(acf) - 1)

        volatility_values = pd.Series(y).rolling(window=7, min_periods=2).std().fillna(0).to_numpy()
        if processed_file_path:
            response = self.session.post(
                self._url("/analysis/volatility"),
                json={
                    "processed_file_path": processed_file_path,
                    "window_size": 7,
                    "annualize": False,
                    "periods_per_year": 252,
                },
                timeout=120,
            )
            if response.ok:
                payload = response.json()
                vol_df = pd.DataFrame(payload.get("preview", []))
                if "volatility" in vol_df.columns:
                    volatility_values = pd.to_numeric(vol_df["volatility"], errors="coerce").fillna(0).to_numpy()
                    if len(volatility_values) < len(y):
                        pad = np.zeros(len(y) - len(volatility_values))
                        volatility_values = np.concatenate([volatility_values, pad])
                    else:
                        volatility_values = volatility_values[: len(y)]

        stats = {
            "count": int(len(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "median": float(np.median(y)),
            "var_coef": float(np.std(y) / np.mean(y)) if np.mean(y) != 0 else 0.0,
        }
        volatility_profile: dict[str, Any] = {}
        if processed_file_path:
            profile_response = self.session.post(
                self._url("/analysis/volatility-profile"),
                json={
                    "processed_file_path": processed_file_path,
                    "window_size": 7,
                    "periods_per_year": 252,
                },
                timeout=120,
            )
            if profile_response.ok:
                volatility_profile = profile_response.json()

        return {
            "stats": stats,
            "series_x": x.tolist() if hasattr(x, "tolist") else list(x),
            "series_y": y.tolist(),
            "trend": trend_line.tolist(),
            "acf_lags": list(range(max_lag + 1)),
            "acf_values": acf[: max_lag + 1].tolist(),
            "volatility": volatility_values.tolist(),
            "histogram_values": y.tolist(),
            "volatility_profile": volatility_profile,
        }

    @staticmethod
    def _round_metric(value: Any, digits: int = 4) -> float | None:
        if value is None or value == "—":
            return None
        try:
            return round(float(value), digits)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _metrics_from_payload(payload: dict[str, Any]) -> dict[str, float | None]:
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        out = {
            "mae": RestBackend._round_metric(metrics.get("mae")),
            "mse": RestBackend._round_metric(metrics.get("mse")),
            "rmse": RestBackend._round_metric(metrics.get("rmse")),
            "mape": RestBackend._round_metric(metrics.get("mape")),
        }
        if out["mse"] is None and out["rmse"] is not None:
            out["mse"] = RestBackend._round_metric(float(out["rmse"]) ** 2)
        return out

    def tune_model_parameters(
        self,
        *,
        processed_file_path: str | None,
        model_name: str,
        horizon: int,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        if not processed_file_path:
            raise ValueError("Не найден путь к обработанному датасету")

        model_name = model_name.strip()
        if model_name == "Linear Regression":
            max_window = max(2, int(parameters.get("window_size", 14)))
            window_values = sorted(set([2, 3, 5, 7, 10, max_window] + list(range(2, max_window + 1))))
            response = self.session.post(
                self._url("/analysis/forecast/linear-regression/tune"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "window_values": window_values,
                    "optimize_by": "rmse",
                },
                timeout=600,
            )
            self._raise_for_status(response)
            payload = response.json()
            best_parameters = payload.get("best_parameters", {})
            return {
                "model": model_name,
                "best_parameters": {"window_size": int(best_parameters.get("window_size", max_window))},
                "metrics": self._metrics_from_payload(payload),
                "optimize_by": payload.get("optimize_by", "rmse"),
                "candidates_count": len(payload.get("candidates", [])),
                "candidates_file_path": payload.get("candidates_file_path"),
                "raw_response": payload,
            }

        if model_name == "ARIMA":
            max_p = max(0, int(parameters.get("p", 2)))
            max_d = max(0, int(parameters.get("d", 1)))
            max_q = max(0, int(parameters.get("q", 2)))
            response = self.session.post(
                self._url("/analysis/forecast/arima/tune"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "p_values": list(range(0, max_p + 1)),
                    "d_values": list(range(0, max_d + 1)),
                    "q_values": list(range(0, max_q + 1)),
                    "optimize_by": "rmse",
                },
                timeout=600,
            )
            self._raise_for_status(response)
            payload = response.json()
            order = payload.get("best_order") or [max_p, max_d, max_q]
            return {
                "model": model_name,
                "best_parameters": {"p": int(order[0]), "d": int(order[1]), "q": int(order[2])},
                "metrics": self._metrics_from_payload(payload),
                "optimize_by": payload.get("optimize_by", "rmse"),
                "candidates_count": len(payload.get("candidates", [])),
                "candidates_file_path": payload.get("candidates_file_path"),
                "raw_response": payload,
            }

        if model_name == "SARIMA":
            max_p = max(0, int(parameters.get("p", 2)))
            max_d = max(0, int(parameters.get("d", 1)))
            max_q = max(0, int(parameters.get("q", 2)))
            max_sp = max(0, int(parameters.get("seasonal_p", 1)))
            max_sd = max(0, int(parameters.get("seasonal_d", 0)))
            max_sq = max(0, int(parameters.get("seasonal_q", 1)))
            seasonal_period = max(2, int(parameters.get("seasonal_period", 7)))
            response = self.session.post(
                self._url("/analysis/forecast/sarima/tune"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "p_values": list(range(0, max_p + 1)),
                    "d_values": list(range(0, max_d + 1)),
                    "q_values": list(range(0, max_q + 1)),
                    "seasonal_p_values": list(range(0, max_sp + 1)),
                    "seasonal_d_values": list(range(0, max_sd + 1)),
                    "seasonal_q_values": list(range(0, max_sq + 1)),
                    "seasonal_period_values": [seasonal_period],
                    "optimize_by": "rmse",
                },
                timeout=900,
            )
            self._raise_for_status(response)
            payload = response.json()
            best_parameters = payload.get("best_parameters", {})
            if not best_parameters:
                order = payload.get("best_order") or [max_p, max_d, max_q]
                seasonal_order = payload.get("best_seasonal_order") or [max_sp, max_sd, max_sq, seasonal_period]
                best_parameters = {
                    "p": int(order[0]),
                    "d": int(order[1]),
                    "q": int(order[2]),
                    "seasonal_p": int(seasonal_order[0]),
                    "seasonal_d": int(seasonal_order[1]),
                    "seasonal_q": int(seasonal_order[2]),
                    "seasonal_period": int(seasonal_order[3]),
                }
            return {
                "model": model_name,
                "best_parameters": best_parameters,
                "metrics": self._metrics_from_payload(payload),
                "optimize_by": payload.get("optimize_by", "rmse"),
                "candidates_count": len(payload.get("candidates", [])),
                "candidates_file_path": payload.get("candidates_file_path"),
                "raw_response": payload,
            }

        if model_name == "GARCH":
            max_p = max(1, int(parameters.get("p", 2)))
            max_q = max(1, int(parameters.get("q", 2)))
            response = self.session.post(
                self._url("/analysis/forecast/garch/tune"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "p_values": list(range(1, max_p + 1)),
                    "q_values": list(range(1, max_q + 1)),
                    "annualize": False,
                    "periods_per_year": 252,
                    "optimize_by": "mae",
                },
                timeout=600,
            )
            self._raise_for_status(response)
            payload = response.json()
            best_parameters = payload.get("best_parameters", {})
            return {
                "model": model_name,
                "best_parameters": {"p": int(best_parameters.get("p", max_p)), "q": int(best_parameters.get("q", max_q))},
                "metrics": self._metrics_from_payload(payload),
                "optimize_by": payload.get("optimize_by", "mae"),
                "candidates_count": len(payload.get("candidates", [])),
                "candidates_file_path": payload.get("candidates_file_path"),
                "raw_response": payload,
            }

        if model_name == "Fuzzy First Order":
            max_sets = max(7, int(parameters.get("fuzzy_sets", 30)))
            base_values = [7, 10, 15, 20, 25, 30, max_sets]
            max_sets_values = sorted({value for value in base_values if 7 <= value <= max_sets})
            if max_sets not in max_sets_values:
                max_sets_values.append(max_sets)
            response = self.session.post(
                self._url("/analysis/forecast/fuzzy-first-order/tune"),
                json={
                    "processed_file_path": processed_file_path,
                    "min_sets_values": [7],
                    "max_sets_values": max_sets_values,
                    "optimize_by": "rmse",
                },
                timeout=600,
            )
            self._raise_for_status(response)
            payload = response.json()
            best_parameters = payload.get("best_parameters", {})
            return {
                "model": model_name,
                "best_parameters": {"max_sets": int(best_parameters.get("max_sets", payload.get("best_fuzzy_sets_count") or max_sets))},
                "metrics": self._metrics_from_payload(payload),
                "optimize_by": payload.get("optimize_by", "rmse"),
                "candidates_count": len(payload.get("candidates", [])),
                "candidates_file_path": payload.get("candidates_file_path"),
                "raw_response": payload,
            }



        if model_name == "MLP Neural Network":
            max_window = max(2, int(parameters.get("window_size", 14)))
            max_hidden_1 = max(1, int(parameters.get("hidden_layer_1", 64)))
            hidden_2 = max(0, int(parameters.get("hidden_layer_2", 32)))
            response = self.session.post(
                self._url("/analysis/forecast/mlp/tune"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "window_values": sorted(set([3, 5, 7, 10, max_window])),
                    "hidden_layer_1_values": sorted(set([32, 64, max_hidden_1])),
                    "hidden_layer_2_values": sorted(set([0, 32, hidden_2])),
                    "max_iter": max(100, int(parameters.get("max_iter", 400))),
                    "random_state": int(parameters.get("random_state", parameters.get("seed", 42))),
                    "include_exogenous": bool(parameters.get("include_exogenous", True)),
                    "optimize_by": "rmse",
                },
                timeout=900,
            )
            self._raise_for_status(response)
            payload = response.json()
            best_parameters = payload.get("best_parameters", {})
            return {
                "model": model_name,
                "best_parameters": best_parameters,
                "metrics": self._metrics_from_payload(payload),
                "optimize_by": payload.get("optimize_by", "rmse"),
                "candidates_count": len(payload.get("candidates", [])),
                "candidates_file_path": payload.get("candidates_file_path"),
                "raw_response": payload,
            }

        if model_name == "LSTM Neural Network":
            max_window = max(2, int(parameters.get("window_size", 14)))
            hidden_size = max(4, int(parameters.get("hidden_size", parameters.get("lstm_hidden_size", 32))))
            response = self.session.post(
                self._url("/analysis/forecast/lstm/tune"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "window_values": sorted(set([3, 5, 7, max_window])),
                    "hidden_size_values": sorted(set([16, 32, hidden_size])),
                    "num_layers_values": [max(1, int(parameters.get("num_layers", 1)))],
                    "epochs": max(5, int(parameters.get("epochs", 40))),
                    "learning_rate": 0.01,
                    "random_state": int(parameters.get("random_state", parameters.get("seed", 42))),
                    "include_exogenous": bool(parameters.get("include_exogenous", True)),
                    "optimize_by": "rmse",
                },
                timeout=1200,
            )
            self._raise_for_status(response)
            payload = response.json()
            best_parameters = payload.get("best_parameters", {})
            return {
                "model": model_name,
                "best_parameters": best_parameters,
                "metrics": self._metrics_from_payload(payload),
                "optimize_by": payload.get("optimize_by", "rmse"),
                "candidates_count": len(payload.get("candidates", [])),
                "candidates_file_path": payload.get("candidates_file_path"),
                "raw_response": payload,
            }

        raise ValueError("Текущий backend поддерживает Linear Regression, ARIMA, SARIMA, GARCH, Fuzzy First Order, MLP и LSTM")

    def train_model(
        self,
        *,
        processed_df: pd.DataFrame,
        processed_file_path: str | None,
        model_name: str,
        horizon: int,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        if not processed_file_path:
            raise ValueError("Не найден путь к обработанному датасету")

        model_name = model_name.strip()
        if model_name == "Linear Regression Auto":
            max_window = max(2, int(parameters.get("window_size", 14)))
            window_values = sorted(set([2, 3, 5, 7, 10, max_window] + list(range(2, max_window + 1))))
            response = self.session.post(
                self._url("/analysis/forecast/linear-regression/tune"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "window_values": window_values,
                    "optimize_by": "rmse",
                },
                timeout=600,
            )
            self._raise_for_status(response)
            payload = response.json()
            metrics = payload.get("metrics", {})
            best_parameters = payload.get("best_parameters", {})
            return {
                "model": "Linear Regression Auto",
                "parameters": {"best_parameters": best_parameters, "optimize_by": payload.get("optimize_by", "rmse")},
                "mae": round(float(metrics.get("mae", 0.0)), 4) if metrics.get("mae") is not None else "—",
                "mse": round(float(metrics.get("mse", 0.0)), 4) if metrics.get("mse") is not None else "—",
                "rmse": round(float(metrics.get("rmse", 0.0)), 4) if metrics.get("rmse") is not None else "—",
                "mape": round(float(metrics.get("mape", 0.0)), 4) if metrics.get("mape") is not None else "—",
                "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                "result_file_path": payload.get("result_file_path"),
                "raw_response": payload,
            }

        if model_name == "Linear Regression":
            response = self.session.post(
                self._url("/analysis/forecast/linear-regression"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "window_size": max(1, int(parameters.get("window_size", parameters.get("lags", 5)))),
                },
                timeout=300,
            )
            self._raise_for_status(response)
            payload = response.json()
            metrics = payload.get("metrics", {})
            return {
                "model": "Linear Regression",
                "parameters": {"window_size": payload.get("window_size", max(1, int(parameters.get("window_size", 5))))},
                "mae": round(float(metrics.get("mae", 0.0)), 4) if metrics.get("mae") is not None else "—",
                "mse": round(float(metrics.get("mse", 0.0)), 4) if metrics.get("mse") is not None else "—",
                "rmse": round(float(metrics.get("rmse", 0.0)), 4) if metrics.get("rmse") is not None else "—",
                "mape": round(float(metrics.get("mape", 0.0)), 4) if metrics.get("mape") is not None else "—",
                "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                "result_file_path": payload.get("result_file_path"),
                "raw_response": payload,
            }

        if model_name == "ARIMA":
            response = self.session.post(
                self._url("/analysis/forecast/arima"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "p": int(parameters.get("p", 1)),
                    "d": int(parameters.get("d", 1)),
                    "q": int(parameters.get("q", 1)),
                },
                timeout=300,
            )
            self._raise_for_status(response)
            payload = response.json()
            metrics = payload.get("metrics", {})
            return {
                "model": "ARIMA",
                "parameters": {"p": int(parameters.get("p", 1)), "d": int(parameters.get("d", 1)), "q": int(parameters.get("q", 1))},
                "mae": round(float(metrics.get("mae", 0.0)), 4) if metrics.get("mae") is not None else "—",
                "mse": round(float(metrics.get("rmse", 0.0)) ** 2, 4) if metrics.get("rmse") is not None else "—",
                "rmse": round(float(metrics.get("rmse", 0.0)), 4) if metrics.get("rmse") is not None else "—",
                "mape": round(float(metrics.get("mape", 0.0)), 4) if metrics.get("mape") is not None else "—",
                "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                "result_file_path": payload.get("result_file_path"),
                "raw_response": payload,
            }

        if model_name == "SARIMA Auto":
            max_p = int(parameters.get("p", 2))
            max_d = int(parameters.get("d", 1))
            max_q = int(parameters.get("q", 2))
            max_sp = int(parameters.get("seasonal_p", 1))
            max_sd = int(parameters.get("seasonal_d", 0))
            max_sq = int(parameters.get("seasonal_q", 1))
            seasonal_period = max(2, int(parameters.get("seasonal_period", 7)))
            response = self.session.post(
                self._url("/analysis/forecast/sarima/tune"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "p_values": list(range(0, max_p + 1)),
                    "d_values": list(range(0, max_d + 1)),
                    "q_values": list(range(0, max_q + 1)),
                    "seasonal_p_values": list(range(0, max_sp + 1)),
                    "seasonal_d_values": list(range(0, max_sd + 1)),
                    "seasonal_q_values": list(range(0, max_sq + 1)),
                    "seasonal_period_values": [seasonal_period],
                    "optimize_by": "rmse",
                },
                timeout=900,
            )
            self._raise_for_status(response)
            payload = response.json()
            metrics = payload.get("metrics", {})
            return {
                "model": "SARIMA Auto",
                "parameters": {
                    "best_order": payload.get("best_order"),
                    "best_seasonal_order": payload.get("best_seasonal_order"),
                    "best_parameters": payload.get("best_parameters", {}),
                    "optimize_by": payload.get("optimize_by", "rmse"),
                },
                "mae": round(float(metrics.get("mae", 0.0)), 4) if metrics.get("mae") is not None else "—",
                "mse": round(float(metrics.get("rmse", 0.0)) ** 2, 4) if metrics.get("rmse") is not None else "—",
                "rmse": round(float(metrics.get("rmse", 0.0)), 4) if metrics.get("rmse") is not None else "—",
                "mape": round(float(metrics.get("mape", 0.0)), 4) if metrics.get("mape") is not None else "—",
                "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                "result_file_path": payload.get("result_file_path"),
                "raw_response": payload,
            }

        if model_name == "SARIMA":
            response = self.session.post(
                self._url("/analysis/forecast/sarima"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "p": int(parameters.get("p", 1)),
                    "d": int(parameters.get("d", 1)),
                    "q": int(parameters.get("q", 1)),
                    "seasonal_p": int(parameters.get("seasonal_p", 1)),
                    "seasonal_d": int(parameters.get("seasonal_d", 0)),
                    "seasonal_q": int(parameters.get("seasonal_q", 1)),
                    "seasonal_period": max(2, int(parameters.get("seasonal_period", 7))),
                },
                timeout=300,
            )
            self._raise_for_status(response)
            payload = response.json()
            metrics = payload.get("metrics", {})
            return {
                "model": "SARIMA",
                "parameters": {
                    "order": payload.get("order"),
                    "seasonal_order": payload.get("seasonal_order"),
                },
                "mae": round(float(metrics.get("mae", 0.0)), 4) if metrics.get("mae") is not None else "—",
                "mse": round(float(metrics.get("rmse", 0.0)) ** 2, 4) if metrics.get("rmse") is not None else "—",
                "rmse": round(float(metrics.get("rmse", 0.0)), 4) if metrics.get("rmse") is not None else "—",
                "mape": round(float(metrics.get("mape", 0.0)), 4) if metrics.get("mape") is not None else "—",
                "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                "result_file_path": payload.get("result_file_path"),
                "raw_response": payload,
            }

        if model_name == "ARIMA Auto":
            response = self.session.post(
                self._url("/analysis/forecast/arima/tune"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "p_values": list(range(0, int(parameters.get("p", 2)) + 1)),
                    "d_values": list(range(0, int(parameters.get("d", 1)) + 1)),
                    "q_values": list(range(0, int(parameters.get("q", 2)) + 1)),
                    "optimize_by": "rmse",
                },
                timeout=300,
            )
            self._raise_for_status(response)
            payload = response.json()
            metrics = payload.get("metrics", {})
            return {
                "model": "ARIMA Auto",
                "parameters": {"best_order": payload.get("best_order")},
                "mae": round(float(metrics.get("mae", 0.0)), 4) if metrics.get("mae") is not None else "—",
                "mse": round(float(metrics.get("rmse", 0.0)) ** 2, 4) if metrics.get("rmse") is not None else "—",
                "rmse": round(float(metrics.get("rmse", 0.0)), 4) if metrics.get("rmse") is not None else "—",
                "mape": round(float(metrics.get("mape", 0.0)), 4) if metrics.get("mape") is not None else "—",
                "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                "result_file_path": payload.get("result_file_path"),
                "raw_response": payload,
            }

        if model_name == "GARCH Auto":
            max_p = max(1, int(parameters.get("p", 2)))
            max_q = max(1, int(parameters.get("q", 2)))
            response = self.session.post(
                self._url("/analysis/forecast/garch/tune"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "p_values": list(range(1, max_p + 1)),
                    "q_values": list(range(1, max_q + 1)),
                    "annualize": False,
                    "periods_per_year": 252,
                    "optimize_by": "mae",
                },
                timeout=600,
            )
            self._raise_for_status(response)
            payload = response.json()
            metrics = payload.get("metrics", {})
            return {
                "model": "GARCH Auto",
                "parameters": {
                    "best_order": payload.get("best_order"),
                    "best_parameters": payload.get("best_parameters", {}),
                    "optimize_by": payload.get("optimize_by", "mae"),
                },
                "mae": round(float(metrics.get("mae", 0.0)), 6) if metrics.get("mae") is not None else "—",
                "mse": round(float(metrics.get("mse", 0.0)), 6) if metrics.get("mse") is not None else "—",
                "rmse": round(float(metrics.get("rmse", 0.0)), 6) if metrics.get("rmse") is not None else "—",
                "mape": round(float(metrics.get("mape", 0.0)), 4) if metrics.get("mape") is not None else "—",
                "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                "result_file_path": payload.get("result_file_path"),
                "raw_response": payload,
            }

        if model_name == "GARCH":
            response = self.session.post(
                self._url("/analysis/forecast/garch"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "p": max(1, int(parameters.get("p", 1))),
                    "q": max(1, int(parameters.get("q", 1))),
                    "annualize": False,
                    "periods_per_year": 252,
                },
                timeout=300,
            )
            self._raise_for_status(response)
            payload = response.json()
            summary = payload.get("summary", {})
            mean_vol = summary.get("mean_forecast_volatility")
            return {
                "model": "GARCH",
                "parameters": {"p": max(1, int(parameters.get("p", 1))), "q": max(1, int(parameters.get("q", 1)))},
                "mae": round(float(mean_vol), 6) if mean_vol is not None else "—",
                "mse": "—",
                "rmse": "—",
                "mape": "—",
                "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                "result_file_path": payload.get("result_file_path"),
                "raw_response": payload,
            }

        if model_name == "Fuzzy First Order Auto":
            max_sets = max(7, int(parameters.get("fuzzy_sets", 30)))
            base_values = [7, 10, 15, 20, 25, 30, max_sets]
            max_sets_values = sorted({value for value in base_values if 7 <= value <= max_sets})
            if max_sets not in max_sets_values:
                max_sets_values.append(max_sets)
            response = self.session.post(
                self._url("/analysis/forecast/fuzzy-first-order/tune"),
                json={
                    "processed_file_path": processed_file_path,
                    "min_sets_values": [7],
                    "max_sets_values": max_sets_values,
                    "optimize_by": "rmse",
                },
                timeout=600,
            )
            self._raise_for_status(response)
            payload = response.json()
            metrics = payload.get("metrics", {})
            return {
                "model": "Fuzzy First Order Auto",
                "parameters": {
                    "best_parameters": payload.get("best_parameters", {}),
                    "best_fuzzy_sets_count": payload.get("best_fuzzy_sets_count"),
                    "optimize_by": payload.get("optimize_by", "rmse"),
                },
                "mae": round(float(metrics.get("mae", 0.0)), 4) if metrics.get("mae") is not None else "—",
                "mse": round(float(metrics.get("mse", 0.0)), 4) if metrics.get("mse") is not None else "—",
                "rmse": round(float(metrics.get("rmse", 0.0)), 4) if metrics.get("rmse") is not None else "—",
                "mape": round(float(metrics.get("mape", 0.0)), 4) if metrics.get("mape") is not None else "—",
                "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                "result_file_path": payload.get("result_file_path"),
                "raw_response": payload,
            }

        if model_name == "Fuzzy First Order":
            response = self.session.post(
                self._url("/analysis/forecast/fuzzy-first-order"),
                json={
                    "processed_file_path": processed_file_path,
                    "min_sets": 7,
                    "max_sets": max(7, int(parameters.get("fuzzy_sets", 25))),
                    "forecast_horizon": horizon,
                },
                timeout=300,
            )
            self._raise_for_status(response)
            payload = response.json()
            metrics = payload.get("metrics", {})
            return {
                "model": "Fuzzy First Order",
                "parameters": {"max_sets": max(7, int(parameters.get("fuzzy_sets", 25)))},
                "mae": round(float(metrics.get("mae", 0.0)), 4) if metrics.get("mae") is not None else "—",
                "mse": round(float(metrics.get("mse", 0.0)), 4) if metrics.get("mse") is not None else "—",
                "rmse": round(float(np.sqrt(metrics.get("mse", 0.0))), 4) if metrics.get("mse") is not None else "—",
                "mape": round(float(metrics.get("mape", 0.0)), 4) if metrics.get("mape") is not None else "—",
                "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                "result_file_path": payload.get("result_file_path"),
                "raw_response": payload,
            }



        if model_name == "MLP Neural Network":
            response = self.session.post(
                self._url("/analysis/forecast/mlp"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "window_size": max(1, int(parameters.get("window_size", 7))),
                    "hidden_layer_1": max(1, int(parameters.get("hidden_layer_1", 64))),
                    "hidden_layer_2": max(0, int(parameters.get("hidden_layer_2", 32))),
                    "max_iter": max(100, int(parameters.get("max_iter", 500))),
                    "random_state": int(parameters.get("random_state", parameters.get("seed", 42))),
                    "include_exogenous": bool(parameters.get("include_exogenous", True)),
                },
                timeout=900,
            )
            self._raise_for_status(response)
            payload = response.json()
            metrics = payload.get("metrics", {})
            return {
                "model": "MLP Neural Network",
                "parameters": {
                    "window_size": payload.get("window_size"),
                    "hidden_layers": payload.get("hidden_layers"),
                    "max_iter": payload.get("max_iter"),
                    "feature_columns": payload.get("feature_columns", []),
                },
                "mae": round(float(metrics.get("mae", 0.0)), 4) if metrics.get("mae") is not None else "—",
                "mse": round(float(metrics.get("mse", 0.0)), 4) if metrics.get("mse") is not None else "—",
                "rmse": round(float(metrics.get("rmse", 0.0)), 4) if metrics.get("rmse") is not None else "—",
                "mape": round(float(metrics.get("mape", 0.0)), 4) if metrics.get("mape") is not None else "—",
                "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                "result_file_path": payload.get("result_file_path"),
                "raw_response": payload,
            }

        if model_name == "LSTM Neural Network":
            response = self.session.post(
                self._url("/analysis/forecast/lstm"),
                json={
                    "processed_file_path": processed_file_path,
                    "forecast_horizon": horizon,
                    "window_size": max(1, int(parameters.get("window_size", 7))),
                    "hidden_size": max(4, int(parameters.get("hidden_size", parameters.get("lstm_hidden_size", 32)))),
                    "num_layers": max(1, int(parameters.get("num_layers", 1))),
                    "epochs": max(5, int(parameters.get("epochs", 60))),
                    "learning_rate": 0.01,
                    "random_state": int(parameters.get("random_state", parameters.get("seed", 42))),
                    "include_exogenous": bool(parameters.get("include_exogenous", True)),
                },
                timeout=1200,
            )
            self._raise_for_status(response)
            payload = response.json()
            metrics = payload.get("metrics", {})
            return {
                "model": "LSTM Neural Network",
                "parameters": {
                    "window_size": payload.get("window_size"),
                    "hidden_size": payload.get("hidden_size"),
                    "num_layers": payload.get("num_layers"),
                    "epochs": payload.get("epochs"),
                    "feature_columns": payload.get("feature_columns", []),
                },
                "mae": round(float(metrics.get("mae", 0.0)), 4) if metrics.get("mae") is not None else "—",
                "mse": round(float(metrics.get("mse", 0.0)), 4) if metrics.get("mse") is not None else "—",
                "rmse": round(float(metrics.get("rmse", 0.0)), 4) if metrics.get("rmse") is not None else "—",
                "mape": round(float(metrics.get("mape", 0.0)), 4) if metrics.get("mape") is not None else "—",
                "date": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                "result_file_path": payload.get("result_file_path"),
                "raw_response": payload,
            }

        raise ValueError("Текущий backend поддерживает Linear Regression, ARIMA, SARIMA, GARCH, Fuzzy First Order, MLP и LSTM")

    def forecast(
        self,
        *,
        processed_df: pd.DataFrame,
        processed_file_path: str | None,
        model_name: str,
        horizon: int,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        trained = self.train_model(
            processed_df=processed_df,
            processed_file_path=processed_file_path,
            model_name=model_name,
            horizon=horizon,
            parameters=parameters,
        )
        raw = trained.get("raw_response", {})
        history_df = processed_df.copy()
        if "timestamp" in history_df.columns:
            history_x = pd.to_datetime(history_df["timestamp"], errors="coerce").tolist()
        else:
            history_x = list(range(len(history_df)))

        history_column = "returns" if model_name == "GARCH" and "returns" in history_df.columns else "value"
        history_y = (
            pd.to_numeric(history_df.get(history_column, []), errors="coerce").fillna(0).tolist()
            if history_column in history_df.columns
            else [0] * len(history_df)
        )

        future_preview_df = pd.DataFrame(raw.get("future_preview") or [])
        # Обратная совместимость: старый backend возвращал только preview с holdout-прогнозом.
        # Новый backend возвращает future_preview без actual — именно его и нужно показывать
        # во вкладке «Прогнозирование».
        if future_preview_df.empty:
            future_preview_df = pd.DataFrame(raw.get("preview", []))

        if not future_preview_df.empty:
            if "timestamp" in future_preview_df.columns:
                future_x = pd.to_datetime(future_preview_df["timestamp"], errors="coerce").tolist()
            elif "target_timestamp" in future_preview_df.columns:
                future_x = pd.to_datetime(future_preview_df["target_timestamp"], errors="coerce").tolist()
            else:
                future_x = list(range(len(history_df), len(history_df) + len(future_preview_df)))

            if model_name == "GARCH" and "forecast_volatility" in future_preview_df.columns:
                forecast_column = "forecast_volatility"
            elif "predicted" in future_preview_df.columns:
                forecast_column = "predicted"
            elif "forecast" in future_preview_df.columns:
                forecast_column = "forecast"
            else:
                forecast_column = future_preview_df.columns[-1]
            forecast_values = pd.to_numeric(future_preview_df[forecast_column], errors="coerce").fillna(0).tolist()
        else:
            future_x = []
            forecast_values = []

        return {
            "history_x": history_x,
            "history_y": history_y,
            "future_x": future_x,
            "forecast": forecast_values,
            "next_value": forecast_values[0] if forecast_values else "—",
            "model": model_name,
        }

    @staticmethod
    def _normalize_experiment_from_api(row: dict[str, Any]) -> dict[str, Any]:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        parameters = row.get("parameters") if isinstance(row.get("parameters"), dict) else {}
        raw_response = row.get("raw_response") if isinstance(row.get("raw_response"), dict) else {}
        created_at = row.get("created_at") or row.get("date") or "—"
        return {
            "id": str(row.get("id", "—")),
            "created_at": created_at,
            "date": str(created_at).replace("T", " ")[:16],
            "project_id": row.get("project_id"),
            "project_name": row.get("project_name", "—"),
            "dataset_id": row.get("dataset_id"),
            "dataset_name": row.get("dataset_name", "—"),
            "model": row.get("model") or row.get("model_name") or "—",
            "parameters": parameters,
            "mae": row.get("mae", metrics.get("mae")),
            "mse": row.get("mse", metrics.get("mse")),
            "rmse": row.get("rmse", metrics.get("rmse")),
            "mape": row.get("mape", metrics.get("mape")),
            "result_file_path": row.get("result_file_path"),
            "status": row.get("status", "Успешно"),
            "raw_response": raw_response,
        }

    def list_experiments(self, project_id: int | None = None) -> list[dict[str, Any]]:
        params = {}
        if project_id is not None:
            params["project_id"] = project_id
        response = self.session.get(self._url("/experiments/"), params=params, timeout=60)
        self._raise_for_status(response)
        payload = response.json()
        rows = payload if isinstance(payload, list) else payload.get("items", [])
        return [self._normalize_experiment_from_api(row) for row in rows]

    @staticmethod
    def _json_number_or_none(value: Any) -> float | None:
        if value in (None, "", "—"):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    def create_experiment(self, record: dict[str, Any]) -> dict[str, Any]:
        mae = self._json_number_or_none(record.get("mae"))
        mse = self._json_number_or_none(record.get("mse"))
        rmse = self._json_number_or_none(record.get("rmse"))
        mape = self._json_number_or_none(record.get("mape"))
        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape}
        payload = {
            "project_id": record.get("project_id"),
            "project_name": record.get("project_name"),
            "dataset_id": record.get("dataset_id"),
            "dataset_name": record.get("dataset_name"),
            "model": record.get("model"),
            "parameters": record.get("parameters") or {},
            "metrics": metrics,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "result_file_path": record.get("result_file_path"),
            "status": record.get("status") or "Успешно",
            "raw_response": record.get("raw_response") or {},
        }
        response = self.session.post(self._url("/experiments/"), json=payload, timeout=60)
        self._raise_for_status(response)
        return self._normalize_experiment_from_api(response.json())

    def delete_experiment(self, experiment_id: str) -> None:
        response = self.session.delete(self._url(f"/experiments/{experiment_id}"), timeout=60)
        self._raise_for_status(response)

    def clear_experiments(self, project_id: int | None = None) -> None:
        params = {}
        if project_id is not None:
            params["project_id"] = project_id
        response = self.session.delete(self._url("/experiments/"), params=params, timeout=60)
        self._raise_for_status(response)

