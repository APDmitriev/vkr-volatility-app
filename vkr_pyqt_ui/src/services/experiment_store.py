from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


class ExperimentStore:
    """Локальное постоянное хранилище истории экспериментов PyQt-клиента.

    Backend ВКР выполняет расчёты, а история запусков фиксируется на стороне UI,
    потому что в текущем API нет отдельного endpoint для журнала экспериментов.
    """

    def __init__(self, file_path: Path | None = None) -> None:
        self.file_path = file_path or Path(__file__).resolve().parents[2] / "ui_experiments.json"
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.file_path.exists():
            return []
        try:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, list) else []
        except Exception:
            return []

    def _write_all(self, rows: list[dict[str, Any]]) -> None:
        self.file_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, default=self._json_default),
            encoding="utf-8",
        )

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    @classmethod
    def _make_json_safe(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): cls._make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._make_json_safe(v) for v in value]
        return str(value)

    @staticmethod
    def _metric(payload: dict[str, Any], name: str) -> Any:
        if name in payload:
            return payload.get(name)
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            return metrics.get(name)
        return None

    def build_record(self, experiment: dict[str, Any], *, state: Any) -> dict[str, Any]:
        """Приводит результат расчёта модели к единому формату журнала экспериментов."""
        raw = self._make_json_safe(experiment)
        raw_response = raw.get("raw_response", {}) if isinstance(raw.get("raw_response"), dict) else {}

        result_file_path = raw.get("result_file_path") or raw_response.get("result_file_path")
        model = raw.get("model") or raw_response.get("model") or "—"
        parameters = raw.get("parameters") or {}

        return {
            "id": raw.get("id") or uuid4().hex[:12],
            "created_at": raw.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date": raw.get("date") or datetime.now().strftime("%d.%m.%Y %H:%M"),
            "project_id": state.current_project_id,
            "project_name": state.current_project,
            "dataset_id": state.current_dataset_id,
            "dataset_name": state.current_dataset_name,
            "model": model,
            "parameters": parameters,
            "mae": self._metric(raw, "mae"),
            "mse": self._metric(raw, "mse"),
            "rmse": self._metric(raw, "rmse"),
            "mape": self._metric(raw, "mape"),
            "result_file_path": result_file_path,
            "status": raw.get("status") or "Успешно",
            "raw_response": raw_response,
        }

    def add(self, experiment: dict[str, Any], *, state: Any) -> dict[str, Any]:
        record = self.build_record(experiment, state=state)
        rows = self._read_all()
        rows.insert(0, record)
        self._write_all(rows[:500])
        return record

    def list(self, *, project_id: int | None = None, project_name: str | None = None) -> list[dict[str, Any]]:
        rows = self._read_all()
        if project_id is not None:
            filtered = [row for row in rows if row.get("project_id") == project_id]
            if filtered:
                return filtered
        if project_name:
            return [row for row in rows if row.get("project_name") == project_name]
        return rows

    def delete(self, experiment_id: str) -> None:
        rows = [row for row in self._read_all() if row.get("id") != experiment_id]
        self._write_all(rows)

    def clear(self, *, project_id: int | None = None, project_name: str | None = None) -> None:
        rows = self._read_all()
        if project_id is not None:
            rows = [row for row in rows if row.get("project_id") != project_id]
        elif project_name:
            rows = [row for row in rows if row.get("project_name") != project_name]
        else:
            rows = []
        self._write_all(rows)

    def export_csv(self, path: str | Path, rows: list[dict[str, Any]]) -> None:
        columns = [
            "id",
            "date",
            "project_name",
            "dataset_name",
            "model",
            "parameters",
            "mae",
            "mse",
            "rmse",
            "mape",
            "result_file_path",
            "status",
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, delimiter=";")
            writer.writeheader()
            for row in rows:
                out = dict(row)
                if isinstance(out.get("parameters"), dict):
                    out["parameters"] = json.dumps(out["parameters"], ensure_ascii=False)
                writer.writerow({key: out.get(key, "") for key in columns})
