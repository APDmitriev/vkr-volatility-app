from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal


@dataclass
class AppConfig:
    backend_mode: str = "rest"
    base_url: str = "http://127.0.0.1:8000"
    db_path: str = "C:/VKR/app.db"
    projects_path: str = "C:/VKR/projects"
    models_path: str = "C:/VKR/models"
    reports_path: str = "C:/VKR/reports"


class AppState(QObject):
    dataset_changed = pyqtSignal(str)
    experiments_changed = pyqtSignal()
    status_changed = pyqtSignal(str)
    project_changed = pyqtSignal(str)
    projects_changed = pyqtSignal()
    config_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._config_path = Path(__file__).resolve().parents[2] / 'ui_settings.json'
        self.current_project = "Проект BTC"
        self.current_project_id: int | None = None
        self.projects: list[dict[str, Any]] = []

        self.current_dataset_name = "demo_series"
        self.current_dataset_id: int | None = None
        self.current_df: pd.DataFrame | None = None
        self.processed_df: pd.DataFrame | None = None
        self.processed_file_path: str | None = None

        self.last_analysis: dict[str, Any] = {}
        self.last_volatility_profile: dict[str, Any] = {}
        self.recommended_model: str | None = None
        self.last_forecast: dict[str, Any] = {}
        self.experiments: list[dict[str, Any]] = []
        self.config = self._load_config()

    def _load_config(self) -> AppConfig:
        if not self._config_path.exists():
            return AppConfig()
        try:
            payload = json.loads(self._config_path.read_text(encoding='utf-8'))
            return AppConfig(**{k: payload.get(k, getattr(AppConfig(), k)) for k in AppConfig.__annotations__.keys()})
        except Exception:
            return AppConfig()

    def save_config(self) -> None:
        self._config_path.write_text(
            json.dumps(asdict(self.config), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def set_projects(self, projects: list[dict[str, Any]]) -> None:
        self.projects = list(projects)
        if self.current_project_id is None and self.projects:
            first = self.projects[0]
            self.current_project_id = int(first.get('id')) if first.get('id') is not None else None
            self.current_project = str(first.get('name', self.current_project))
        self.projects_changed.emit()

    def set_dataset(self, name: str, df: pd.DataFrame, dataset_id: int | None = None) -> None:
        self.current_dataset_name = name
        self.current_df = df.copy()
        if dataset_id is not None:
            self.current_dataset_id = dataset_id
        self.dataset_changed.emit(name)

    def set_processed_dataset(self, df: pd.DataFrame, processed_file_path: str | None = None) -> None:
        self.processed_df = df.copy()
        self.current_df = df.copy()
        self.processed_file_path = processed_file_path
        self.dataset_changed.emit(self.current_dataset_name)

    def reset_dataset_processing(self) -> None:
        self.processed_df = None
        self.processed_file_path = None

    def set_status(self, text: str) -> None:
        self.status_changed.emit(text)

    def set_project(self, project_name: str, project_id: int | None = None) -> None:
        self.current_project = project_name
        if project_id is not None:
            self.current_project_id = project_id
        self.project_changed.emit(project_name)

    def set_project_by_name(self, project_name: str) -> None:
        for project in self.projects:
            if str(project.get('name')) == project_name:
                self.set_project(project_name, int(project['id']) if project.get('id') is not None else None)
                return
        self.set_project(project_name, None)

    def add_experiment(self, experiment: dict[str, Any]) -> None:
        self.experiments.insert(0, experiment)
        self.experiments = self.experiments[:50]
        self.experiments_changed.emit()

    def update_config(self, config: AppConfig) -> None:
        self.config = config
        self.save_config()
        self.config_changed.emit()
