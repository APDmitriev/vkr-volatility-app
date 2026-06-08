from __future__ import annotations

from typing import Any

import pandas as pd

from src.services.app_state import AppState
from src.services.experiment_store import ExperimentStore
from src.services.mock_backend import MockBackend
from src.services.rest_backend import RestBackend


class BackendService:
    def __init__(self, state: AppState) -> None:
        self.state = state
        self.mock = MockBackend()
        self.experiment_store = ExperimentStore()

    def _backend(self) -> MockBackend | RestBackend:
        if self.state.config.backend_mode == "rest":
            return RestBackend(self.state.config.base_url)
        return self.mock

    def initialize(self) -> None:
        if self.state.config.backend_mode == "rest":
            try:
                backend = self._backend()
                assert isinstance(backend, RestBackend)
                backend.health()
                self.load_projects()
                self.reload_experiments()
                self.state.set_status("REST backend доступен")
            except Exception as exc:
                self.state.set_status(f"REST backend недоступен: {exc}")
                self.reload_experiments()
        else:
            self.state.set_projects(
                [
                    {"id": 1, "name": "Проект BTC", "description": "Демо-проект", "created_at": "—"},
                    {"id": 2, "name": "Проект Акции", "description": "Демо-проект", "created_at": "—"},
                ]
            )
            self.reload_experiments()

    def load_projects(self) -> list[dict[str, Any]]:
        if self.state.config.backend_mode == "rest":
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            projects = backend.list_projects()
            self.state.set_projects(projects)
            self.reload_experiments()
            return projects
        return list(self.state.projects)

    def create_project(self, name: str, description: str = "") -> dict[str, Any]:
        if self.state.config.backend_mode == "rest":
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            project = backend.create_project(name, description)
            self.load_projects()
            self.state.set_project(project["name"], int(project["id"]))
            self.reload_experiments()
            self.state.set_status(f"Проект '{project['name']}' создан")
            return project
        project = {"id": len(self.state.projects) + 1, "name": name, "description": description, "created_at": "—"}
        self.state.set_projects(self.state.projects + [project])
        self.state.set_project(name, project["id"])
        self.reload_experiments()
        return project

    def delete_project_by_name(self, name: str) -> None:
        project = next((p for p in self.state.projects if str(p.get("name")) == name), None)
        if project is None:
            raise ValueError("Проект не найден в текущем списке")

        if self.state.config.backend_mode == "rest":
            project_id = project.get("id")
            if project_id is None:
                raise ValueError("У проекта нет id для удаления из backend")
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            backend.delete_project(int(project_id))
            self.load_projects()
        else:
            self.state.set_projects([p for p in self.state.projects if str(p.get("name")) != name])

        if self.state.current_project == name:
            first = self.state.projects[0] if self.state.projects else None
            if first:
                self.state.set_project(str(first.get("name")), first.get("id"))
            else:
                self.state.set_project("Демо-проект", None)
        self.reload_experiments()
        self.state.set_status(f"Проект '{name}' удалён")

    def ensure_project(self) -> tuple[int | None, str]:
        if self.state.config.backend_mode != "rest":
            return self.state.current_project_id, self.state.current_project

        if self.state.current_project_id is not None:
            return self.state.current_project_id, self.state.current_project

        projects = self.load_projects()
        for project in projects:
            if str(project.get("name")) == self.state.current_project:
                project_id = int(project["id"])
                self.state.set_project(self.state.current_project, project_id)
                self.reload_experiments()
                return project_id, self.state.current_project

        created = self.create_project(self.state.current_project, "Проект из PyQt-клиента")
        return int(created["id"]), str(created["name"])

    def ensure_dataset(self) -> tuple[str, pd.DataFrame]:
        if self.state.current_df is not None:
            return self.state.current_dataset_name, self.state.current_df.copy()
        if self.state.config.backend_mode == "mock":
            payload = self.mock.load_dataset(None, "CSV", ",", "timestamp", "value")
            df = payload["dataframe"]
            name = payload["dataset_name"]
            self.state.set_dataset(name, df)
            return name, df
        raise ValueError("Сначала загрузите датасет на странице 'Данные'")


    def list_datasets(self) -> list[dict[str, Any]]:
        if self.state.config.backend_mode == "rest":
            project_id, _ = self.ensure_project()
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            return backend.list_datasets(project_id=project_id)
        name = self.state.current_dataset_name or "demo_series"
        return [
            {
                "id": self.state.current_dataset_id or 1,
                "name": name,
                "rows_count": len(self.state.current_df) if self.state.current_df is not None else 0,
                "created_at": "—",
                "file_path": "mock",
            }
        ]

    def open_existing_dataset(self, dataset: dict[str, Any]) -> dict[str, Any]:
        if self.state.config.backend_mode == "rest":
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            payload = backend.open_existing_dataset(dataset)
            self.state.set_dataset(payload["dataset_name"], payload["dataframe"], payload.get("dataset_id"))
            self.state.reset_dataset_processing()
            self.state.set_status(f"Выбран датасет '{payload['dataset_name']}'")
            return payload

        if self.state.current_df is not None:
            return {
                "dataset_name": self.state.current_dataset_name,
                "preview": self.state.current_df.head(200),
                "dataframe": self.state.current_df.copy(),
                "dataset_id": self.state.current_dataset_id,
            }
        payload = self.mock.load_dataset(None, "CSV", ",", "timestamp", "value")
        self.state.set_dataset(payload["dataset_name"], payload["dataframe"])
        return payload

    def load_dataset(
        self,
        file_path: str | None,
        file_type: str,
        delimiter: str,
        time_column: str | None,
        target_column: str | None,
    ) -> dict[str, Any]:
        if self.state.config.backend_mode == "rest":
            project_id, _ = self.ensure_project()
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            payload = backend.load_dataset(
                file_path=file_path,
                file_type=file_type,
                delimiter=delimiter,
                time_column=time_column,
                target_column=target_column,
                project_id=project_id,
            )
            df = payload["dataframe"]
            self.state.set_dataset(payload["dataset_name"], df, payload.get("dataset_id"))
            self.state.reset_dataset_processing()
            self.state.set_status(f"Датасет '{payload['dataset_name']}' загружен на backend")
            return payload

        payload = self.mock.load_dataset(file_path, file_type, delimiter, time_column, target_column)
        df = payload["dataframe"]
        self.state.set_dataset(payload["dataset_name"], df)
        self.state.reset_dataset_processing()
        self.state.set_status(f"Датасет '{payload['dataset_name']}' загружен")
        return payload

    def preprocess_dataset(self, df: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        if self.state.config.backend_mode == "rest":
            if self.state.current_dataset_id is None:
                raise ValueError("Сначала загрузите датасет")
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            payload = backend.preprocess_dataset(
                dataset_id=self.state.current_dataset_id,
                date_column=kwargs["time_column"],
                value_column=kwargs["target_column"],
                missing_strategy=kwargs.get("missing_strategy", "Удалить пропуски"),
                returns_method=kwargs.get("returns_method", "simple"),
                drop_duplicate_rows=kwargs.get("drop_duplicate_rows", True),
                drop_duplicate_timestamps=kwargs.get("drop_duplicate_timestamps", True),
                sort_by_date=kwargs.get("sort_by_date", True),
            )
            out_df = payload["dataframe"]
            self.state.set_processed_dataset(out_df, payload.get("processed_file_path"))
            self.state.set_status("Предобработка выполнена на backend")
            return payload

        payload = self.mock.preprocess_dataset(df, **kwargs)
        self.state.set_processed_dataset(payload["dataframe"], None)
        self.state.set_status("Предобработка завершена")
        return payload

    def ensure_preprocessed(self, *, time_column: str, target_column: str) -> tuple[pd.DataFrame, str | None]:
        if self.state.config.backend_mode == "rest":
            if self.state.processed_df is not None and self.state.processed_file_path:
                return self.state.processed_df.copy(), self.state.processed_file_path
            _, current_df = self.ensure_dataset()
            payload = self.preprocess_dataset(
                current_df,
                time_column=time_column,
                target_column=target_column,
                missing_strategy="Удалить пропуски",
                returns_method="simple",
                drop_duplicate_rows=True,
                drop_duplicate_timestamps=True,
                sort_by_date=True,
            )
            return payload["dataframe"].copy(), payload.get("processed_file_path")

        _, df = self.ensure_dataset()
        return df.copy(), None

    def analyze_dataset(self, df: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        if self.state.processed_df is None:
            raise ValueError("Для анализа нужен предобработанный датасет")

        if self.state.config.backend_mode == "rest":
            processed_df = self.state.processed_df.copy()
            processed_file_path = self.state.processed_file_path
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            payload = backend.analyze_dataset(
                processed_df=processed_df,
                processed_file_path=processed_file_path,
                time_column=kwargs["time_column"],
                target_column=kwargs["target_column"],
            )
        else:
            payload = self._backend().analyze_dataset(self.state.processed_df.copy(), **kwargs)
        self.state.last_analysis = payload
        profile = payload.get("volatility_profile", {}) if isinstance(payload, dict) else {}
        self.state.last_volatility_profile = profile if isinstance(profile, dict) else {}
        recommended_model = self.state.last_volatility_profile.get("recommended_model")
        self.state.recommended_model = str(recommended_model) if recommended_model else None
        self.state.set_status("Анализ завершён")
        return payload

    def tune_model_parameters(self, df: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        if self.state.config.backend_mode == "rest":
            processed_df, processed_file_path = self.ensure_preprocessed(
                time_column=kwargs["time_column"],
                target_column=kwargs["target_column"],
            )
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            payload = backend.tune_model_parameters(
                processed_file_path=processed_file_path,
                model_name=kwargs["model_name"],
                horizon=kwargs["horizon"],
                parameters=kwargs.get("parameters", {}),
            )
        else:
            backend = self._backend()
            if hasattr(backend, "tune_model_parameters"):
                payload = backend.tune_model_parameters(df, **kwargs)
            else:
                payload = {
                    "model": kwargs["model_name"],
                    "best_parameters": kwargs.get("parameters", {}),
                    "metrics": {},
                    "optimize_by": "mock",
                    "candidates_count": 0,
                }
        self.state.set_status("Параметры обучения подобраны и подставлены в форму")
        return payload

    def train_model(self, df: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        if self.state.config.backend_mode == "rest":
            processed_df, processed_file_path = self.ensure_preprocessed(
                time_column=kwargs["time_column"],
                target_column=kwargs["target_column"],
            )
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            payload = backend.train_model(
                processed_df=processed_df,
                processed_file_path=processed_file_path,
                model_name=kwargs["model_name"],
                horizon=kwargs["horizon"],
                parameters=kwargs.get("parameters", {}),
            )
        else:
            payload = self._backend().train_model(df, **kwargs)

        if self.state.config.backend_mode == "rest":
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            record = self.experiment_store.build_record(payload, state=self.state)
            stored = backend.create_experiment(record)
        else:
            stored = self.experiment_store.add(payload, state=self.state)

        self.reload_experiments()
        self.state.set_status(f"Обучение завершено. Эксперимент {stored['id']} сохранён в истории")
        return stored

    def forecast(self, df: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
        if self.state.config.backend_mode == "rest":
            processed_df, processed_file_path = self.ensure_preprocessed(
                time_column=kwargs["time_column"],
                target_column=kwargs["target_column"],
            )
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            payload = backend.forecast(
                processed_df=processed_df,
                processed_file_path=processed_file_path,
                model_name=kwargs["model_name"],
                horizon=kwargs["horizon"],
                parameters=kwargs.get("parameters", {}),
            )
        else:
            payload = self._backend().forecast(df, **kwargs)
        self.state.last_forecast = payload
        self.state.set_status("Прогноз построен")
        return payload

    def reload_experiments(self) -> list[dict[str, Any]]:
        if self.state.config.backend_mode == "rest":
            try:
                backend = self._backend()
                assert isinstance(backend, RestBackend)
                rows = backend.list_experiments(project_id=self.state.current_project_id)
            except Exception as exc:
                self.state.set_status(f"Не удалось загрузить историю экспериментов из backend: {exc}")
                rows = []
        else:
            rows = self.experiment_store.list(
                project_id=self.state.current_project_id,
                project_name=self.state.current_project,
            )
        self.state.experiments = rows
        self.state.experiments_changed.emit()
        return rows

    def get_experiments(self) -> list[dict[str, Any]]:
        # Важно: этот метод должен только возвращать уже загруженную историю.
        # Если здесь вызывать reload_experiments(), то при пустой истории возникает
        # цикл: страница -> get_experiments() -> reload_experiments() ->
        # experiments_changed -> страница -> get_experiments() ...
        # На Windows это проявляется как RecursionError и нативное падение PyQt
        # с кодом 0xC0000409 при открытии страницы «Обучение моделей».
        return list(self.state.experiments)

    def delete_experiment(self, experiment_id: str) -> None:
        if self.state.config.backend_mode == "rest":
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            backend.delete_experiment(experiment_id)
        else:
            self.experiment_store.delete(experiment_id)
        self.reload_experiments()
        self.state.set_status(f"Эксперимент {experiment_id} удалён")

    def clear_experiments(self) -> None:
        if self.state.config.backend_mode == "rest":
            backend = self._backend()
            assert isinstance(backend, RestBackend)
            backend.clear_experiments(project_id=self.state.current_project_id)
        else:
            self.experiment_store.clear(
                project_id=self.state.current_project_id,
                project_name=self.state.current_project,
            )
        self.reload_experiments()
        self.state.set_status("История экспериментов очищена")

    def export_experiments_csv(self, path: str) -> None:
        rows = self.get_experiments()
        self.experiment_store.export_csv(path, rows)
        self.state.set_status(f"История экспериментов экспортирована: {path}")
