from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QSignalBlocker, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QWidget,
    QMessageBox,
)

from src.services.app_state import AppState
from src.services.backend_service import BackendService
from src.styles.qss import APP_STYLES

APP_TITLE = "Система анализа и прогнозирования временных рядов"


def create_projects_page(state: AppState, backend: BackendService) -> QWidget:
    from src.pages.projects_page import ProjectsPage
    return ProjectsPage(state, backend)


def create_data_page(state: AppState, backend: BackendService) -> QWidget:
    from src.pages.data_page import DataPage
    return DataPage(state, backend)


def create_analysis_page(state: AppState, backend: BackendService) -> QWidget:
    from src.pages.analysis_page import AnalysisPage
    return AnalysisPage(state, backend)


def create_training_page(state: AppState, backend: BackendService) -> QWidget:
    from src.pages.training_page import TrainingPage
    return TrainingPage(state, backend)


def create_experiments_page(state: AppState, backend: BackendService) -> QWidget:
    from src.pages.experiments_page import ExperimentsPage
    return ExperimentsPage(state, backend)


def create_forecast_page(state: AppState, backend: BackendService) -> QWidget:
    from src.pages.forecast_page import ForecastPage
    return ForecastPage(state, backend)


def create_comparison_page(state: AppState, backend: BackendService) -> QWidget:
    from src.pages.comparison_page import ComparisonPage
    return ComparisonPage(state, backend)


def create_reports_page(state: AppState, backend: BackendService) -> QWidget:
    from src.pages.reports_page import ReportsPage
    return ReportsPage(state, backend)


def create_settings_page(state: AppState, backend: BackendService) -> QWidget:
    from src.pages.settings_page import SettingsPage
    return SettingsPage(state, backend)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.state = AppState()
        self.backend = BackendService(self.state)

        self.setWindowTitle(APP_TITLE)
        self.resize(1480, 920)
        self.setMinimumSize(1280, 780)

        self.page_factories: list[Callable[[AppState, BackendService], QWidget]] = [
            create_projects_page,
            create_data_page,
            create_analysis_page,
            create_training_page,
            create_experiments_page,
            create_forecast_page,
            create_comparison_page,
            create_reports_page,
            create_settings_page,
        ]
        self.created_pages: dict[int, QWidget] = {}

        self._build_toolbar()
        self._build_ui()
        self._build_statusbar()
        self.setStyleSheet(APP_STYLES)

        self.state.status_changed.connect(self.status_label.setText)
        self.state.projects_changed.connect(self.refresh_project_selector)
        self.state.config_changed.connect(self.refresh_backend_badge)
        self.project_selector.currentTextChanged.connect(self.state.set_project_by_name)
        self.state.project_changed.connect(lambda _: self.refresh_project_selector())
        self.state.project_changed.connect(lambda _: self.safe_reload_experiments())

        self.refresh_project_selector()
        self.refresh_backend_badge()



        QTimer.singleShot(0, self.safe_initialize_backend)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Главная панель")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        app_label = QLabel(APP_TITLE)
        app_label.setObjectName("appHeaderTitle")
        toolbar.addWidget(app_label)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.project_selector = QComboBox()
        self.project_selector.setMinimumWidth(220)
        toolbar.addWidget(QLabel("Активный проект:"))
        toolbar.addWidget(self.project_selector)

        self.backend_badge = QLabel()
        self.backend_badge.setObjectName("backendBadge")
        toolbar.addWidget(self.backend_badge)

        to_projects_action = QAction("Проекты", self)
        to_data_action = QAction("Данные", self)
        to_settings_action = QAction("Настройки", self)
        refresh_backend_action = QAction("Обновить backend", self)
        refresh_backend_action.triggered.connect(self.safe_initialize_backend)
        to_projects_action.triggered.connect(lambda: self.nav_list.setCurrentRow(0))
        to_data_action.triggered.connect(lambda: self.nav_list.setCurrentRow(1))
        to_settings_action.triggered.connect(lambda: self.nav_list.setCurrentRow(8))
        toolbar.addAction(to_projects_action)
        toolbar.addAction(to_data_action)
        toolbar.addAction(to_settings_action)
        toolbar.addAction(refresh_backend_action)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("sidebar")
        self.nav_list.setFixedWidth(265)
        for item in [
            "Проекты",
            "Данные",
            "Анализ",
            "Обучение моделей",
            "История экспериментов",
            "Прогнозирование",
            "Сравнение результатов",
            "Отчёты",
            "Настройки",
        ]:
            QListWidgetItem(item, self.nav_list)

        self.stack = QStackedWidget()
        for _ in self.page_factories:
            self.stack.addWidget(QWidget())

        self.nav_list.currentRowChanged.connect(self.open_page)
        main_layout.addWidget(self.nav_list)
        main_layout.addWidget(self.stack, 1)

        self.nav_list.setCurrentRow(0)

    def open_page(self, index: int) -> None:
        if index < 0 or index >= len(self.page_factories):
            return
        if index not in self.created_pages:
            try:
                page = self.page_factories[index](self.state, self.backend)
                self.stack.insertWidget(index, page)
                old_widget = self.stack.widget(index + 1)
                if old_widget is not None and old_widget not in self.created_pages.values():
                    self.stack.removeWidget(old_widget)
                    old_widget.deleteLater()
                self.created_pages[index] = page
            except Exception as exc:
                QMessageBox.critical(self, "Ошибка открытия страницы", str(exc))
                return
        self.stack.setCurrentIndex(index)

    def _build_statusbar(self) -> None:
        statusbar = QStatusBar()
        self.setStatusBar(statusbar)
        self.status_label = QLabel("Готово")
        self.global_progress = QProgressBar()
        self.global_progress.setRange(0, 100)
        self.global_progress.setValue(0)
        self.global_progress.setFixedWidth(180)
        statusbar.addWidget(QLabel("Статус:"))
        statusbar.addWidget(self.status_label)
        statusbar.addPermanentWidget(QLabel("Общий прогресс:"))
        statusbar.addPermanentWidget(self.global_progress)

    def safe_initialize_backend(self) -> None:
        try:
            self.backend.initialize()
        except Exception as exc:
            self.state.set_status(f"Backend не инициализирован: {exc}")

    def safe_reload_experiments(self) -> None:
        try:
            self.backend.reload_experiments()
        except Exception as exc:
            self.state.set_status(f"История экспериментов не обновлена: {exc}")

    def refresh_project_selector(self) -> None:
        projects = self.state.projects or [{"name": self.state.current_project}]
        with QSignalBlocker(self.project_selector):
            self.project_selector.clear()
            for project in projects:
                self.project_selector.addItem(str(project.get("name", "Без названия")))
            current_name = self.state.current_project
            idx = self.project_selector.findText(current_name)
            self.project_selector.setCurrentIndex(max(idx, 0))

    def refresh_backend_badge(self) -> None:
        if self.state.config.backend_mode == "rest":
            self.backend_badge.setText(f" backend: REST | {self.state.config.base_url} ")
        else:
            self.backend_badge.setText(" backend: MOCK ")
