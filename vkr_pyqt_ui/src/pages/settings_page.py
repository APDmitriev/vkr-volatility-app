from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLineEdit, QMessageBox, QPushButton

from src.services.app_state import AppConfig, AppState
from src.services.backend_service import BackendService
from src.widgets.common import BasePage, Card


class SettingsPage(BasePage):
    def __init__(self, state: AppState, backend: BackendService) -> None:
        super().__init__("Настройки")
        self.state = state
        self.backend = backend

        paths_card = Card("Пути")
        paths_form = QFormLayout()
        self.db_path_edit = QLineEdit(state.config.db_path)
        self.projects_path_edit = QLineEdit(state.config.projects_path)
        self.models_path_edit = QLineEdit(state.config.models_path)
        self.reports_path_edit = QLineEdit(state.config.reports_path)
        paths_form.addRow("База данных:", self.db_path_edit)
        paths_form.addRow("Каталог проектов:", self.projects_path_edit)
        paths_form.addRow("Каталог моделей:", self.models_path_edit)
        paths_form.addRow("Каталог отчётов:", self.reports_path_edit)
        paths_card.layout.addLayout(paths_form)
        self.add_card(paths_card)

        system_card = Card("Система")
        system_form = QFormLayout()
        self.backend_mode_combo = QComboBox(); self.backend_mode_combo.addItems(["mock", "rest"]); self.backend_mode_combo.setCurrentText(state.config.backend_mode)
        self.base_url_edit = QLineEdit(state.config.base_url)
        self.logging_combo = QComboBox(); self.logging_combo.addItems(["Включено", "Отключено"])
        self.theme_combo = QComboBox(); self.theme_combo.addItems(["Светлая", "Тёмная"])
        system_form.addRow("Режим backend:", self.backend_mode_combo)
        system_form.addRow("Base URL:", self.base_url_edit)
        system_form.addRow("Логирование:", self.logging_combo)
        system_form.addRow("Тема:", self.theme_combo)
        system_card.layout.addLayout(system_form)

        hint = QLineEdit()
        hint.setReadOnly(True)
        hint.setText("В режиме rest настройки сохраняются в ui_settings.json. После перезапуска UI режим и Base URL не потеряются.")
        system_card.layout.addWidget(hint)

        buttons_row = QHBoxLayout()
        self.test_connection_btn = QPushButton("Проверить backend")
        self.save_settings_btn = QPushButton("Сохранить настройки")
        buttons_row.addWidget(self.test_connection_btn)
        buttons_row.addWidget(self.save_settings_btn)
        buttons_row.addStretch()
        system_card.layout.addLayout(buttons_row)
        self.add_card(system_card)
        self.root_layout.addStretch()

        self.save_settings_btn.clicked.connect(self.save_settings)
        self.test_connection_btn.clicked.connect(self.test_connection)

    def _build_config(self) -> AppConfig:
        return AppConfig(
            backend_mode=self.backend_mode_combo.currentText(),
            base_url=self.base_url_edit.text().strip() or "http://127.0.0.1:8000",
            db_path=self.db_path_edit.text().strip(),
            projects_path=self.projects_path_edit.text().strip(),
            models_path=self.models_path_edit.text().strip(),
            reports_path=self.reports_path_edit.text().strip(),
        )

    def save_settings(self) -> None:
        self.state.update_config(self._build_config())
        self.backend.initialize()
        QMessageBox.information(self, "Настройки", "Настройки сохранены в ui_settings.json")

    def test_connection(self) -> None:
        old_config = self.state.config
        try:
            self.state.update_config(self._build_config())
            self.backend.initialize()
            QMessageBox.information(self, "Backend", "Проверка завершена. Смотри строку статуса внизу окна.")
        except Exception as exc:
            self.state.update_config(old_config)
            QMessageBox.critical(self, "Ошибка подключения", str(exc))
