from __future__ import annotations

from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from src.services.app_state import AppState
from src.services.backend_service import BackendService
from src.widgets.common import BasePage, Card


class ExperimentsPage(BasePage):
    def __init__(self, state: AppState, backend: BackendService) -> None:
        super().__init__("История экспериментов")
        self.state = state
        self.backend = backend
        self.current_rows: list[dict] = []
        self.detail_values: dict[str, QLabel] = {}

        actions_card = Card("Действия")
        actions_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Обновить")
        self.export_btn = QPushButton("Экспорт CSV")
        self.delete_btn = QPushButton("Удалить выбранный")
        self.clear_btn = QPushButton("Очистить историю проекта")
        actions_row.addWidget(self.refresh_btn)
        actions_row.addWidget(self.export_btn)
        actions_row.addWidget(self.delete_btn)
        actions_row.addWidget(self.clear_btn)
        actions_row.addStretch()
        actions_card.layout.addLayout(actions_row)
        self.add_card(actions_card)

        table_card = Card("Журнал запусков")
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels([
            "ID",
            "Дата",
            "Проект",
            "Датасет",
            "Модель",
            "MAE",
            "MSE",
            "RMSE",
            "MAPE",
            "Файл результата",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(320)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setStretchLastSection(False)
        table_card.layout.addWidget(self.table)
        self.add_card(table_card)

        details_card = Card("Детали выбранного эксперимента")
        details_grid = QGridLayout()
        details_grid.setHorizontalSpacing(18)
        details_grid.setVerticalSpacing(10)
        detail_rows = [
            ("Модель", "model"),
            ("Проект", "project_name"),
            ("Датасет", "dataset_name"),
            ("Дата", "date"),
            ("MAE", "mae"),
            ("MSE", "mse"),
            ("RMSE", "rmse"),
            ("MAPE", "mape"),
            ("Файл результата", "result_file_path"),
        ]
        for row_idx, (caption, key) in enumerate(detail_rows):
            caption_label = QLabel(f"{caption}:")
            caption_label.setObjectName("mutedText")
            value_label = QLabel("—")
            value_label.setWordWrap(True)
            details_grid.addWidget(caption_label, row_idx, 0)
            details_grid.addWidget(value_label, row_idx, 1)
            self.detail_values[key] = value_label
        details_grid.setColumnStretch(0, 0)
        details_grid.setColumnStretch(1, 1)
        details_card.layout.addLayout(details_grid)
        self.add_card(details_card)
        self.root_layout.addStretch()

        self.refresh_btn.clicked.connect(self.reload_from_backend)
        self.export_btn.clicked.connect(self.export_csv)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.clear_btn.clicked.connect(self.clear_history)
        self.table.itemSelectionChanged.connect(self.show_selected_details)
        self.state.experiments_changed.connect(self.render_table)
        self.state.project_changed.connect(lambda _: self.backend.reload_experiments())
        self.render_table()

    @staticmethod
    def _value(value: object) -> str:
        if value is None or value == "":
            return "—"
        return str(value)

    def reload_from_backend(self) -> None:
        try:
            self.backend.reload_experiments()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка истории экспериментов", str(exc))

    def render_table(self) -> None:
        self.current_rows = list(self.state.experiments)
        self.table.setRowCount(len(self.current_rows))
        for r, row in enumerate(self.current_rows):
            values = [
                row.get("id", "—"),
                row.get("date", "—"),
                row.get("project_name", "—"),
                row.get("dataset_name", "—"),
                row.get("model", "—"),
                row.get("mae", "—"),
                row.get("mse", "—"),
                row.get("rmse", "—"),
                row.get("mape", "—"),
                row.get("result_file_path", "—"),
            ]
            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(self._value(value)))
        if self.current_rows and self.table.currentRow() < 0:
            self.table.selectRow(0)
        if not self.current_rows:
            self._clear_details()

    def _clear_details(self) -> None:
        for value_label in self.detail_values.values():
            value_label.setText("—")

    def selected_experiment_id(self) -> str | None:
        row_idx = self.table.currentRow()
        if row_idx < 0 or row_idx >= len(self.current_rows):
            return None
        return str(self.current_rows[row_idx].get("id"))

    def show_selected_details(self) -> None:
        row_idx = self.table.currentRow()
        if row_idx < 0 or row_idx >= len(self.current_rows):
            self._clear_details()
            return
        row = self.current_rows[row_idx]
        for key, value_label in self.detail_values.items():
            value_label.setText(self._value(row.get(key)))

    def export_csv(self) -> None:
        if not self.current_rows:
            self.state.set_status("Нет экспериментов для экспорта")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить историю экспериментов",
            "experiments_history.csv",
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            self.backend.export_experiments_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def delete_selected(self) -> None:
        experiment_id = self.selected_experiment_id()
        if not experiment_id:
            QMessageBox.information(self, "Удаление", "Выберите эксперимент.")
            return
        answer = QMessageBox.question(self, "Удаление", f"Удалить эксперимент {experiment_id}?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.backend.delete_experiment(experiment_id)

    def clear_history(self) -> None:
        answer = QMessageBox.question(
            self,
            "Очистка истории",
            "Очистить историю экспериментов для текущего проекта?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.backend.clear_experiments()
