from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from src.services.app_state import AppState
from src.services.backend_service import BackendService
from src.widgets.charts import MplCanvas
from src.widgets.common import BasePage, Card


class DataPage(BasePage):
    def __init__(self, state: AppState, backend: BackendService) -> None:
        super().__init__("Данные")
        self.state = state
        self.backend = backend
        self.datasets: list[dict[str, Any]] = []

        datasets_card = Card("Датасеты текущего проекта")
        datasets_actions = QHBoxLayout()
        self.refresh_datasets_btn = QPushButton("Обновить список")
        self.open_dataset_btn = QPushButton("Открыть выбранный")
        self.refresh_datasets_btn.setMinimumWidth(160)
        self.open_dataset_btn.setMinimumWidth(170)
        datasets_actions.addWidget(self.refresh_datasets_btn)
        datasets_actions.addWidget(self.open_dataset_btn)
        datasets_actions.addStretch()
        datasets_card.layout.addLayout(datasets_actions)

        self.datasets_table = QTableWidget(0, 5)
        self.datasets_table.setHorizontalHeaderLabels(["ID", "Название", "Строк", "Дата", "Файл"])
        self.datasets_table.verticalHeader().setVisible(False)
        self.datasets_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.datasets_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.datasets_table.setAlternatingRowColors(True)
        self.datasets_table.setMinimumHeight(120)
        self.datasets_table.horizontalHeader().setStretchLastSection(True)
        datasets_card.layout.addWidget(self.datasets_table)
        self.add_card(datasets_card)

        upload_card = Card("Загрузка нового датасета")
        upload_grid = QGridLayout()
        upload_grid.setHorizontalSpacing(14)
        upload_grid.setVerticalSpacing(14)
        upload_grid.setColumnStretch(1, 1)
        upload_grid.setColumnStretch(3, 1)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Выберите CSV или XLSX файл")
        self.browse_btn = QPushButton("Выбрать файл")
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(["CSV", "XLSX"])
        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItems([",", ";", "tab"])
        self.time_column_combo = QComboBox()
        self.target_column_combo = QComboBox()
        self.preview_btn = QPushButton("Загрузить и показать")

        self.browse_btn.setMinimumWidth(160)
        self.preview_btn.setMinimumWidth(210)
        self.time_column_combo.setMinimumWidth(220)
        self.target_column_combo.setMinimumWidth(220)
        self.file_type_combo.setMinimumWidth(130)
        self.delimiter_combo.setMinimumWidth(130)

        upload_grid.addWidget(QLabel("Файл:"), 0, 0)
        upload_grid.addWidget(self.file_path_edit, 0, 1, 1, 3)
        upload_grid.addWidget(self.browse_btn, 0, 4)

        upload_grid.addWidget(QLabel("Тип файла:"), 1, 0)
        upload_grid.addWidget(self.file_type_combo, 1, 1)
        upload_grid.addWidget(QLabel("Разделитель:"), 1, 2)
        upload_grid.addWidget(self.delimiter_combo, 1, 3)

        upload_grid.addWidget(QLabel("Колонка времени:"), 2, 0)
        upload_grid.addWidget(self.time_column_combo, 2, 1)
        upload_grid.addWidget(QLabel("Колонка значений:"), 2, 2)
        upload_grid.addWidget(self.target_column_combo, 2, 3)
        upload_grid.addWidget(self.preview_btn, 2, 4)

        upload_card.layout.addLayout(upload_grid)
        self.add_card(upload_card)

        middle_splitter = QSplitter(Qt.Orientation.Horizontal)

        table_card = Card("Таблица данных")
        self.data_table = QTableWidget(0, 0)
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setMinimumHeight(220)
        table_card.layout.addWidget(self.data_table)

        preprocess_card = Card("Параметры предобработки")
        preprocess_card.setMinimumWidth(600)
        preprocess_grid = QGridLayout()
        preprocess_grid.setHorizontalSpacing(12)
        preprocess_grid.setVerticalSpacing(10)
        preprocess_grid.setColumnStretch(0, 0)
        preprocess_grid.setColumnStretch(1, 1)
        preprocess_grid.setColumnMinimumWidth(0, 170)
        preprocess_grid.setColumnMinimumWidth(1, 330)

        self.missing_combo = QComboBox()
        self.missing_combo.addItems(["Удалить пропуски", "Заполнить предыдущим", "Заполнить следующим"])
        self.drop_rows_check = QCheckBox("Удалять полностью дублирующиеся строки")
        self.drop_rows_check.setChecked(True)
        self.drop_timestamps_check = QCheckBox("Удалять повторяющиеся timestamps")
        self.drop_timestamps_check.setChecked(True)
        self.sort_check = QCheckBox("Сортировать по времени")
        self.sort_check.setChecked(True)
        self.apply_preprocess_btn = QPushButton("Применить предобработку")

        self.missing_combo.setMinimumWidth(330)
        self.apply_preprocess_btn.setMinimumWidth(330)

        preprocess_grid.addWidget(QLabel("Пропуски:"), 0, 0)
        preprocess_grid.addWidget(self.missing_combo, 0, 1)
        preprocess_grid.addWidget(self.drop_rows_check, 1, 0, 1, 2)
        preprocess_grid.addWidget(self.drop_timestamps_check, 2, 0, 1, 2)
        preprocess_grid.addWidget(self.sort_check, 3, 0, 1, 2)
        preprocess_grid.addWidget(self.apply_preprocess_btn, 4, 0, 1, 2)
        preprocess_card.layout.addLayout(preprocess_grid)
        preprocess_card.layout.addStretch()

        middle_splitter.addWidget(table_card)
        middle_splitter.addWidget(preprocess_card)
        middle_splitter.setChildrenCollapsible(False)
        middle_splitter.setStretchFactor(0, 5)
        middle_splitter.setStretchFactor(1, 3)
        middle_splitter.setSizes([700, 620])
        self.add_card(middle_splitter)

        chart_card = Card("График временного ряда")
        self.series_chart = MplCanvas(300)
        chart_card.layout.addWidget(self.series_chart)
        self.add_card(chart_card)

        self.dataset_status_label = QLabel()

        self.refresh_datasets_btn.clicked.connect(self.refresh_datasets)
        self.open_dataset_btn.clicked.connect(self.open_selected_dataset)
        self.browse_btn.clicked.connect(self.choose_file)
        self.preview_btn.clicked.connect(self.preview_dataset)
        self.apply_preprocess_btn.clicked.connect(self.apply_preprocessing)
        self.state.project_changed.connect(self.on_project_changed)
        self.refresh_datasets(silent=True)
        self.root_layout.addStretch()

    def refresh_datasets(self, silent: bool = False) -> None:
        try:
            rows = self.backend.list_datasets()
            self.datasets = rows
            self.populate_datasets_table(rows)
            self.state.set_status(f"Загружен список датасетов: {len(rows)}")
        except Exception as exc:
            self.datasets = []
            self.populate_datasets_table([])
            self.state.set_status(f"Не удалось загрузить список датасетов: {exc}")
            if not silent:
                QMessageBox.critical(self, "Ошибка загрузки списка датасетов", str(exc))

    def populate_datasets_table(self, rows: list[dict[str, Any]]) -> None:
        self.datasets_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.get("id", ""),
                row.get("name", ""),
                row.get("rows_count", ""),
                str(row.get("created_at", ""))[:19],
                row.get("file_path", ""),
            ]
            for col_index, value in enumerate(values):
                self.datasets_table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        self.datasets_table.resizeColumnsToContents()
        self.datasets_table.horizontalHeader().setStretchLastSection(True)

    def open_selected_dataset(self) -> None:
        row_index = self.datasets_table.currentRow()
        if row_index < 0:
            QMessageBox.warning(self, "Датасет не выбран", "Выберите датасет в таблице")
            return
        if row_index >= len(self.datasets):
            QMessageBox.warning(self, "Ошибка выбора", "Не удалось определить выбранный датасет")
            return
        try:
            payload = self.backend.open_existing_dataset(self.datasets[row_index])
            df = payload["dataframe"]
            preview = payload["preview"]
            self._refresh_column_combos(df)
            date_column = payload.get("date_column")
            value_column = payload.get("value_column")
            if date_column:
                self.time_column_combo.setCurrentText(str(date_column))
            if value_column:
                self.target_column_combo.setCurrentText(str(value_column))
            self.populate_table(preview if isinstance(preview, pd.DataFrame) else pd.DataFrame(preview))
            self.draw_series(df)
            dataset_id = payload.get("dataset_id")
            self.state.set_status(
                f"Выбран исходный датасет: {payload.get('dataset_name', '—')}"
                + (f", id={dataset_id}" if dataset_id else "")
            )
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка открытия датасета", str(exc))

    def on_project_changed(self, _name: str) -> None:
        self.clear_dataset_view()
        self.refresh_datasets(silent=True)

    def clear_dataset_view(self) -> None:
        self.datasets = []
        self.datasets_table.setRowCount(0)
        self.data_table.setRowCount(0)
        self.data_table.setColumnCount(0)
        self.state.set_status("Проект изменён")

    def choose_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл данных",
            "",
            "Data files (*.csv *.xlsx *.xls);;All files (*.*)",
        )
        if file_path:
            self.file_path_edit.setText(file_path)
            suffix = Path(file_path).suffix.lower()
            self.file_type_combo.setCurrentText("XLSX" if suffix in {".xlsx", ".xls"} else "CSV")
            self._fill_columns_from_local_file(file_path)

    def _fill_columns_from_local_file(self, file_path: str) -> None:
        try:
            df = self._read_local_dataframe(file_path, self.file_type_combo.currentText(), self.delimiter_combo.currentText())
            self._refresh_column_combos(df)
        except Exception:
            # Не показываем ошибку здесь: файл может быть временно недоступен, основная проверка будет при загрузке.
            pass

    def preview_dataset(self, initial: bool = False) -> None:
        try:
            payload = self.backend.load_dataset(
                self.file_path_edit.text().strip() or None,
                self.file_type_combo.currentText(),
                self.delimiter_combo.currentText(),
                self.time_column_combo.currentText() or None,
                self.target_column_combo.currentText() or None,
            )
            df = payload["dataframe"]
            preview = payload["preview"]
            self._refresh_column_combos(df)
            self.populate_table(preview if isinstance(preview, pd.DataFrame) else pd.DataFrame(preview))
            self.draw_series(df)
            dataset_id = payload.get("dataset_id")
            if dataset_id:
                self.state.set_status(f"Исходный датасет загружен на backend, id={dataset_id}")
                self.refresh_datasets(silent=True)
            else:
                self.state.set_status("Демо-датасет загружен")
        except Exception as exc:
            if not initial:
                QMessageBox.critical(self, "Ошибка загрузки", str(exc))

    def _refresh_column_combos(self, df: pd.DataFrame) -> None:
        columns = [str(c) for c in df.columns]
        current_time = self.time_column_combo.currentText()
        current_target = self.target_column_combo.currentText()
        self.time_column_combo.clear()
        self.target_column_combo.clear()
        self.time_column_combo.addItems(columns)
        self.target_column_combo.addItems(columns)
        if not columns:
            return
        preferred_time = next(
            (c for c in columns if c.lower() in {"timestamp", "date", "datetime"}),
            columns[0],
        )
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        preferred_target = next(
            (c for c in numeric_cols if c.lower() in {"value", "close", "target", "price"}),
            numeric_cols[0] if numeric_cols else (columns[1] if len(columns) > 1 else columns[0]),
        )
        self.time_column_combo.setCurrentText(current_time if current_time in columns else preferred_time)
        self.target_column_combo.setCurrentText(current_target if current_target in columns else preferred_target)

    def apply_preprocessing(self) -> None:
        if self.state.current_df is None:
            QMessageBox.warning(self, "Нет данных", "Сначала выберите или загрузите датасет")
            return
        try:
            payload = self.backend.preprocess_dataset(
                self.state.current_df,
                time_column=self.time_column_combo.currentText(),
                target_column=self.target_column_combo.currentText(),
                missing_strategy=self.missing_combo.currentText(),
                returns_method="simple",
                drop_duplicate_rows=self.drop_rows_check.isChecked(),
                drop_duplicate_timestamps=self.drop_timestamps_check.isChecked(),
                sort_by_date=self.sort_check.isChecked(),
            )
            df = payload["dataframe"]
            self._refresh_column_combos(df)
            if "timestamp" in df.columns:
                self.time_column_combo.setCurrentText("timestamp")
            if "value" in df.columns:
                self.target_column_combo.setCurrentText("value")
            self.populate_table(payload["preview"] if isinstance(payload["preview"], pd.DataFrame) else pd.DataFrame(payload["preview"]))
            self.draw_series(df)
            self.state.set_status("Предобработка выполнена. Датасет доступен для анализа")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка предобработки", str(exc))

    def populate_table(self, df: pd.DataFrame) -> None:
        max_rows = min(len(df), 200)
        self.data_table.setColumnCount(len(df.columns))
        self.data_table.setRowCount(max_rows)
        self.data_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(max_rows):
            for c, column in enumerate(df.columns):
                self.data_table.setItem(r, c, QTableWidgetItem(str(df.iloc[r][column])))
        self.data_table.resizeColumnsToContents()
        self.data_table.horizontalHeader().setStretchLastSection(True)

    def draw_series(self, df: pd.DataFrame) -> None:
        if df.empty:
            self.series_chart.clear("График временного ряда", "Нет данных")
            return
        time_col = self.time_column_combo.currentText()
        target_col = self.target_column_combo.currentText()
        if not target_col or target_col not in df.columns:
            self.series_chart.clear("График временного ряда", "Не выбрана колонка значений")
            return
        x = df[time_col] if time_col in df.columns else range(len(df))
        y = pd.to_numeric(df[target_col], errors="coerce")
        mask = y.notna()
        if not mask.any():
            self.series_chart.clear("График временного ряда", "Нет числовых значений")
            return
        if hasattr(x, "loc"):
            x = x.loc[mask]
        y = y.loc[mask]
        self.series_chart.plot_series(x, y, title="Временной ряд", label=target_col)

    @staticmethod
    def _read_local_dataframe(file_path: str, file_type: str, delimiter: str) -> pd.DataFrame:
        path = Path(file_path)
        if path.suffix.lower() in {".xlsx", ".xls"} or file_type.lower() == "xlsx":
            return pd.read_excel(path)
        sep = "\t" if delimiter == "tab" else delimiter
        return pd.read_csv(path, sep=sep)
