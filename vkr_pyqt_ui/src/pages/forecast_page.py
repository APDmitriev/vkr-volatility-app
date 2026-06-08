from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def safe_to_datetime(values):
    try:
        return pd.to_datetime(values)
    except (ValueError, TypeError, OverflowError):
        return values

from PyQt6.QtWidgets import QFileDialog, QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView

from src.services.app_state import AppState
from src.services.backend_service import BackendService
from src.widgets.charts import MplCanvas
from src.widgets.common import BasePage, Card


class ForecastPage(BasePage):
    def __init__(self, state: AppState, backend: BackendService) -> None:
        super().__init__("Прогнозирование")
        self.state = state
        self.backend = backend

        controls_card = Card("Параметры прогноза")
        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Модель:"))
        self.model_combo = QComboBox(); self.model_combo.addItems(["Linear Regression", "ARIMA", "SARIMA", "GARCH", "Fuzzy First Order", "MLP Neural Network", "LSTM Neural Network"])
        controls_row.addWidget(self.model_combo)
        controls_row.addWidget(QLabel("Данные:"))
        self.dataset_combo = QComboBox(); self.dataset_combo.addItems([self.state.current_dataset_name])
        controls_row.addWidget(self.dataset_combo)
        controls_row.addWidget(QLabel("Горизонт:"))
        self.horizon_spin = QSpinBox(); self.horizon_spin.setRange(1, 365); self.horizon_spin.setValue(10)
        controls_row.addWidget(self.horizon_spin)
        self.run_forecast_btn = QPushButton("Построить прогноз")
        controls_row.addWidget(self.run_forecast_btn)
        controls_row.addStretch()
        controls_card.layout.addLayout(controls_row)
        self.add_card(controls_card)

        chart_card = Card("График фактических и прогнозных значений")
        self.forecast_chart = MplCanvas(320)
        chart_card.layout.addWidget(self.forecast_chart)
        self.add_card(chart_card)

        table_card = Card("Таблица прогноза")
        self.forecast_table = QTableWidget(0, 2)
        self.forecast_table.setHorizontalHeaderLabels(["timestamp", "forecast"])
        self.forecast_table.verticalHeader().setVisible(False)
        self.forecast_table.setMinimumHeight(360)
        self.forecast_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.forecast_table.horizontalHeader().setStretchLastSection(False)
        table_card.layout.addWidget(self.forecast_table)
        self.add_card(table_card)

        bottom_row = QHBoxLayout()
        next_value_card = Card("Следующее прогнозное значение")
        self.next_value_label = QLabel("—")
        self.next_value_label.setObjectName("bigMetric")
        next_value_card.layout.addWidget(self.next_value_label)
        export_card = Card("Экспорт")
        export_buttons = QHBoxLayout()
        self.export_csv_btn = QPushButton("CSV")
        self.export_excel_btn = QPushButton("Excel")
        export_buttons.addWidget(self.export_csv_btn)
        export_buttons.addWidget(self.export_excel_btn)
        export_buttons.addStretch()
        export_card.layout.addLayout(export_buttons)
        bottom_row.addWidget(next_value_card)
        bottom_row.addWidget(export_card)
        self.root_layout.addLayout(bottom_row)

        self.current_forecast_df: pd.DataFrame | None = None

        self.run_forecast_btn.clicked.connect(self.run_forecast)
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_excel_btn.clicked.connect(self.export_excel)
        self.state.dataset_changed.connect(self.on_dataset_changed)
        self.root_layout.addStretch()

    def on_dataset_changed(self, name: str) -> None:
        if self.dataset_combo.findText(name) == -1:
            self.dataset_combo.addItem(name)
        self.dataset_combo.setCurrentText(name)

    def run_forecast(self) -> None:
        try:
            _, df = self.backend.ensure_dataset()
            time_col = self._guess_time_col(df)
            target_col = self._guess_target_col(df)
            params = {"window_size": 14, "p": 2, "d": 1, "q": 2, "seasonal_p": 1, "seasonal_d": 0, "seasonal_q": 1, "seasonal_period": 7, "fuzzy_sets": 30, "hidden_layer_1": 64, "hidden_layer_2": 32, "max_iter": 500, "hidden_size": 32, "num_layers": 1, "epochs": 60, "include_exogenous": True}
            result = self.backend.forecast(
                df,
                time_column=time_col,
                target_column=target_col,
                model_name=self.model_combo.currentText(),
                horizon=self.horizon_spin.value(),
                parameters=params,
            )
            history_x = safe_to_datetime(result["history_x"])
            self.forecast_chart.figure.clear()
            ax = self.forecast_chart.figure.add_subplot(111)
            ax.plot(history_x, result["history_y"], label="История", linewidth=1.8)
            ax.plot(result["future_x"], result["forecast"], label="Прогноз", linewidth=1.8, linestyle="--")
            ax.set_title(f"Прогноз: {result.get('model', self.model_combo.currentText())}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            self.forecast_chart.draw_idle()

            rows = []
            self.forecast_table.setRowCount(len(result["forecast"]))
            for r, (x, y) in enumerate(zip(result["future_x"], result["forecast"])):
                x_text = x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x)
                forecast_value = self._safe_number(y)
                self.forecast_table.setItem(r, 0, QTableWidgetItem(x_text))
                self.forecast_table.setItem(r, 1, QTableWidgetItem(str(forecast_value)))
                rows.append({"timestamp": x_text, "forecast": forecast_value})

            self.current_forecast_df = pd.DataFrame(rows, columns=["timestamp", "forecast"])
            self.next_value_label.setText(str(result["next_value"]))
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка прогноза", str(exc))

    def export_csv(self) -> None:
        self._export_forecast("csv")

    def export_excel(self) -> None:
        self._export_forecast("xlsx")

    def _export_forecast(self, file_type: str) -> None:
        if self.current_forecast_df is None or self.current_forecast_df.empty:
            QMessageBox.warning(self, "Экспорт", "Сначала постройте прогноз")
            return

        default_name = self._default_export_name(file_type)
        if file_type == "csv":
            filters = "CSV (*.csv)"
        else:
            filters = "Excel (*.xlsx)"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить прогноз",
            default_name,
            filters,
        )
        if not file_path:
            return

        path = Path(file_path)
        expected_suffix = ".csv" if file_type == "csv" else ".xlsx"
        if path.suffix.lower() != expected_suffix:
            path = path.with_suffix(expected_suffix)

        try:
            export_df = self.current_forecast_df.copy()
            if file_type == "csv":
                export_df.to_csv(path, index=False, encoding="utf-8-sig")
            else:
                export_df.to_excel(path, index=False)
            self.state.set_status(f"Прогноз экспортирован: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def _default_export_name(self, file_type: str) -> str:
        model = self.model_combo.currentText().lower().replace(" ", "_")
        dataset = (self.dataset_combo.currentText() or "dataset").lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "csv" if file_type == "csv" else "xlsx"
        return f"forecast_{dataset}_{model}_{timestamp}.{suffix}"

    @staticmethod
    def _safe_number(value):
        try:
            number = float(value)
            return round(number, 6)
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _guess_time_col(df: pd.DataFrame) -> str:
        for c in df.columns:
            if str(c).lower() in {"timestamp", "date", "datetime"}:
                return str(c)
        return str(df.columns[0])

    @staticmethod
    def _guess_target_col(df: pd.DataFrame) -> str:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return str(c)
        return str(df.columns[1] if len(df.columns) > 1 else df.columns[0])
