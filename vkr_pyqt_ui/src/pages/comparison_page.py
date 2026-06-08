from __future__ import annotations

import math
import textwrap
from typing import Any

from PyQt6.QtWidgets import (
    QComboBox,
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
from src.widgets.charts import MplCanvas
from src.widgets.common import BasePage, Card


class ComparisonPage(BasePage):
    """Страница сравнения сохранённых экспериментов.

    История загружается из backend через endpoint /api/v1/experiments/.
    Таблица отображает все эксперименты текущего проекта, отсортированные
    от лучшего к худшему. Графики строятся только по 8 лучшим запускам,
    чтобы визуализация оставалась читаемой.
    """

    MAX_CHART_BARS = 8

    def __init__(self, state: AppState, backend: BackendService) -> None:
        super().__init__("Сравнение результатов")
        self.state = state
        self.backend = backend
        self.current_rows: list[dict[str, Any]] = []

        filters_card = Card("Фильтры")
        filters_row = QHBoxLayout()
        filters_row.addWidget(QLabel("Датасет:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["Все датасеты", state.current_dataset_name])
        filters_row.addWidget(self.dataset_combo)

        filters_row.addWidget(QLabel("Период:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["Все эксперименты проекта"])
        filters_row.addWidget(self.period_combo)

        self.refresh_btn = QPushButton("Обновить")
        filters_row.addWidget(self.refresh_btn)
        filters_row.addStretch()
        filters_card.layout.addLayout(filters_row)
        self.add_card(filters_card)

        table_card = Card("Таблица моделей")
        self.compare_table = QTableWidget(0, 7)
        self.compare_table.setHorizontalHeaderLabels([
            "Место",
            "Модель",
            "Датасет",
            "RMSE",
            "MAE",
            "MAPE",
            "Дата",
        ])
        self.compare_table.verticalHeader().setVisible(False)
        self.compare_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.compare_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.compare_table.setMinimumHeight(330)
        header = self.compare_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setStretchLastSection(False)
        table_card.layout.addWidget(self.compare_table)
        self.add_card(table_card)

        charts_row = QHBoxLayout()
        self.rmse_chart = MplCanvas(540)
        self.mae_chart = MplCanvas(540)
        self.rmse_chart.setMinimumHeight(540)
        self.mae_chart.setMinimumHeight(540)
        rmse_card = Card("RMSE: 8 лучших экспериментов")
        rmse_card.layout.addWidget(self.rmse_chart)
        mae_card = Card("MAE: 8 лучших экспериментов")
        mae_card.layout.addWidget(self.mae_chart)
        charts_row.addWidget(rmse_card)
        charts_row.addWidget(mae_card)
        self.root_layout.addLayout(charts_row)
        self.root_layout.addStretch()

        self.refresh_btn.clicked.connect(self.refresh)
        self.dataset_combo.currentTextChanged.connect(lambda _: self.render())
        self.state.experiments_changed.connect(self.render)
        self.state.dataset_changed.connect(self.on_dataset_changed)
        self.state.project_changed.connect(lambda _: self.refresh())
        self.refresh()

    def on_dataset_changed(self, name: str) -> None:
        if name and self.dataset_combo.findText(name) == -1:
            self.dataset_combo.addItem(name)

    @staticmethod
    def _to_float(value: object) -> float | None:
        if value in (None, "", "—"):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    @staticmethod
    def _fmt_metric(value: object) -> str:
        number = ComparisonPage._to_float(value)
        if number is None:
            return "—"
        if abs(number) >= 1000:
            return f"{number:.2f}"
        if abs(number) >= 1:
            return f"{number:.4f}"
        return f"{number:.6f}"

    def _quality_key(self, row: dict[str, Any]) -> tuple[float, float, float, float]:
        """Чем меньше ключ, тем лучше эксперимент.

        Основной критерий — RMSE. Если RMSE отсутствует, используется MAE,
        затем MAPE и MSE. Эксперименты без метрик попадают в конец списка.
        """
        rmse = self._to_float(row.get("rmse"))
        mae = self._to_float(row.get("mae"))
        mape = self._to_float(row.get("mape"))
        mse = self._to_float(row.get("mse"))

        inf = float("inf")
        primary = rmse if rmse is not None else (mae if mae is not None else (mape if mape is not None else inf))
        return (
            primary,
            mae if mae is not None else inf,
            mape if mape is not None else inf,
            mse if mse is not None else inf,
        )

    def _filtered_and_sorted_rows(self) -> list[dict[str, Any]]:
        dataset_filter = self.dataset_combo.currentText()
        rows = list(self.current_rows)
        if dataset_filter and dataset_filter != "Все датасеты":
            rows = [
                row for row in rows
                if str(row.get("dataset_name", "")) == dataset_filter
            ]
        return sorted(rows, key=self._quality_key)

    def refresh(self) -> None:
        try:
            # В REST-режиме этот вызов обращается к backend endpoint /api/v1/experiments/.
            self.current_rows = self.backend.reload_experiments()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сравнения", str(exc))
            self.current_rows = []
        self.render()

    def render(self) -> None:
        rows = self._filtered_and_sorted_rows()
        self.compare_table.setRowCount(len(rows))

        for r, row in enumerate(rows):
            mape = self._fmt_metric(row.get("mape"))
            if mape != "—":
                mape = f"{mape}%"
            values = [
                str(r + 1),
                str(row.get("model", "—")),
                str(row.get("dataset_name", "—")),
                self._fmt_metric(row.get("rmse")),
                self._fmt_metric(row.get("mae")),
                mape,
                str(row.get("date", row.get("created_at", "—"))),
            ]
            for c, value in enumerate(values):
                self.compare_table.setItem(r, c, QTableWidgetItem(value))

        if not rows:
            self.rmse_chart.clear("RMSE", "Нет сохранённых экспериментов")
            self.mae_chart.clear("MAE", "Нет сохранённых экспериментов")
            return

        chart_rows = [
            row for row in rows
            if self._to_float(row.get("rmse")) is not None or self._to_float(row.get("mae")) is not None
        ][: self.MAX_CHART_BARS]

        if not chart_rows:
            self.rmse_chart.clear("RMSE", "Нет метрик для построения графика")
            self.mae_chart.clear("MAE", "Нет метрик для построения графика")
            return

        self._plot_metric_chart(
            self.rmse_chart,
            chart_rows,
            metric="rmse",
            title="RMSE по лучшим экспериментам",
            ylabel="RMSE",
        )
        self._plot_metric_chart(
            self.mae_chart,
            chart_rows,
            metric="mae",
            title="MAE по лучшим экспериментам",
            ylabel="MAE",
        )

    @staticmethod
    def _chart_label(model_name: str) -> str:
        """Возвращает компактную подпись графика без номера эксперимента."""
        clean_name = model_name.strip() or "Модель"
        parts = textwrap.wrap(clean_name, width=14, break_long_words=False)
        if len(parts) <= 3:
            return "\n".join(parts)
        return "\n".join(parts[:2] + [" ".join(parts[2:])])

    def _plot_metric_chart(
        self,
        canvas: MplCanvas,
        rows: list[dict[str, Any]],
        *,
        metric: str,
        title: str,
        ylabel: str,
    ) -> None:
        labels = []
        values = []
        fallback_metric = "mae" if metric == "rmse" else "rmse"

        for index, row in enumerate(rows, start=1):
            model_name = str(row.get("model", f"Модель {index}"))
            value = self._to_float(row.get(metric))
            if value is None:
                value = self._to_float(row.get(fallback_metric))
            if value is None:
                continue
            labels.append(self._chart_label(model_name))
            values.append(value)

        if not values:
            canvas.clear(title, "Нет данных для графика")
            return

        try:
            canvas.figure.set_layout_engine(None)
        except Exception:
            pass

        canvas.figure.clear()
        canvas.figure.set_constrained_layout(False)
        canvas.figure.set_size_inches(7.2, 5.2, forward=False)
        ax = canvas.figure.add_subplot(111)
        x = list(range(len(values)))
        ax.bar(x, values)
        ax.set_title(title, pad=8, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.margins(x=0.03)
        canvas.figure.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.25)
        canvas.axes = ax
        canvas.draw_idle()
