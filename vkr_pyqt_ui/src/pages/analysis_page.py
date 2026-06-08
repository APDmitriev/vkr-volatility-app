from __future__ import annotations

import pandas as pd


def safe_to_datetime(values):
    try:
        return pd.to_datetime(values)
    except (ValueError, TypeError, OverflowError):
        return values

from PyQt6.QtWidgets import (
    QComboBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.services.app_state import AppState
from src.services.backend_service import BackendService
from src.widgets.charts import MplCanvas
from src.widgets.common import BasePage, Card


class AnalysisPage(BasePage):
    def __init__(self, state: AppState, backend: BackendService) -> None:
        super().__init__("Анализ")
        self.state = state
        self.backend = backend
        self._charts_ready = False

        top_card = Card("Параметры анализа")
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Датасет:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([self._current_analysis_dataset_name()])
        self.run_analysis_btn = QPushButton("Запустить анализ")
        top_row.addWidget(self.dataset_combo)
        top_row.addWidget(self.run_analysis_btn)
        top_row.addStretch()
        top_card.layout.addLayout(top_row)
        self.add_card(top_card)

        stats_card = Card("Сводная статистика")
        self.stats_table = QTableWidget(1, 7)
        self.stats_table.setHorizontalHeaderLabels(["count", "mean", "std", "min", "max", "median", "var_coef"])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.horizontalHeader().setStretchLastSection(False)
        self.stats_table.verticalHeader().setDefaultSectionSize(30)
        self.stats_table.setFixedHeight(92)
        stats_card.layout.addWidget(self.stats_table)
        self.add_card(stats_card)

        profile_card = Card("Профиль волатильности и рекомендация модели")
        self.profile_summary = QLabel("")
        self.profile_summary.setWordWrap(True)
        self.profile_summary.setObjectName("mutedText")
        self.profile_summary.setVisible(False)
        profile_card.layout.addWidget(self.profile_summary)
        self.profile_table = QTableWidget(0, 3)
        self.profile_table.setHorizontalHeaderLabels(["Модель", "Оценка", "Причина"])
        self.profile_table.verticalHeader().setVisible(False)
        self.profile_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.profile_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.profile_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        profile_card.layout.addWidget(self.profile_table)
        self.add_card(profile_card)

        tabs_card = Card("Результаты анализа")
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.setMinimumHeight(360)
        self.analysis_tabs.setUsesScrollButtons(False)

        self.series_container = self._make_placeholder_tab("")
        self.trend_container = self._make_placeholder_tab("")
        self.acf_container = self._make_placeholder_tab("")
        self.volatility_container = self._make_placeholder_tab("")
        self.hist_container = self._make_placeholder_tab("")

        self.analysis_tabs.addTab(self.series_container, "Общая статистика")
        self.analysis_tabs.addTab(self.trend_container, "Тренд")
        self.analysis_tabs.addTab(self.acf_container, "Автокорреляция")
        self.analysis_tabs.addTab(self.volatility_container, "Волатильность")
        self.analysis_tabs.addTab(self.hist_container, "Распределение")
        tabs_card.layout.addWidget(self.analysis_tabs)
        self.add_card(tabs_card)
        self.root_layout.addStretch()

        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.state.dataset_changed.connect(self.on_dataset_changed)

    def _current_analysis_dataset_name(self) -> str:
        if self.state.processed_df is not None:
            return f"{self.state.current_dataset_name} — предобработан"
        return "Нет предобработанного датасета"

    def _make_placeholder_tab(self, text: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setObjectName("mutedText")
        layout.addWidget(label)
        layout.addStretch()
        return widget

    def _ensure_charts(self) -> None:
        if self._charts_ready:
            return
        self.series_chart = MplCanvas(280)
        self.hist_chart = MplCanvas(280)
        self.acf_chart = MplCanvas(280)
        self.volatility_chart = MplCanvas(280)
        self.trend_chart = MplCanvas(280)

        for index, chart in [
            (0, self.series_chart),
            (1, self.trend_chart),
            (2, self.acf_chart),
            (3, self.volatility_chart),
            (4, self.hist_chart),
        ]:
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(chart)
            self.analysis_tabs.removeTab(index)
            self.analysis_tabs.insertTab(index, container, ["Общая статистика", "Тренд", "Автокорреляция", "Волатильность", "Распределение"][index])
        self._charts_ready = True

    def on_dataset_changed(self, name: str) -> None:
        display_name = self._current_analysis_dataset_name()
        if self.dataset_combo.findText(display_name) == -1:
            self.dataset_combo.addItem(display_name)
        self.dataset_combo.setCurrentText(display_name)

    def run_analysis(self) -> None:
        try:
            if self.state.processed_df is None:
                raise ValueError("Для анализа нужен предобработанный датасет. Откройте вкладку «Данные», выберите или загрузите датасет и нажмите «Применить предобработку».")
            df = self.state.processed_df.copy()
            time_col = self._guess_time_col(df)
            target_col = self._guess_target_col(df)
            payload = self.backend.analyze_dataset(df, time_column=time_col, target_column=target_col)
            self.fill_stats(payload["stats"])
            self._ensure_charts()
            x = safe_to_datetime(payload["series_x"])
            self.series_chart.plot_series(x, payload["series_y"], title="Исходный ряд", label=target_col)
            self.trend_chart.plot_two_series(x, payload["series_y"], payload["trend"], title="Ряд и тренд", label1="Ряд", label2="Тренд")
            self.acf_chart.plot_bar(payload["acf_lags"], payload["acf_values"], title="ACF", xlabel="Lag", ylabel="Correlation")
            self.volatility_chart.plot_series(x, payload["volatility"], title="Оценка волатильности", label="Rolling std")
            self.hist_chart.plot_hist(payload["histogram_values"], title="Распределение значений")
            self.fill_volatility_profile(payload.get("volatility_profile", {}))
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка анализа", str(exc))


    def fill_volatility_profile(self, profile: dict) -> None:
        if not profile:
            self.profile_summary.setVisible(False)
            self.profile_summary.setText("")
            self.profile_table.setRowCount(0)
            return

        def fmt(value, digits: int = 4) -> str:
            if value is None:
                return "—"
            try:
                return str(round(float(value), digits))
            except (TypeError, ValueError):
                return str(value)

        signs = []
        signs.append(f"уровень волатильности — {profile.get('volatility_level_ru', profile.get('volatility_level', '—'))}")
        signs.append(f"std доходностей — {fmt(profile.get('std_returns'))}")
        signs.append(f"доля скачков — {fmt((profile.get('spike_ratio') or 0) * 100, 2)}%")
        if profile.get("seasonality_detected"):
            signs.append(f"сезонность — обнаружена, период {profile.get('seasonal_period', '—')}")
        else:
            signs.append("сезонность — не обнаружена")
        signs.append("кластеризация волатильности — " + ("обнаружена" if profile.get("volatility_clustering_detected") else "не обнаружена"))
        signs.append("автокорреляция — " + ("обнаружена" if profile.get("autocorrelation_detected") else "не обнаружена"))

        recommended = profile.get("recommended_model", "—")
        alternatives = ", ".join(profile.get("alternative_models", [])) or "—"
        self.profile_summary.setVisible(True)
        self.profile_summary.setText(
            f"Рекомендуемая модель: {recommended}. Альтернативы: {alternatives}.\n"
            f"Ключевые признаки: {'; '.join(signs)}."
        )

        ranking = profile.get("model_ranking", [])
        self.profile_table.setRowCount(len(ranking))
        for r, row in enumerate(ranking):
            values = [
                row.get("model", "—"),
                fmt(row.get("score"), 3),
                row.get("reason", "—"),
            ]
            for c, value in enumerate(values):
                self.profile_table.setItem(r, c, QTableWidgetItem(str(value)))

    def fill_stats(self, stats: dict[str, float]) -> None:
        values = [
            str(stats.get("count", "—")),
            f"{stats.get('mean', 0):.4f}",
            f"{stats.get('std', 0):.4f}",
            f"{stats.get('min', 0):.4f}",
            f"{stats.get('max', 0):.4f}",
            f"{stats.get('median', 0):.4f}",
            f"{stats.get('var_coef', 0):.4f}",
        ]
        for c, value in enumerate(values):
            self.stats_table.setItem(0, c, QTableWidgetItem(value))

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
