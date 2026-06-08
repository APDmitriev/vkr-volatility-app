from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QSpinBox,
)

from src.services.app_state import AppState
from src.services.backend_service import BackendService
from src.widgets.common import BasePage, Card


SUPPORTED_MODELS = [
    "Linear Regression",
    "ARIMA",
    "SARIMA",
    "GARCH",
    "Fuzzy First Order",
    "MLP Neural Network",
    "LSTM Neural Network",
]


MODEL_PARAMETER_GROUPS = {
    "Linear Regression": {"window", "exogenous"},
    "ARIMA": {"arima"},
    "SARIMA": {"arima", "sarima"},
    "GARCH": {"garch"},
    "Fuzzy First Order": {"fuzzy"},
    "MLP Neural Network": {"window", "mlp", "seed", "exogenous"},
    "LSTM Neural Network": {"window", "lstm", "seed", "exogenous"},
}


class TrainingPage(BasePage):
    def __init__(self, state: AppState, backend: BackendService) -> None:
        super().__init__("Обучение моделей")
        self.state = state
        self.backend = backend
        self.parameter_rows: dict[str, list[tuple[QLabel, object]]] = {}

        splitter = QSplitter()

        config_card = Card("Параметры модели")
        form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(SUPPORTED_MODELS)
        self.horizon_spin = QSpinBox(); self.horizon_spin.setRange(1, 365); self.horizon_spin.setValue(10)
        self.train_ratio_spin = QSpinBox(); self.train_ratio_spin.setRange(50, 95); self.train_ratio_spin.setValue(80)
        self.window_size_spin = QSpinBox(); self.window_size_spin.setRange(1, 365); self.window_size_spin.setValue(14)
        self.random_seed_spin = QSpinBox(); self.random_seed_spin.setRange(0, 99999); self.random_seed_spin.setValue(42)

        self.p_spin = QSpinBox(); self.p_spin.setRange(0, 10); self.p_spin.setValue(2)
        self.d_spin = QSpinBox(); self.d_spin.setRange(0, 5); self.d_spin.setValue(1)
        self.q_spin = QSpinBox(); self.q_spin.setRange(0, 10); self.q_spin.setValue(2)
        self.seasonal_p_spin = QSpinBox(); self.seasonal_p_spin.setRange(0, 5); self.seasonal_p_spin.setValue(1)
        self.seasonal_d_spin = QSpinBox(); self.seasonal_d_spin.setRange(0, 2); self.seasonal_d_spin.setValue(0)
        self.seasonal_q_spin = QSpinBox(); self.seasonal_q_spin.setRange(0, 5); self.seasonal_q_spin.setValue(1)
        self.seasonal_period_spin = QSpinBox(); self.seasonal_period_spin.setRange(2, 365); self.seasonal_period_spin.setValue(7)
        self.fuzzy_sets_spin = QSpinBox(); self.fuzzy_sets_spin.setRange(7, 100); self.fuzzy_sets_spin.setValue(25)

        self.hidden_layer_1_spin = QSpinBox(); self.hidden_layer_1_spin.setRange(1, 512); self.hidden_layer_1_spin.setValue(64)
        self.hidden_layer_2_spin = QSpinBox(); self.hidden_layer_2_spin.setRange(0, 512); self.hidden_layer_2_spin.setValue(32)
        self.max_iter_spin = QSpinBox(); self.max_iter_spin.setRange(100, 5000); self.max_iter_spin.setValue(500)
        self.lstm_hidden_size_spin = QSpinBox(); self.lstm_hidden_size_spin.setRange(4, 512); self.lstm_hidden_size_spin.setValue(32)
        self.lstm_layers_spin = QSpinBox(); self.lstm_layers_spin.setRange(1, 4); self.lstm_layers_spin.setValue(1)
        self.lstm_epochs_spin = QSpinBox(); self.lstm_epochs_spin.setRange(5, 1000); self.lstm_epochs_spin.setValue(60)
        self.use_exogenous_check = QCheckBox("Использовать дополнительные числовые признаки")
        self.use_exogenous_check.setChecked(True)

        self._add_row(form, "Модель:", self.model_combo, "common")
        self._add_row(form, "Горизонт прогноза:", self.horizon_spin, "common")
        self._add_row(form, "Окно признаков:", self.window_size_spin, "window")
        self._add_row(form, "Использование признаков:", self.use_exogenous_check, "exogenous")
        self._add_row(form, "Random seed:", self.random_seed_spin, "seed")
        self._add_row(form, "p:", self.p_spin, "arima")
        self._add_row(form, "d:", self.d_spin, "arima")
        self._add_row(form, "q:", self.q_spin, "arima")
        self._add_row(form, "P сезонное:", self.seasonal_p_spin, "sarima")
        self._add_row(form, "D сезонное:", self.seasonal_d_spin, "sarima")
        self._add_row(form, "Q сезонное:", self.seasonal_q_spin, "sarima")
        self._add_row(form, "s длина сезона:", self.seasonal_period_spin, "sarima")
        self._add_row(form, "Число нечётких множеств:", self.fuzzy_sets_spin, "fuzzy")
        self._add_row(form, "MLP: скрытый слой 1:", self.hidden_layer_1_spin, "mlp")
        self._add_row(form, "MLP: скрытый слой 2:", self.hidden_layer_2_spin, "mlp")
        self._add_row(form, "MLP: максимум итераций:", self.max_iter_spin, "mlp")
        self._add_row(form, "LSTM: hidden size:", self.lstm_hidden_size_spin, "lstm")
        self._add_row(form, "LSTM: число слоёв:", self.lstm_layers_spin, "lstm")
        self._add_row(form, "LSTM: эпохи:", self.lstm_epochs_spin, "lstm")
        config_card.layout.addLayout(form)

        config_card.layout.addStretch()

        execution_card = Card("Выполнение")
        recommendation_row = QHBoxLayout()
        self.apply_recommended_model_btn = QPushButton("Выбрать рекомендованную модель")
        recommendation_row.addWidget(self.apply_recommended_model_btn)
        recommendation_row.addStretch()
        execution_card.layout.addLayout(recommendation_row)

        actions_row = QHBoxLayout()
        self.tune_params_btn = QPushButton("Подобрать параметры обучения")
        self.run_training_btn = QPushButton("Запустить")
        self.save_model_btn = QPushButton("Сохранить модель")
        self.save_model_btn.setEnabled(False)
        actions_row.addWidget(self.tune_params_btn)
        actions_row.addWidget(self.run_training_btn)
        actions_row.addWidget(self.save_model_btn)
        actions_row.addStretch()
        execution_card.layout.addLayout(actions_row)

        self.training_status = QLabel("Статус: ожидание")
        self.training_progress = QProgressBar(); self.training_progress.setRange(0, 100); self.training_progress.setValue(0)
        execution_card.layout.addWidget(self.training_status)
        execution_card.layout.addWidget(self.training_progress)

        self.training_log = QPlainTextEdit(); self.training_log.setReadOnly(True); self.training_log.setMinimumHeight(140)
        self.training_log.setPlainText("")
        execution_card.layout.addWidget(self.training_log)

        metrics_grid = QGridLayout()
        self.mae_value = QLabel("—"); self.mse_value = QLabel("—"); self.rmse_value = QLabel("—"); self.mape_value = QLabel("—")
        metrics_grid.addWidget(QLabel("MAE / avg vol:"), 0, 0); metrics_grid.addWidget(self.mae_value, 0, 1)
        metrics_grid.addWidget(QLabel("MSE:"), 1, 0); metrics_grid.addWidget(self.mse_value, 1, 1)
        metrics_grid.addWidget(QLabel("RMSE:"), 0, 2); metrics_grid.addWidget(self.rmse_value, 0, 3)
        metrics_grid.addWidget(QLabel("MAPE:"), 1, 2); metrics_grid.addWidget(self.mape_value, 1, 3)
        execution_card.layout.addLayout(metrics_grid)

        splitter.addWidget(config_card)
        splitter.addWidget(execution_card)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([430, 760])
        self.add_card(splitter)

        experiments_card = Card("Таблица экспериментов")
        self.training_table = QTableWidget(0, 7)
        self.training_table.setHorizontalHeaderLabels(["ID", "Дата", "Датасет", "Модель", "MAE", "RMSE", "MAPE"])
        self.training_table.verticalHeader().setVisible(False)
        self.training_table.setMinimumHeight(320)
        self.training_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.training_table.horizontalHeader().setStretchLastSection(False)
        experiments_card.layout.addWidget(self.training_table)
        self.add_card(experiments_card)

        self.model_combo.currentTextChanged.connect(self.update_parameter_visibility)
        self.apply_recommended_model_btn.clicked.connect(self.apply_recommended_model)
        self.tune_params_btn.clicked.connect(self.tune_parameters)
        self.run_training_btn.clicked.connect(self.run_training)
        self.state.experiments_changed.connect(self.refresh_experiments)
        self.update_parameter_visibility()
        self.refresh_experiments()
        self.root_layout.addStretch()

    def _add_row(self, form: QFormLayout, label_text: str, widget: object, group: str) -> None:
        label = QLabel(label_text)
        form.addRow(label, widget)  # type: ignore[arg-type]
        self.parameter_rows.setdefault(group, []).append((label, widget))

    def update_parameter_visibility(self) -> None:
        model = self.model_combo.currentText()
        visible_groups = {"common"} | MODEL_PARAMETER_GROUPS.get(model, set())
        for group, rows in self.parameter_rows.items():
            visible = group in visible_groups
            for label, widget in rows:
                label.setVisible(visible)
                widget.setVisible(visible)  # type: ignore[attr-defined]

    @staticmethod
    def _model_note(model: str) -> str:
        notes = {
            "Linear Regression": "Линейная регрессия использует лаговые значения ряда. Дополнительные числовые признаки могут использоваться как лаговые факторы.",
            "ARIMA": "ARIMA использует только целевой ряд value и параметры p, d, q.",
            "SARIMA": "SARIMA расширяет ARIMA сезонными параметрами P, D, Q и длиной сезона s.",
            "GARCH": "GARCH прогнозирует условную волатильность доходностей, а не уровень ряда напрямую.",
            "Fuzzy First Order": "Нечёткая модель использует число нечётких множеств и подходит для скачкообразных рядов.",
            "MLP Neural Network": "MLP строит лаговую таблицу признаков и может использовать дополнительные числовые колонки после предобработки.",
            "LSTM Neural Network": "LSTM обучается на последовательностях наблюдений и может использовать дополнительные числовые признаки вместе с лагами value.",
        }
        return notes.get(model, "")

    def _recommended_model_text(self) -> str:
        profile = getattr(self.state, "last_volatility_profile", {}) or {}
        recommended = getattr(self.state, "recommended_model", None) or profile.get("recommended_model")
        if not recommended:
            return "Рекомендация модели: сначала выполните анализ предобработанного датасета на странице «Анализ»."
        explanation = profile.get("explanation", "")
        if explanation:
            return f"Рекомендация модели: {recommended}. {explanation}"
        return f"Рекомендация модели: {recommended}."

    def apply_recommended_model(self) -> None:
        profile = getattr(self.state, "last_volatility_profile", {}) or {}
        recommended = getattr(self.state, "recommended_model", None) or profile.get("recommended_model")
        if not recommended:
            QMessageBox.information(
                self,
                "Рекомендация не рассчитана",
                "Сначала выполните анализ предобработанного датасета на странице «Анализ». После этого система определит рекомендуемую модель.",
            )
            return
        index = self.model_combo.findText(str(recommended))
        if index == -1:
            QMessageBox.warning(self, "Модель не найдена", f"Модель {recommended} отсутствует в списке доступных моделей.")
            return
        self.model_combo.setCurrentIndex(index)
        self.training_log.setPlainText(f"[INFO] Выбрана модель: {recommended}")
        self.training_status.setText("Статус: выбрана рекомендованная модель")

    def _collect_parameters(self) -> dict:
        return {
            "p": self.p_spin.value(),
            "d": self.d_spin.value(),
            "q": self.q_spin.value(),
            "seasonal_p": self.seasonal_p_spin.value(),
            "seasonal_d": self.seasonal_d_spin.value(),
            "seasonal_q": self.seasonal_q_spin.value(),
            "seasonal_period": self.seasonal_period_spin.value(),
            "window_size": self.window_size_spin.value(),
            "fuzzy_sets": self.fuzzy_sets_spin.value(),
            "seed": self.random_seed_spin.value(),
            "random_state": self.random_seed_spin.value(),
            "hidden_layer_1": self.hidden_layer_1_spin.value(),
            "hidden_layer_2": self.hidden_layer_2_spin.value(),
            "max_iter": self.max_iter_spin.value(),
            "lstm_hidden_size": self.lstm_hidden_size_spin.value(),
            "hidden_size": self.lstm_hidden_size_spin.value(),
            "num_layers": self.lstm_layers_spin.value(),
            "epochs": self.lstm_epochs_spin.value(),
            "include_exogenous": self.use_exogenous_check.isChecked(),
        }

    def _apply_parameters(self, params: dict) -> None:
        if "window_size" in params:
            self.window_size_spin.setValue(max(1, int(params["window_size"])))
        if "p" in params:
            self.p_spin.setValue(max(0, int(params["p"])))
        if "d" in params:
            self.d_spin.setValue(max(0, int(params["d"])))
        if "q" in params:
            self.q_spin.setValue(max(0, int(params["q"])))
        if "seasonal_p" in params:
            self.seasonal_p_spin.setValue(max(0, int(params["seasonal_p"])))
        if "seasonal_d" in params:
            self.seasonal_d_spin.setValue(max(0, int(params["seasonal_d"])))
        if "seasonal_q" in params:
            self.seasonal_q_spin.setValue(max(0, int(params["seasonal_q"])))
        if "seasonal_period" in params:
            self.seasonal_period_spin.setValue(max(2, int(params["seasonal_period"])))
        if "max_sets" in params:
            self.fuzzy_sets_spin.setValue(max(7, int(params["max_sets"])))
        elif "fuzzy_sets" in params:
            self.fuzzy_sets_spin.setValue(max(7, int(params["fuzzy_sets"])))
        if "hidden_layer_1" in params:
            self.hidden_layer_1_spin.setValue(max(1, int(params["hidden_layer_1"])))
        if "hidden_layer_2" in params:
            self.hidden_layer_2_spin.setValue(max(0, int(params["hidden_layer_2"])))
        if "max_iter" in params:
            self.max_iter_spin.setValue(max(100, int(params["max_iter"])))
        if "hidden_size" in params:
            self.lstm_hidden_size_spin.setValue(max(4, int(params["hidden_size"])))
        if "lstm_hidden_size" in params:
            self.lstm_hidden_size_spin.setValue(max(4, int(params["lstm_hidden_size"])))
        if "num_layers" in params:
            self.lstm_layers_spin.setValue(max(1, int(params["num_layers"])))
        if "epochs" in params:
            self.lstm_epochs_spin.setValue(max(5, int(params["epochs"])))
        if "random_state" in params:
            self.random_seed_spin.setValue(max(0, int(params["random_state"])))
        if "seed" in params:
            self.random_seed_spin.setValue(max(0, int(params["seed"])))

    def _set_metric_labels(self, metrics: dict) -> None:
        def fmt(value, digits: int = 4) -> str:
            if value is None or value == "—":
                return "—"
            try:
                return str(round(float(value), digits))
            except (TypeError, ValueError):
                return str(value)

        self.mae_value.setText(fmt(metrics.get("mae")))
        self.mse_value.setText(fmt(metrics.get("mse")))
        self.rmse_value.setText(fmt(metrics.get("rmse")))
        mape = metrics.get("mape")
        self.mape_value.setText(f"{fmt(mape)}%" if mape not in (None, "—") else "—")

    def tune_parameters(self) -> None:
        model_name = self.model_combo.currentText()
        try:
            _, df = self.backend.ensure_dataset()
            time_col = self._guess_time_col(df)
            target_col = self._guess_target_col(df)
            self.training_status.setText("Статус: подбор параметров выполняется")
            self.training_progress.setValue(25)
            self.tune_params_btn.setEnabled(False)
            self.run_training_btn.setEnabled(False)

            result = self.backend.tune_model_parameters(
                df,
                time_column=time_col,
                target_column=target_col,
                model_name=model_name,
                horizon=self.horizon_spin.value(),
                train_ratio=self.train_ratio_spin.value(),
                parameters=self._collect_parameters(),
            )
            best_parameters = result.get("best_parameters", {})
            self._apply_parameters(best_parameters)
            self._set_metric_labels(result.get("metrics", {}))
            self.training_status.setText("Статус: параметры подобраны")
            self.training_progress.setValue(100)
            self.training_log.setPlainText(
                f"[INFO] Модель: {model_name}\n"
                f"[INFO] Лучшие параметры: {best_parameters}\n"
                f"[INFO] Критерий: {result.get('optimize_by', 'rmse')}"
            )
        except Exception as exc:
            self.training_status.setText("Статус: ошибка подбора")
            self.training_progress.setValue(0)
            QMessageBox.critical(self, "Ошибка подбора параметров", str(exc))
        finally:
            self.tune_params_btn.setEnabled(True)
            self.run_training_btn.setEnabled(True)

    def run_training(self) -> None:
        model_name = self.model_combo.currentText()
        try:
            _, df = self.backend.ensure_dataset()
            time_col = self._guess_time_col(df)
            target_col = self._guess_target_col(df)
            self.training_status.setText("Статус: обучение выполняется")
            self.training_progress.setValue(35)
            self.tune_params_btn.setEnabled(False)
            self.run_training_btn.setEnabled(False)
            result = self.backend.train_model(
                df,
                time_column=time_col,
                target_column=target_col,
                model_name=model_name,
                horizon=self.horizon_spin.value(),
                train_ratio=self.train_ratio_spin.value(),
                parameters=self._collect_parameters(),
            )
            self.training_status.setText("Статус: обучение завершено")
            self.training_progress.setValue(100)
            self.training_log.setPlainText(
                f"[INFO] Модель: {model_name}\n"
                f"[INFO] Параметры: {result.get('parameters', {})}\n"
                "[INFO] Эксперимент сохранён"
            )
            self._set_metric_labels(result)
            self.refresh_experiments()
        except Exception as exc:
            self.training_status.setText("Статус: ошибка")
            self.training_progress.setValue(0)
            QMessageBox.critical(self, "Ошибка обучения", str(exc))
        finally:
            self.tune_params_btn.setEnabled(True)
            self.run_training_btn.setEnabled(True)

    def refresh_experiments(self) -> None:
        rows = self.backend.get_experiments()
        self.training_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                row.get("id", "—"),
                row.get("date", "—"),
                row.get("dataset_name", "—"),
                row.get("model", "—"),
                str(row.get("mae", "—")),
                str(row.get("rmse", "—")),
                str(row.get("mape", "—")),
            ]
            for c, value in enumerate(values):
                self.training_table.setItem(r, c, QTableWidgetItem(str(value)))

    @staticmethod
    def _guess_time_col(df: pd.DataFrame) -> str:
        for c in df.columns:
            if str(c).lower() in {"timestamp", "date", "datetime"}:
                return str(c)
        return str(df.columns[0])

    @staticmethod
    def _guess_target_col(df: pd.DataFrame) -> str:
        if "value" in df.columns:
            return "value"
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return str(c)
        return str(df.columns[1] if len(df.columns) > 1 else df.columns[0])
