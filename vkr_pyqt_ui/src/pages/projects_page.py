from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QFormLayout, QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem

from src.services.app_state import AppState
from src.services.backend_service import BackendService
from src.widgets.common import BasePage, Card


class ProjectsPage(BasePage):
    def __init__(self, state: AppState, backend: BackendService) -> None:
        super().__init__("Проекты")
        self.state = state
        self.backend = backend

        mode_card = Card("Подключение")
        top_row = QHBoxLayout()
        self.mode_label = QLabel()
        self.endpoint_label = QLabel()
        top_row.addWidget(self.mode_label)
        top_row.addStretch()
        top_row.addWidget(self.endpoint_label)
        mode_card.layout.addLayout(top_row)
        self.add_card(mode_card)

        top_splitter = QSplitter()

        list_card = Card("Список проектов")
        buttons_row = QHBoxLayout()
        self.create_btn = QPushButton("Создать проект")
        self.open_btn = QPushButton("Сделать активным")
        self.delete_btn = QPushButton("Удалить")
        self.refresh_btn = QPushButton("Обновить")
        buttons_row.addWidget(self.create_btn)
        buttons_row.addWidget(self.open_btn)
        buttons_row.addWidget(self.delete_btn)
        buttons_row.addWidget(self.refresh_btn)
        buttons_row.addStretch()
        list_card.layout.addLayout(buttons_row)

        self.project_list = QListWidget()
        self.project_list.currentTextChanged.connect(self.on_project_selected)
        list_card.layout.addWidget(self.project_list)

        details_card = Card("Карточка проекта")
        form = QFormLayout()
        self.name_value = QLabel("—")
        self.description_value = QLabel("—"); self.description_value.setWordWrap(True)
        self.created_value = QLabel("—")
        self.id_value = QLabel("—")
        form.addRow("ID:", self.id_value)
        form.addRow("Название:", self.name_value)
        form.addRow("Описание:", self.description_value)
        form.addRow("Дата создания:", self.created_value)
        details_card.layout.addLayout(form)

        top_splitter.addWidget(list_card)
        top_splitter.addWidget(details_card)
        top_splitter.setSizes([420, 620])
        self.add_card(top_splitter)

        experiments_card = Card("Последние эксперименты")
        self.experiments_table = QTableWidget(0, 3)
        self.experiments_table.setHorizontalHeaderLabels(["Модель", "Датасет", "Дата"])
        self.experiments_table.verticalHeader().setVisible(False)
        self.experiments_table.horizontalHeader().setStretchLastSection(True)
        experiments_card.layout.addWidget(self.experiments_table)
        self.add_card(experiments_card)

        self.create_btn.clicked.connect(self.create_project)
        self.open_btn.clicked.connect(self.activate_project)
        self.delete_btn.clicked.connect(self.delete_project)
        self.refresh_btn.clicked.connect(self.refresh_projects)
        self.state.projects_changed.connect(self.render_projects)
        self.state.experiments_changed.connect(self.refresh_experiments)
        self.state.config_changed.connect(self.refresh_mode_banner)

        self.refresh_mode_banner()
        self.render_projects()
        self.refresh_experiments()
        self.root_layout.addStretch()

    def refresh_mode_banner(self) -> None:
        mode = self.state.config.backend_mode
        if mode == 'rest':
            self.mode_label.setText('Режим: REST backend')
            self.endpoint_label.setText(f"Base URL: {self.state.config.base_url}")
        else:
            self.mode_label.setText('Режим: MOCK — проекты создаются только локально в интерфейсе')
            self.endpoint_label.setText('Чтобы сохранять проекты в backend, переключи режим в Настройки -> rest')

    def refresh_projects(self) -> None:
        try:
            self.refresh_mode_banner()
            if self.state.config.backend_mode == 'rest':
                self.backend.load_projects()
            else:
                self.render_projects()
        except Exception as exc:
            QMessageBox.critical(self, 'Ошибка проектов', str(exc))

    def render_projects(self) -> None:
        projects = self.state.projects
        selected_name = self.project_list.currentItem().text() if self.project_list.currentItem() else self.state.current_project

        with QSignalBlocker(self.project_list):
            self.project_list.clear()
            for project in projects:
                QListWidgetItem(str(project.get('name', 'Без названия')), self.project_list)

            if projects:
                current_row = next(
                    (i for i, project in enumerate(projects) if str(project.get('name')) == selected_name),
                    next((i for i, project in enumerate(projects) if str(project.get('name')) == self.state.current_project), 0),
                )
                self.project_list.setCurrentRow(current_row)

        item = self.project_list.currentItem()
        if item is not None:
            self.on_project_selected(item.text())
        else:
            self.id_value.setText('—')
            self.name_value.setText('—')
            self.description_value.setText('—')
            self.created_value.setText('—')

    def create_project(self) -> None:
        if self.state.config.backend_mode != 'rest':
            QMessageBox.warning(
                self,
                'Сейчас включён mock',
                'Сейчас включён режим mock, поэтому проект создастся только в интерфейсе. '                'Открой Настройки, выбери backend_mode = rest, укажи Base URL backend и сохрани.',
            )
        name, ok = QInputDialog.getText(self, 'Новый проект', 'Название проекта:')
        if not ok or not name.strip():
            return
        description, _ = QInputDialog.getText(self, 'Описание', 'Описание проекта:')
        try:
            project = self.backend.create_project(name.strip(), description.strip())
            self.state.set_status(f"Проект '{project['name']}' создан")
        except Exception as exc:
            QMessageBox.critical(self, 'Ошибка создания проекта', str(exc))

    def activate_project(self) -> None:
        item = self.project_list.currentItem()
        if not item:
            return
        self.state.set_project_by_name(item.text())
        self.state.set_status(f'Активный проект: {item.text()}')


    def delete_project(self) -> None:
        item = self.project_list.currentItem()
        if not item:
            QMessageBox.warning(self, 'Проект не выбран', 'Сначала выберите проект в списке')
            return
        project_name = item.text()
        answer = QMessageBox.question(
            self,
            'Удаление проекта',
            f"Удалить проект '{project_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.backend.delete_project_by_name(project_name)
            QMessageBox.information(self, 'Проект удалён', f"Проект '{project_name}' удалён")
        except Exception as exc:
            QMessageBox.critical(self, 'Ошибка удаления проекта', str(exc))

    def on_project_selected(self, project_name: str) -> None:
        project = next((p for p in self.state.projects if str(p.get('name')) == project_name), None)
        if not project:
            return
        self.id_value.setText(str(project.get('id', '—')))
        self.name_value.setText(str(project.get('name', '—')))
        self.description_value.setText(str(project.get('description') or '—'))
        self.created_value.setText(str(project.get('created_at', '—')))

    def refresh_experiments(self) -> None:


        rows = list(self.state.experiments)
        self.experiments_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                row.get('model', '—'),
                row.get('dataset_name') or self.state.current_dataset_name,
                row.get('date') or row.get('created_at', '—'),
            ]
            for c, value in enumerate(values):
                self.experiments_table.setItem(r, c, QTableWidgetItem(str(value)))
