from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QGroupBox, QLabel, QScrollArea, QVBoxLayout, QWidget


class BasePage(QScrollArea):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self.setWidget(container)

        self.root_layout = QVBoxLayout(container)
        self.root_layout.setContentsMargins(24, 24, 24, 24)
        self.root_layout.setSpacing(18)

        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        self.root_layout.addWidget(title_label)

    def add_card(self, widget: QWidget) -> None:
        self.root_layout.addWidget(widget)


class Card(QGroupBox):
    def __init__(self, title: str = "") -> None:
        super().__init__(title)
        self.setObjectName("card")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(18, 18, 18, 18)
        self.layout.setSpacing(14)
