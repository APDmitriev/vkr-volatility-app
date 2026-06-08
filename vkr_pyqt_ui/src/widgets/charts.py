from __future__ import annotations

from typing import Iterable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QSizePolicy


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, min_height: int = 240) -> None:
        self.figure = Figure(figsize=(5, 3), dpi=100, constrained_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.updateGeometry()

    def clear(self, title: str, subtitle: str | None = None) -> None:
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title(title)
        if subtitle:
            self.axes.text(0.5, 0.5, subtitle, ha="center", va="center", transform=self.axes.transAxes)
        self.draw_idle()

    def plot_series(self, x: Iterable, y: Iterable, *, title: str, label: str, ylabel: str = "Value") -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(list(x), list(y), linewidth=1.8, label=label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        self.axes = ax
        self.draw_idle()

    def plot_two_series(self, x: Iterable, y1: Iterable, y2: Iterable, *, title: str, label1: str, label2: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        lx = list(x)
        ax.plot(lx, list(y1), linewidth=1.8, label=label1)
        ax.plot(lx, list(y2), linewidth=1.8, linestyle="--", label=label2)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()
        self.axes = ax
        self.draw_idle()

    def plot_bar(self, x: Iterable, y: Iterable, *, title: str, xlabel: str = "", ylabel: str = "") -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(list(x), list(y))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        self.axes = ax
        self.draw_idle()

    def plot_hist(self, values: Iterable, *, title: str, bins: int = 20) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.hist(list(values), bins=bins)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        self.axes = ax
        self.draw_idle()
