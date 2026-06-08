from __future__ import annotations

import sys
import faulthandler
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from src.main_window import APP_TITLE, MainWindow


def main() -> None:
    log_path = Path(__file__).resolve().parents[1] / "ui_crash.log"
    try:
        faulthandler.enable(log_path.open("w", encoding="utf-8"), all_threads=True)
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
