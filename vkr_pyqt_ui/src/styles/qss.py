APP_STYLES = """
QMainWindow {
    background: #F2F0F0;
}
QToolBar {
    background: #FAF9F9;
    border-bottom: 1px solid #D8D2D2;
    spacing: 12px;
    padding: 10px 16px;
}
#backendBadge {
    background: #F4ECEE;
    color: #5A1423;
    border: 1px solid #D7B7BE;
    border-radius: 10px;
    padding: 6px 10px;
    margin-left: 12px;
    font-weight: 700;
}
#appHeaderTitle {
    font-size: 17px;
    font-weight: 800;
    color: #2A2527;
    margin-right: 16px;
}
#sidebar {
    background: #2D2A2B;
    color: #F8F5F5;
    border: none;
    padding: 14px 0;
    font-size: 14px;
    outline: 0;
}
#sidebar::item {
    padding: 14px 18px;
    margin: 4px 12px;
    border-radius: 12px;
}
#sidebar::item:selected {
    background: #7A1F34;
    color: #FFFFFF;
}
#sidebar::item:hover {
    background: #454041;
}
#pageTitle {
    font-size: 30px;
    font-weight: 800;
    color: #2A2527;
    margin-bottom: 2px;
}
QGroupBox#card {
    background: #FFFFFF;
    border: 1px solid #D8D2D2;
    border-radius: 18px;
    margin-top: 12px;
    font-weight: 700;
    color: #2A2527;
}
QGroupBox#card::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: #4A3F42;
}
QLabel {
    color: #302B2D;
    font-size: 14px;
}
#mutedText {
    color: #7B7375;
}
#bigMetric {
    font-size: 24px;
    font-weight: 800;
    padding: 12px 0;
    color: #7A1F34;
}
QLineEdit, QComboBox, QSpinBox, QTextEdit, QPlainTextEdit, QTableWidget, QListWidget, QTabWidget::pane {
    border: 1px solid #D0C7C9;
    border-radius: 12px;
    background: #FFFFFF;
    color: #2A2527;
}
QLineEdit, QComboBox, QSpinBox {
    padding: 0 12px;
    min-height: 38px;
}
QTextEdit, QPlainTextEdit {
    padding: 10px 12px;
}
QComboBox {
    padding-right: 30px;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 26px;
    border-left: 1px solid #D8D2D2;
}
QPushButton {
    background: #7A1F34;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0 16px;
    min-height: 40px;
    font-weight: 700;
}
QPushButton:hover {
    background: #8D2A40;
}
QPushButton:pressed {
    background: #5F1728;
}
QHeaderView::section {
    background: #EEE9EA;
    color: #2A2527;
    padding: 10px 8px;
    border: none;
    border-bottom: 1px solid #D8D2D2;
    font-weight: 700;
}
QTableWidget {
    gridline-color: #ECE7E8;
    alternate-background-color: #FAF8F8;
    selection-background-color: #E8CDD3;
    selection-color: #2A2527;
}
QTableWidget::item {
    padding: 6px;
}
QProgressBar {
    border: 1px solid #D0C7C9;
    border-radius: 10px;
    text-align: center;
    background: #FFFFFF;
    min-height: 26px;
    color: #2A2527;
}
QProgressBar::chunk {
    background: #7A1F34;
    border-radius: 9px;
}
QTabBar::tab {
    background: #EEE9EA;
    color: #302B2D;
    padding: 10px 14px;
    margin-right: 6px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
}
QTabBar::tab:selected {
    background: #FFFFFF;
    color: #7A1F34;
    font-weight: 800;
}
QStatusBar {
    background: #FAF9F9;
    border-top: 1px solid #D8D2D2;
}
QScrollArea {
    border: none;
}
"""
