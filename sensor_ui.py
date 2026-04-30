import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from environment_monitor import SensorPanel, build_dashboard_stylesheet, start_ble_thread


class SensorDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能植物管护系统 - 环境传感器")
        self.resize(1420, 980)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("智能植物管护系统")
        title.setFont(QFont("Microsoft YaHei", 26, QFont.Weight.Bold))
        title.setStyleSheet("color: #1f2a24;")
        main_layout.addWidget(title)

        subtitle = QLabel("环境传感器监控与风扇控制")
        subtitle.setStyleSheet("color: #47554d; font-size: 15px;")
        main_layout.addWidget(subtitle)

        main_layout.addWidget(SensorPanel(), 1)
        self.setStyleSheet(build_dashboard_stylesheet(include_tabs=False))


def main():
    start_ble_thread()

    app = QApplication(sys.argv)
    app.setApplicationName("智能植物管护系统")
    app.setFont(QFont("Microsoft YaHei", 12))

    window = SensorDashboard()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
