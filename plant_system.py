import os
import sys
import time
from pathlib import Path

import cv2

# opencv-python injects its bundled Qt5 plugin path into the environment,
# which conflicts with this PyQt6 application at startup.
for qt_env_var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    qt_env_value = os.environ.get(qt_env_var)
    if qt_env_value and "cv2/qt" in qt_env_value:
        os.environ.pop(qt_env_var, None)

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


BASE_DIR = Path(__file__).resolve().parent
YOLO_DIR = BASE_DIR / "rknn3588-yolov8"
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

from func import detect_frame  # noqa: E402
from rknnpool import rknnPoolExecutor  # noqa: E402
from environment_monitor import (  # noqa: E402
    DASHBOARD_COLORS,
    SensorPanel,
    build_dashboard_stylesheet,
    start_ble_thread,
)


CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
THREAD_COUNT = 3
MODEL_PATH = YOLO_DIR / "rknnModel" / "plant_det.rknn"


def cv_to_qimage(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()


def format_top3_lines(candidates):
    if not candidates:
        return ["未获得候选类别"]
    return [f"Top {index}. {item['label']} | {item['score']:.2f}" for index, item in enumerate(candidates[:3], start=1)]


def format_detection_lines(detections):
    if not detections:
        return ["未检测到明显目标"]
    lines = []
    for index, detection in enumerate(detections[:8], start=1):
        x1, y1, x2, y2 = detection["box"]
        lines.append(
            f"{index}. {detection['label']} | {detection['score']:.2f} | ({x1}, {y1})-({x2}, {y2})"
        )
    if len(detections) > 8:
        lines.append(f"... 其余 {len(detections) - 8} 个结果已省略")
    return lines


class CameraPreviewDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview_frame)
        self.preview_label = None
        self.status_label = None
        self.capture_button = None
        self.latest_frame = None
        self.captured_frame = None
        self.init_ui()
        self.start_preview()

    def init_ui(self):
        self.setWindowTitle("摄像头预览")
        self.resize(980, 720)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        self.status_label = QLabel("调整画面后点击拍照")
        self.status_label.setStyleSheet(
            f"color: {DASHBOARD_COLORS['ink']}; font-size: 13px; font-weight: 600;"
        )
        layout.addWidget(self.status_label)

        self.preview_label = QLabel("正在打开摄像头...")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(860, 580)
        self.preview_label.setStyleSheet(
            f"background-color: {DASHBOARD_COLORS['ink']}; color: {DASHBOARD_COLORS['panel']}; "
            f"border-radius: 14px; border: 1px solid {DASHBOARD_COLORS['accent']};"
        )
        layout.addWidget(self.preview_label, 1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        cancel_button = QPushButton("取消")
        self.capture_button = QPushButton("拍照")
        self.capture_button.setEnabled(False)

        cancel_button.clicked.connect(self.reject)
        self.capture_button.clicked.connect(self.capture_current_frame)

        button_row.addWidget(cancel_button)
        button_row.addWidget(self.capture_button)
        layout.addLayout(button_row)

    def start_preview(self):
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if not self.cap.isOpened():
            self.stop_preview()
            raise RuntimeError("无法打开摄像头")
        self.timer.start(33)

    def stop_preview(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_preview_frame(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status_label.setText("摄像头预览中断，请检查摄像头连接")
            self.capture_button.setEnabled(self.latest_frame is not None)
            return

        self.latest_frame = frame
        self.status_label.setText("调整画面后点击拍照")
        self.capture_button.setEnabled(True)
        self.show_preview_frame(frame)

    def show_preview_frame(self, frame):
        image = cv_to_qimage(frame)
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            self.preview_label.setText("预览加载失败")
            self.preview_label.setPixmap(QPixmap())
            return

        target_size = self.preview_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            self.preview_label.setPixmap(pixmap)
            return

        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def capture_current_frame(self):
        if self.latest_frame is None:
            self.status_label.setText("尚未获取到摄像头画面")
            return

        self.captured_frame = self.latest_frame.copy()
        self.accept()

    def done(self, result):
        self.stop_preview()
        super().done(result)

    def resizeEvent(self, event):
        if self.latest_frame is not None:
            self.show_preview_frame(self.latest_frame)
        super().resizeEvent(event)


class ImageDetectionWorker(QObject):
    status_changed = pyqtSignal(str)
    result_ready = pyqtSignal(QImage, list, list, str)
    finished = pyqtSignal()

    def __init__(self, model_path: Path, image_path: str):
        super().__init__()
        self.model_path = str(model_path)
        self.image_path = image_path

    def run(self):
        pool = None
        try:
            if not Path(self.model_path).exists():
                raise RuntimeError(f"模型文件不存在: {self.model_path}")

            frame = cv2.imread(self.image_path)
            if frame is None:
                raise RuntimeError(f"无法读取图片: {self.image_path}")

            self.status_changed.emit("正在加载非量化模型并识别图片...")
            pool = rknnPoolExecutor(
                rknnModel=self.model_path,
                TPEs=THREAD_COUNT,
                func=detect_frame,
            )
            pool.put(frame.copy())
            result, ok = pool.get()
            if not ok or result is None:
                raise RuntimeError("图片识别失败")

            image = cv_to_qimage(result["image"])
            detections = result["detections"]
            top_candidates = result.get("top_candidates", [])[:3]
            summary = "未检测到目标" if not detections else f"检测到 {len(detections)} 个目标"
            self.result_ready.emit(image, detections, top_candidates, summary)
            self.status_changed.emit("图片识别完成")
        except Exception as exc:
            self.status_changed.emit(f"识别失败: {exc}")
        finally:
            if pool is not None:
                pool.release()
            self.finished.emit()


class DetectionPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.worker_thread = None
        self.worker = None
        self.current_image_path = None
        self.preview_label = None
        self.status_label = None
        self.result_label = None
        self.path_label = None
        self.top3_view = None
        self.detail_view = None
        self.import_button = None
        self.capture_button = None
        self.detect_button = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        control_row = QHBoxLayout()
        self.import_button = QPushButton("导入图片")
        self.capture_button = QPushButton("摄像头拍照")
        self.detect_button = QPushButton("开始识别")
        self.detect_button.setEnabled(False)

        self.import_button.clicked.connect(self.import_image)
        self.capture_button.clicked.connect(self.capture_photo)
        self.detect_button.clicked.connect(self.detect_current_image)

        control_row.addWidget(self.import_button)
        control_row.addWidget(self.capture_button)
        control_row.addWidget(self.detect_button)
        control_row.addStretch(1)
        layout.addLayout(control_row)

        self.status_label = QLabel("检测状态: 请选择图片或拍照")
        self.status_label.setStyleSheet(
            f"color: {DASHBOARD_COLORS['ink']}; font-size: 13px; font-weight: 600;"
        )
        layout.addWidget(self.status_label)

        self.path_label = QLabel("当前图片: 未选择")
        self.path_label.setStyleSheet(
            f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 12px;"
        )
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        self.result_label = QLabel("结果概览: --")
        self.result_label.setStyleSheet(
            f"color: {DASHBOARD_COLORS['ink']}; font-size: 13px; font-weight: 600;"
        )
        layout.addWidget(self.result_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        preview_group = QGroupBox("图片预览")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("导入图片或拍照后将在这里显示")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(720, 520)
        self.preview_label.setStyleSheet(
            f"background-color: {DASHBOARD_COLORS['ink']}; color: {DASHBOARD_COLORS['panel']}; "
            f"border-radius: 14px; border: 1px solid {DASHBOARD_COLORS['accent']};"
        )
        preview_layout.addWidget(self.preview_label)

        side_group = QGroupBox("识别结果")
        side_layout = QVBoxLayout(side_group)

        top3_title = QLabel("Top 3 可能性")
        top3_title.setStyleSheet(
            f"color: {DASHBOARD_COLORS['ink']}; font-size: 14px; font-weight: 700;"
        )
        side_layout.addWidget(top3_title)

        self.top3_view = QTextEdit()
        self.top3_view.setReadOnly(True)
        self.top3_view.setMinimumHeight(150)
        side_layout.addWidget(self.top3_view)

        detail_title = QLabel("检测详情")
        detail_title.setStyleSheet(
            f"color: {DASHBOARD_COLORS['ink']}; font-size: 14px; font-weight: 700;"
        )
        side_layout.addWidget(detail_title)

        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        side_layout.addWidget(self.detail_view, 1)

        splitter.addWidget(preview_group)
        splitter.addWidget(side_group)
        splitter.setSizes([920, 360])
        layout.addWidget(splitter, 1)

        self.top3_view.setPlainText("等待识别结果...")
        self.detail_view.setPlainText("检测详情将在这里显示")

    def import_image(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择待识别图片",
            str(BASE_DIR),
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not image_path:
            return
        self.set_current_image(image_path, "已导入图片，点击开始识别")

    def capture_photo(self):
        try:
            dialog = CameraPreviewDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted or dialog.captured_frame is None:
                self.update_status("检测状态: 已取消拍照")
                return

            capture_dir = BASE_DIR / "captures"
            capture_dir.mkdir(exist_ok=True)
            file_path = capture_dir / f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            if not cv2.imwrite(str(file_path), dialog.captured_frame):
                raise RuntimeError("照片保存失败")
            self.set_current_image(str(file_path), "拍照成功，点击开始识别")
        except Exception as exc:
            self.update_status(f"检测状态: 拍照失败 - {exc}")

    def set_current_image(self, image_path, status_text):
        self.current_image_path = image_path
        self.path_label.setText(f"当前图片: {image_path}")
        self.detect_button.setEnabled(True)
        self.result_label.setText("结果概览: 待识别")
        self.top3_view.setPlainText("等待识别结果...")
        self.detail_view.setPlainText("检测详情将在这里显示")
        self.update_status(f"检测状态: {status_text}")

        frame = cv2.imread(image_path)
        if frame is not None:
            self.show_image(cv_to_qimage(frame))

    def detect_current_image(self):
        if not self.current_image_path or self.worker_thread is not None:
            return

        self.worker_thread = QThread(self)
        self.worker = ImageDetectionWorker(MODEL_PATH, self.current_image_path)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.status_changed.connect(self.update_status)
        self.worker.result_ready.connect(self.show_detection_result)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker)

        self.import_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.detect_button.setEnabled(False)
        self.update_status("检测状态: 正在准备识别...")
        self.worker_thread.start()

    def show_detection_result(self, image, detections, top_candidates, summary):
        self.show_image(image)
        self.result_label.setText(f"结果概览: {summary}")
        self.top3_view.setPlainText("\n".join(format_top3_lines(top_candidates)))
        self.detail_view.setPlainText("\n".join(format_detection_lines(detections)))

    def show_image(self, image):
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            self.preview_label.setText("图片加载失败")
            self.preview_label.setPixmap(QPixmap())
            return
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def update_status(self, text):
        if not text.startswith("检测状态:"):
            text = f"检测状态: {text}"
        self.status_label.setText(text)

    def _clear_worker(self):
        self.worker = None
        self.worker_thread = None
        self.import_button.setEnabled(True)
        self.capture_button.setEnabled(True)
        self.detect_button.setEnabled(self.current_image_path is not None)

    def resizeEvent(self, event):
        pixmap = self.preview_label.pixmap()
        if pixmap is not None and not pixmap.isNull():
            scaled = pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled)
        super().resizeEvent(event)

    def shutdown(self):
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(5000)


class PlantSystemWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detection_panel = None
        self.setWindowTitle("智能植物管护系统")
        self.resize(1420, 900)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("智能植物管护系统")
        title.setFont(QFont("Microsoft YaHei", 22, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']};")
        layout.addWidget(title)

        subtitle = QLabel("图片识别 + BLE 环境传感器监控")
        subtitle.setFont(QFont("Microsoft YaHei", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']};")
        layout.addWidget(subtitle)

        tabs = QTabWidget()
        self.detection_panel = DetectionPanel()
        tabs.addTab(self.detection_panel, "图片识别")
        tabs.addTab(SensorPanel(), "传感器监控")
        layout.addWidget(tabs, 1)

        self.setStyleSheet(build_dashboard_stylesheet(include_tabs=True))

    def closeEvent(self, event):
        if self.detection_panel is not None:
            self.detection_panel.shutdown()
        super().closeEvent(event)


def main():
    start_ble_thread()

    app = QApplication(sys.argv)
    app.setApplicationName("智能植物管护系统")

    window = PlantSystemWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
