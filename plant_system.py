import asyncio
import json
import os
import sys
import time
import zipfile
from datetime import datetime
from xml.sax.saxutils import escape
from pathlib import Path

import cv2

# opencv-python injects its bundled Qt5 plugin path into the environment,
# which conflicts with this PyQt6 application at startup.
for qt_env_var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    qt_env_value = os.environ.get(qt_env_var)
    if qt_env_value and "cv2/qt" in qt_env_value:
        os.environ.pop(qt_env_var, None)

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDateTimeEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


BASE_DIR = Path(__file__).resolve().parent
YOLO_DIR = BASE_DIR / "rknn3588-yolov8"
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

from data_store import (  # noqa: E402
    fetch_history_bounds,
    fetch_recognition_records,
    fetch_sensor_records,
    init_database,
    record_recognition_result,
    verify_login,
)
from func import CLASSES, detect_frame, translate_label  # noqa: E402
from rknnpool import rknnPoolExecutor  # noqa: E402
from environment_monitor import (  # noqa: E402
    DASHBOARD_COLORS,
    SensorPanel,
    build_dashboard_stylesheet,
    get_live_monitor_snapshot,
    start_ble_thread,
)
import environment_monitor as env_monitor  # noqa: E402


CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
# The UI only allows one outstanding recognition task at a time,
# so keep a single RKNN instance loaded and reuse it for every frame.
THREAD_COUNT = 1
MODEL_PATH = YOLO_DIR / "rknnModel" / "plant_det.rknn"
DEFAULT_DETECTION_INTERVAL = 1.0
RECOGNITION_RESULT_DIR = BASE_DIR / "recognition_results"
SAMPLE_VIDEO_PATH = BASE_DIR / "plant_det_source.mp4"
HISTORY_CHART_MAX_POINTS = 240
VIDEO_SOURCE_OPTIONS = (
    {
        "label": "本地视频 plant_det_source.mp4",
        "source": str(SAMPLE_VIDEO_PATH),
        "status_name": "本地视频 plant_det_source.mp4",
        "loop_on_end": True,
    },
    {
        "label": "默认摄像头",
        "source": CAMERA_ID,
        "status_name": "默认摄像头",
        "loop_on_end": False,
    },
)
PEST_LABELS = {
    label
    for label in CLASSES[1081:1119]
    if "healthy" not in label.lower()
}
ENVIRONMENT_THRESHOLD_SPECS = (
    {
        "field": "soil_humidity",
        "title": "土壤湿度",
        "unit": "",
        "decimals": 0,
        "step": 10.0,
        "default_min": 500.0,
        "default_max": 3200.0,
        "spin_min": 0.0,
        "spin_max": 4095.0,
        "card_color": "#7b5c2e",
    },
    {
        "field": "air_temperature",
        "title": "空气温度",
        "unit": "C",
        "decimals": 1,
        "step": 0.5,
        "default_min": 18.0,
        "default_max": 30.0,
        "spin_min": -20.0,
        "spin_max": 80.0,
        "card_color": "#d14f45",
    },
    {
        "field": "air_humidity",
        "title": "空气湿度",
        "unit": "%",
        "decimals": 0,
        "step": 1.0,
        "default_min": 35.0,
        "default_max": 80.0,
        "spin_min": 0.0,
        "spin_max": 100.0,
        "card_color": "#4b78c7",
    },
    {
        "field": "wind_speed",
        "title": "风速",
        "unit": "m/s",
        "decimals": 1,
        "step": 0.1,
        "default_min": 0.0,
        "default_max": 8.0,
        "spin_min": 0.0,
        "spin_max": 30.0,
        "card_color": "#7c4cc9",
    },
    {
        "field": "light_lux",
        "title": "光照强度",
        "unit": "lx",
        "decimals": 0,
        "step": 10.0,
        "default_min": 300.0,
        "default_max": 1800.0,
        "spin_min": 0.0,
        "spin_max": 100000.0,
        "card_color": "#8b6a16",
    },
    {
        "field": "pressure_hpa",
        "title": "气压",
        "unit": "hPa",
        "decimals": 1,
        "step": 0.5,
        "default_min": 980.0,
        "default_max": 1030.0,
        "spin_min": 900.0,
        "spin_max": 1100.0,
        "card_color": "#2f8f67",
    },
)
ENVIRONMENT_THRESHOLD_MAP = {
    spec["field"]: spec for spec in ENVIRONMENT_THRESHOLD_SPECS
}


def cv_to_qimage(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()


def format_label_for_display(label):
    if not label:
        return "--"
    return translate_label(label)


def extract_plant_name(label):
    if not label:
        return "未知植物"
    result = translate_label(label)
    if " / " in result:
        return result.split(" / ")[0]
    return result


def build_default_threshold_settings():
    return {
        spec["field"]: {
            "min": float(spec["default_min"]),
            "max": float(spec["default_max"]),
        }
        for spec in ENVIRONMENT_THRESHOLD_SPECS
    }


def preferred_source_index():
    for index, option in enumerate(VIDEO_SOURCE_OPTIONS):
        source = option["source"]
        if isinstance(source, str) and Path(source).exists():
            return index
    for index, option in enumerate(VIDEO_SOURCE_OPTIONS):
        if not option["loop_on_end"]:
            return index
    return 0


def format_metric_value(field, value):
    spec = ENVIRONMENT_THRESHOLD_MAP[field]
    unit = spec["unit"]
    if value is None:
        return "--" if not unit else f"-- {unit}"
    if spec["decimals"] == 0:
        text = f"{float(value):.0f}"
    else:
        text = f"{float(value):.{spec['decimals']}f}"
    return text if not unit else f"{text} {unit}"


def build_environment_alarm_lines(latest_sensor, threshold_settings):
    if not latest_sensor:
        return []

    lines = []
    for spec in ENVIRONMENT_THRESHOLD_SPECS:
        field = spec["field"]
        value = latest_sensor.get(field)
        if value is None:
            continue

        field_threshold = threshold_settings.get(field, {})
        lower = field_threshold.get("min")
        upper = field_threshold.get("max")
        metric_text = format_metric_value(field, value)

        if lower is not None and float(value) < float(lower):
            lines.append(
                f"环境告警: {spec['title']}过低，当前 {metric_text}，低于下限 {format_metric_value(field, lower)}"
            )
        if upper is not None and float(value) > float(upper):
            lines.append(
                f"环境告警: {spec['title']}过高，当前 {metric_text}，高于上限 {format_metric_value(field, upper)}"
            )
    return lines


def build_recognition_content_lines(payload):
    if not payload:
        return ["等待识别结果..."]

    detections = payload.get("detections", [])
    grouped = {}
    total_targets = 0
    for detection in detections:
        label = detection.get("label")
        name = extract_plant_name(label)
        entry = grouped.setdefault(name, {"count": 0, "best_score": 0.0, "has_pest": False})
        entry["count"] += 1
        entry["best_score"] = max(entry["best_score"], float(detection.get("score") or 0.0))
        if label in PEST_LABELS:
            entry["has_pest"] = True
        total_targets += 1

    ts = payload.get("ts")
    lines = []
    if ts:
        lines.append(f"识别时间: {time.strftime('%H:%M:%S', time.localtime(ts))}")

    if not grouped:
        lines.append("当前帧未识别到花卉目标")
        return lines

    lines.append(f"识别到 {total_targets} 个花卉目标，涉及 {len(grouped)} 种对象")
    lines.append("")
    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: (-item[1]["count"], -item[1]["best_score"], item[0]),
    )
    for index, (name, info) in enumerate(sorted_groups, start=1):
        suffix = " | 含病虫害风险" if info["has_pest"] else ""
        lines.append(f"{index}. {name} | 数量 {info['count']}{suffix}")
    return lines


def build_realtime_alert_items(payload, latest_sensor, threshold_settings):
    items = []
    if payload:
        pest_groups = {}
        for detection in payload.get("detections", []):
            label = detection.get("label")
            if label not in PEST_LABELS:
                continue
            entry = pest_groups.setdefault(
                label,
                {
                    "count": 0,
                    "best_score": 0.0,
                    "plant_name": extract_plant_name(label),
                },
            )
            entry["count"] += 1
            entry["best_score"] = max(entry["best_score"], float(detection.get("score") or 0.0))

        pest_time = payload.get("ts")
        pest_time_text = time.strftime("%H:%M:%S", time.localtime(pest_time)) if pest_time else "--"
        for label, info in sorted(
            pest_groups.items(),
            key=lambda item: (-item[1]["best_score"], item[0]),
        ):
            items.append(
                {
                    "title": "病虫害告警",
                    "body": (
                        f"时间 {pest_time_text} | {format_label_for_display(label)} | "
                        f"关联植被 {info['plant_name']} | 数量 {info['count']}"
                    ),
                    "level": "danger",
                }
            )

    for line in build_environment_alarm_lines(latest_sensor, threshold_settings):
        items.append(
            {
                "title": "环境告警",
                "body": line.replace("环境告警: ", ""),
                "level": "danger",
            }
        )

    if not items:
        return [
            {
                "title": "运行状态",
                "body": "当前无实时报警",
                "level": "normal",
            }
        ]
    return items


def format_top3_lines(candidates):
    if not candidates:
        return ["未获得候选类别"]
    return [
        f"Top {index}. {format_label_for_display(item['label'])}"
        for index, item in enumerate(candidates[:3], start=1)
    ]


def format_detection_lines(detections):
    if not detections:
        return ["当前识别帧未检测到明显目标框"]
    lines = []
    for index, detection in enumerate(detections[:8], start=1):
        x1, y1, x2, y2 = detection["box"]
        lines.append(
            f"{index}. {format_label_for_display(detection['label'])} | ({x1}, {y1})-({x2}, {y2})"
        )
    if len(detections) > 8:
        lines.append(f"... 其余 {len(detections) - 8} 个结果已省略")
    return lines


def choose_plant_candidate(detections, top_candidates):
    for candidate in detections:
        if candidate["label"] not in PEST_LABELS:
            return {"label": candidate["label"], "score": float(candidate["score"])}
    for candidate in top_candidates:
        if candidate["label"] not in PEST_LABELS:
            return {"label": candidate["label"], "score": float(candidate["score"])}
    if detections:
        fallback = detections[0]
        return {"label": fallback["label"], "score": float(fallback["score"])}
    if top_candidates:
        fallback = top_candidates[0]
        return {"label": fallback["label"], "score": float(fallback["score"])}
    return None


def build_recognition_payload(detections, top_candidates, interval_seconds, username):
    top_candidates = top_candidates[:3]
    ts = time.time()
    pest_detections = sorted(
        (item for item in detections if item["label"] in PEST_LABELS),
        key=lambda item: item["score"],
        reverse=True,
    )
    plant_candidate = choose_plant_candidate(detections, top_candidates)

    payload = {
        "ts": ts,
        "username": username,
        "interval_seconds": interval_seconds,
        "detections": detections,
        "top_candidates": top_candidates,
    }

    if pest_detections:
        best_pest = pest_detections[0]
        payload.update(
            {
                "summary": "识别到病虫害",
                "pest_detected": True,
                "pest_label": best_pest["label"],
                "pest_confidence": float(best_pest["score"]),
                "plant_type": extract_plant_name(best_pest["label"]),
                "plant_confidence": None,
            }
        )
        payload["summary_lines"] = [
            "状态: 识别到病虫害",
            f"病虫害: {format_label_for_display(best_pest['label'])}",
            f"置信度: {best_pest['score']:.2f}",
            f"关联植物: {extract_plant_name(best_pest['label'])}",
            f"识别时间: {time.strftime('%H:%M:%S', time.localtime(ts))}",
        ]
        return payload

    plant_label = plant_candidate["label"] if plant_candidate else None
    plant_score = float(plant_candidate["score"]) if plant_candidate else None
    payload.update(
        {
            "summary": "未识别到病虫害",
            "pest_detected": False,
            "pest_label": None,
            "pest_confidence": None,
            "plant_type": extract_plant_name(plant_label),
            "plant_confidence": plant_score,
        }
    )
    payload["summary_lines"] = [
        "状态: 未识别到病虫害",
        f"植物类型: {payload['plant_type']}",
        f"置信度: {plant_score:.2f}" if plant_score is not None else "置信度: --",
        f"识别时间: {time.strftime('%H:%M:%S', time.localtime(ts))}",
    ]
    return payload


def save_recognition_image(image, timestamp):
    RECOGNITION_RESULT_DIR.mkdir(exist_ok=True)
    file_path = RECOGNITION_RESULT_DIR / f"recognition_{time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp))}_{int((timestamp % 1) * 1000):03d}.jpg"
    if not cv2.imwrite(str(file_path), image):
        raise RuntimeError("识别结果图片保存失败")
    return str(file_path)


def qimage_to_pixmap(image, label_size):
    pixmap = QPixmap.fromImage(image)
    if pixmap.isNull():
        return QPixmap()
    return pixmap.scaled(
        label_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def timestamp_to_qdatetime(timestamp):
    return QDateTimeEdit().dateTime().fromSecsSinceEpoch(int(timestamp))


def qdatetime_to_timestamp(widget):
    return float(widget.dateTime().toSecsSinceEpoch())


def export_rows_to_csv(file_path, headers, rows):
    lines = [",".join(_csv_escape_cell(header) for header in headers)]
    for row in rows:
        lines.append(",".join(_csv_escape_cell(value) for value in row))
    Path(file_path).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _csv_escape_cell(value):
    text = "" if value is None else str(value)
    if any(char in text for char in [',', '"', '\n']):
        return '"' + text.replace('"', '""') + '"'
    return text


def export_rows_to_xlsx(file_path, sheet_name, headers, rows):
    shared_strings = []
    shared_index = {}

    def register_shared_string(value):
        text = "" if value is None else str(value)
        if text not in shared_index:
            shared_index[text] = len(shared_strings)
            shared_strings.append(text)
        return shared_index[text]

    all_rows = [headers, *rows]
    sheet_rows = []
    for row_number, row in enumerate(all_rows, start=1):
        cells = []
        for col_number, value in enumerate(row, start=1):
            ref = f"{_excel_column_name(col_number)}{row_number}"
            shared_id = register_shared_string(value)
            cells.append(f'<c r="{ref}" t="s"><v>{shared_id}</v></c>')
        sheet_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        f'{"".join(sheet_rows)}'
        '</sheetData>'
        '</worksheet>'
    )
    shared_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(shared_strings)}" uniqueCount="{len(shared_strings)}">'
        + "".join(f"<si><t>{escape(text)}</t></si>" for text in shared_strings)
        + '</sst>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
        'Target="sharedStrings.xml"/>'
        '</Relationships>'
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
        '<cellXfs count="1"><xf xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )

    with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", root_rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        archive.writestr("xl/sharedStrings.xml", shared_xml)
        archive.writestr("xl/styles.xml", styles_xml)


def _excel_column_name(index):
    result = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.username_edit = None
        self.password_edit = None
        self.error_label = None
        self._build_ui()

    def _build_ui(self):
        self.setObjectName("loginDialog")
        self.setWindowTitle("系统登录")
        self.setModal(True)
        self.resize(460, 280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("智能植物管护系统登录")
        title.setFont(QFont("Microsoft YaHei", 22, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("默认账号: admin  默认密码: admin123")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 15px;")
        layout.addWidget(subtitle)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(14)

        self.username_edit = QLineEdit("admin")
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.returnPressed.connect(self._try_login)

        form.addRow("用户名", self.username_edit)
        form.addRow("密码", self.password_edit)
        layout.addLayout(form)

        self.error_label = QLabel("请输入账号密码后登录")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 14px;")
        layout.addWidget(self.error_label)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        cancel_button = QPushButton("退出")
        login_button = QPushButton("登录")
        cancel_button.setProperty("variant", "secondary")

        cancel_button.clicked.connect(self.reject)
        login_button.clicked.connect(self._try_login)

        button_row.addWidget(cancel_button)
        button_row.addWidget(login_button)
        layout.addLayout(button_row)

        self.setStyleSheet(
            build_dashboard_stylesheet(include_tabs=False)
            + f"""
            QDialog#loginDialog {{
                background-color: {DASHBOARD_COLORS['bg']};
            }}
            """
        )

    @property
    def username(self):
        return self.username_edit.text().strip()

    def _try_login(self):
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        if not username or not password:
            self.error_label.setText("用户名和密码不能为空")
            self.error_label.setStyleSheet("color: #b3403b; font-size: 14px;")
            return

        if verify_login(username, password):
            self.accept()
            return

        self.error_label.setText("用户名或密码错误")
        self.error_label.setStyleSheet("color: #b3403b; font-size: 14px;")


class VideoRecognitionWorker(QObject):
    status_changed = pyqtSignal(str)
    ready_changed = pyqtSignal(bool, str)
    result_ready = pyqtSignal(QImage, object)
    cycle_finished = pyqtSignal()

    def __init__(self, model_path: Path, username: str):
        super().__init__()
        self.model_path = str(model_path)
        self.username = username
        self.pool = None

    @pyqtSlot()
    def setup(self):
        try:
            if not Path(self.model_path).exists():
                raise RuntimeError(f"模型文件不存在: {self.model_path}")

            self.status_changed.emit("检测状态: 正在加载识别模型...")
            self.pool = rknnPoolExecutor(
                rknnModel=self.model_path,
                TPEs=THREAD_COUNT,
                func=detect_frame,
            )
            self.ready_changed.emit(True, "检测状态: 主画面识别模型已就绪")
        except Exception as exc:  # pylint: disable=broad-except
            self.ready_changed.emit(False, f"检测状态: 模型加载失败 - {exc}")

    @pyqtSlot(object, float)
    def process_frame(self, frame, interval_seconds):
        try:
            if self.pool is None:
                raise RuntimeError("识别模型尚未初始化")

            self.status_changed.emit("检测状态: 正在识别当前视频帧...")
            self.pool.put(frame.copy())
            result, ok = self.pool.get()
            if not ok or result is None:
                raise RuntimeError("视频帧识别失败")

            detections = result["detections"]
            top_candidates = result.get("top_candidates", [])
            payload = build_recognition_payload(detections, top_candidates, interval_seconds, self.username)
            payload["image_path"] = save_recognition_image(result["image"], payload["ts"])
            record_recognition_result(payload)
            self.result_ready.emit(cv_to_qimage(result["image"]), payload)
            self.status_changed.emit("检测状态: 本轮识别完成")
        except Exception as exc:  # pylint: disable=broad-except
            self.status_changed.emit(f"检测状态: 识别失败 - {exc}")
        finally:
            self.cycle_finished.emit()

    def shutdown(self):
        if self.pool is not None:
            self.pool.release()
            self.pool = None


class VideoDetectionPanel(QWidget):
    detect_requested = pyqtSignal(object, float)

    def __init__(self, username: str, threshold_settings: dict):
        super().__init__()
        self.username = username
        self.threshold_settings = threshold_settings
        self.capture = None
        self.panel_active = False
        self.latest_frame = None
        self.latest_display_image = None
        self.recognition_pending = False
        self.worker_ready = False
        self.worker_thread = None
        self.worker = None
        self.last_payload = None

        self.source_combo = None
        self.result_image_label = None
        self.time_label = None
        self.recognition_content_view = None
        self.alert_scroll = None
        self.alert_cards_layout = None
        self.alert_empty_label = None
        self.interval_spin = None
        self.environment_status_label = None
        self.environment_value_labels = {}
        self.active_source_config = None

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._refresh_video_frame)
        self.detect_timer = QTimer(self)
        self.detect_timer.timeout.connect(self._request_detection)
        self.environment_timer = QTimer(self)
        self.environment_timer.timeout.connect(self._refresh_environment_panel)
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self._update_time)

        self._build_ui()
        self._setup_worker()
        self._refresh_environment_panel()

    def set_active(self, active: bool):
        self.panel_active = active
        if active:
            self._refresh_environment_panel()
            self._update_time()
            if not self.environment_timer.isActive():
                self.environment_timer.start(1000)
            if not self.time_timer.isActive():
                self.time_timer.start(1000)
            if self.capture is None:
                QTimer.singleShot(0, self.start_stream)
            else:
                if not self.preview_timer.isActive():
                    self.preview_timer.start(33)
                self._restart_detection_timer()
        else:
            self.preview_timer.stop()
            self.detect_timer.stop()
            self.environment_timer.stop()
            self.time_timer.stop()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        control_row = QHBoxLayout()
        control_row.setSpacing(12)

        source_label = QLabel("视频源")
        source_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 13px;")

        self.source_combo = QComboBox()
        for option in VIDEO_SOURCE_OPTIONS:
            self.source_combo.addItem(option["label"])
        self.source_combo.setCurrentIndex(preferred_source_index())
        self.source_combo.currentIndexChanged.connect(self._handle_source_changed)

        interval_label = QLabel("识别间隔(秒)")
        interval_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 13px;")

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setDecimals(1)
        self.interval_spin.setRange(0.5, 10.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setValue(DEFAULT_DETECTION_INTERVAL)
        self.interval_spin.valueChanged.connect(self._restart_detection_timer)

        self.time_label = QLabel("--:--:--")
        self.time_label.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']}; font-size: 16px; font-weight: 700;")

        user_label = QLabel(f"当前用户: {self.username}")
        user_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 13px;")

        control_row.addWidget(source_label)
        control_row.addWidget(self.source_combo)
        control_row.addSpacing(12)
        control_row.addWidget(interval_label)
        control_row.addWidget(self.interval_spin)
        control_row.addStretch(1)
        control_row.addWidget(self.time_label)
        control_row.addSpacing(16)
        control_row.addWidget(user_label)
        layout.addLayout(control_row)

        auto_hint = QLabel("进入页面后自动开始识别，优先使用本地视频流，不可用时回退到摄像头。")
        auto_hint.setWordWrap(True)
        auto_hint.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 13px;")
        layout.addWidget(auto_hint)

        result_group = QGroupBox("识别画面")
        result_layout = QVBoxLayout(result_group)
        self.result_image_label = QLabel("正在接入视频流并等待识别结果...")
        self.result_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image_label.setMinimumSize(320, 240)
        self.result_image_label.setStyleSheet(
            f"background-color: {DASHBOARD_COLORS['card']}; color: {DASHBOARD_COLORS['muted_dark']}; "
            f"border-radius: 14px; border: 1px solid {DASHBOARD_COLORS['border']};"
        )
        result_layout.addWidget(self.result_image_label)

        environment_group = QGroupBox("环境信息")
        environment_layout = QVBoxLayout(environment_group)
        environment_layout.setSpacing(8)
        environment_grid = QGridLayout()
        environment_grid.setHorizontalSpacing(8)
        environment_grid.setVerticalSpacing(8)

        card_definitions = [
            ("air_temperature", "空气温度", ENVIRONMENT_THRESHOLD_MAP["air_temperature"]["card_color"]),
            ("air_humidity", "空气湿度", ENVIRONMENT_THRESHOLD_MAP["air_humidity"]["card_color"]),
            ("soil_humidity", "土壤湿度", ENVIRONMENT_THRESHOLD_MAP["soil_humidity"]["card_color"]),
            ("wind_speed", "风速", ENVIRONMENT_THRESHOLD_MAP["wind_speed"]["card_color"]),
            ("light_lux", "光照强度", ENVIRONMENT_THRESHOLD_MAP["light_lux"]["card_color"]),
            ("pressure_hpa", "气压", ENVIRONMENT_THRESHOLD_MAP["pressure_hpa"]["card_color"]),
            ("fan_pwm_percent", "风扇速度", DASHBOARD_COLORS["secondary"]),
        ]

        for index, (field, title, color) in enumerate(card_definitions):
            card, value_label = self._create_environment_card(title, "--", color)
            self.environment_value_labels[field] = value_label
            environment_grid.addWidget(card, index // 2, index % 2)

        self.environment_status_label = QLabel("等待环境数据...")
        self.environment_status_label.setWordWrap(True)
        self.environment_status_label.setStyleSheet(
            f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 13px; line-height: 1.4;"
        )

        environment_layout.addLayout(environment_grid)
        environment_layout.addStretch(1)
        environment_layout.addWidget(self.environment_status_label)

        content_group = QGroupBox("识别内容")
        content_layout = QVBoxLayout(content_group)
        self.recognition_content_view = QTextEdit()
        self.recognition_content_view.setReadOnly(True)
        self.recognition_content_view.setMinimumHeight(80)
        self.recognition_content_view.setPlainText("等待识别结果...")
        content_layout.addWidget(self.recognition_content_view)

        alert_group = QGroupBox("实时报警")
        alert_layout = QVBoxLayout(alert_group)
        self.alert_empty_label = QLabel("当前无实时报警")
        self.alert_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alert_empty_label.setMinimumHeight(80)
        self.alert_empty_label.setStyleSheet(
            "background-color: #fff6f6; color: #7f8b85; border: 1px dashed #efb0b0; border-radius: 14px;"
        )

        self.alert_scroll = QScrollArea()
        self.alert_scroll.setObjectName("alertScroll")
        self.alert_scroll.setWidgetResizable(True)
        self.alert_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.alert_scroll.setMinimumHeight(80)
        self.alert_scroll.viewport().setObjectName("alertScrollViewport")
        self.alert_scroll.setStyleSheet(
            f"""
            QScrollArea#alertScroll,
            QWidget#alertScrollViewport,
            QWidget#alertCardsHost {{
                background-color: {DASHBOARD_COLORS['panel']};
            }}
            QScrollArea#alertScroll {{
                border: 1px solid {DASHBOARD_COLORS['border']};
                border-radius: 14px;
            }}
            """
        )

        alert_cards_host = QWidget()
        alert_cards_host.setObjectName("alertCardsHost")
        self.alert_cards_layout = QVBoxLayout(alert_cards_host)
        self.alert_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.alert_cards_layout.setSpacing(10)
        self.alert_cards_layout.addStretch(1)
        self.alert_scroll.setWidget(alert_cards_host)

        alert_layout.addWidget(self.alert_empty_label)
        alert_layout.addWidget(self.alert_scroll)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.addWidget(result_group)
        top_splitter.addWidget(environment_group)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.setChildrenCollapsible(False)
        bottom_splitter.addWidget(content_group)
        bottom_splitter.addWidget(alert_group)
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)

        main_vsplit = QSplitter(Qt.Orientation.Vertical)
        main_vsplit.setChildrenCollapsible(False)
        main_vsplit.addWidget(top_splitter)
        main_vsplit.addWidget(bottom_splitter)
        main_vsplit.setStretchFactor(0, 1)
        main_vsplit.setStretchFactor(1, 1)
        layout.addWidget(main_vsplit, 1)
        self._render_alert_cards(build_realtime_alert_items(None, None, self.threshold_settings))

    def _create_environment_card(self, title, value, color):
        frame = QFrame()
        frame.setStyleSheet(
            f"background-color: {DASHBOARD_COLORS['panel']}; border: 1px solid {DASHBOARD_COLORS['border']}; border-radius: 12px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted']}; font-size: 13px;")
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setWordWrap(True)
        value_label.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(value_label)
        layout.addStretch(1)
        return frame, value_label

    def _setup_worker(self):
        self.worker_thread = QThread(self)
        self.worker = VideoRecognitionWorker(MODEL_PATH, self.username)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.setup)
        self.detect_requested.connect(self.worker.process_frame)
        self.worker.ready_changed.connect(self._handle_worker_ready)
        self.worker.status_changed.connect(self.update_status)
        self.worker.result_ready.connect(self._handle_detection_result)
        self.worker.cycle_finished.connect(self._mark_detection_idle)
        self.worker_thread.start()

    def _handle_worker_ready(self, ok, message):
        self.worker_ready = ok
        self.update_status(message)

    def _current_source_config(self):
        if self.source_combo is None:
            return VIDEO_SOURCE_OPTIONS[0]

        index = self.source_combo.currentIndex()
        if index < 0 or index >= len(VIDEO_SOURCE_OPTIONS):
            return VIDEO_SOURCE_OPTIONS[0]
        return VIDEO_SOURCE_OPTIONS[index]

    def _set_source_index(self, index):
        if self.source_combo is None:
            return
        self.source_combo.blockSignals(True)
        self.source_combo.setCurrentIndex(index)
        self.source_combo.blockSignals(False)

    def _open_capture_for_config(self, source_config):
        source = source_config["source"]
        if source_config["loop_on_end"] and not Path(source).exists():
            return None

        capture = cv2.VideoCapture(source)
        if not source_config["loop_on_end"]:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if not capture.isOpened():
            capture.release()
            return None
        return capture

    def _candidate_source_configs(self):
        primary = self._current_source_config()
        candidates = [primary]
        if primary["loop_on_end"]:
            candidates.extend(
                option
                for option in VIDEO_SOURCE_OPTIONS
                if option["source"] != primary["source"] and not option["loop_on_end"]
            )
        return candidates

    def _reset_monitor_texts(self):
        self.last_payload = None
        self.latest_display_image = None
        if self.result_image_label is not None:
            self.result_image_label.setPixmap(QPixmap())
            self.result_image_label.setText("正在接入视频流并等待识别结果...")
        if self.recognition_content_view is not None:
            self.recognition_content_view.setPlainText("等待识别结果...")

    def start_stream(self):
        if not self.panel_active:
            return
        if self.capture is not None:
            return

        self._reset_monitor_texts()
        capture = None
        chosen_config = None
        chosen_index = None
        fallback_used = False

        for candidate in self._candidate_source_configs():
            capture = self._open_capture_for_config(candidate)
            if capture is None:
                continue
            chosen_config = candidate
            chosen_index = next(
                (index for index, option in enumerate(VIDEO_SOURCE_OPTIONS) if option["source"] == candidate["source"]),
                0,
            )
            fallback_used = candidate["source"] != self._current_source_config()["source"]
            break

        if capture is None or chosen_config is None:
            source_config = self._current_source_config()
            self.update_status(f"检测状态: 无法打开{source_config['status_name']}")
            self._refresh_environment_panel()
            return

        self.capture = capture
        self.active_source_config = chosen_config
        self.latest_frame = None
        self.preview_timer.start(33)
        self._restart_detection_timer()

        if chosen_index is not None and chosen_index != self.source_combo.currentIndex():
            self._set_source_index(chosen_index)

        if fallback_used:
            self.update_status(
                f"检测状态: 本地视频流不可用，已切换到{chosen_config['status_name']}并开始自动识别"
            )
        else:
            self.update_status(f"检测状态: {chosen_config['status_name']}已启动，正在自动识别")

    def stop_stream(self, checked=False, status_text="检测状态: 视频流已停止"):
        self.preview_timer.stop()
        self.detect_timer.stop()
        self.recognition_pending = False
        self.latest_frame = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.active_source_config = None
        if status_text:
            self.update_status(status_text)

    def _handle_source_changed(self, _index):
        if self.capture is not None:
            self.stop_stream(status_text="检测状态: 正在切换视频源...")
        if self.panel_active:
            self.start_stream()

    def _restart_detection_timer(self):
        if self.capture is None or self.interval_spin is None:
            return
        self.detect_timer.stop()
        self.detect_timer.start(int(float(self.interval_spin.value()) * 1000))

    def _refresh_video_frame(self):
        if self.capture is None:
            return

        source_config = self.active_source_config or self._current_source_config()
        ok, frame = self.capture.read()
        if (not ok or frame is None) and source_config["loop_on_end"]:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if not ok or frame is None:
            self.stop_stream(status_text=f"检测状态: {source_config['status_name']}读取失败")
            if self.panel_active:
                QTimer.singleShot(400, self.start_stream)
            return

        self.latest_frame = frame
        if self.last_payload is None:
            self._show_image_on_label(self.result_image_label, cv_to_qimage(frame))

    def _request_detection(self):
        if not self.panel_active or not self.worker_ready or self.latest_frame is None or self.recognition_pending:
            return
        self.recognition_pending = True
        self.detect_requested.emit(self.latest_frame.copy(), float(self.interval_spin.value()))

    def _mark_detection_idle(self):
        self.recognition_pending = False

    def _handle_detection_result(self, image, payload):
        self.last_payload = payload
        self._show_image_on_label(self.result_image_label, image)
        if self.recognition_content_view is not None:
            self.recognition_content_view.setPlainText("\n".join(build_recognition_content_lines(payload)))
        self._refresh_environment_panel()

    def _show_image_on_label(self, label, image):
        self.latest_display_image = image
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            label.setText("图片加载失败")
            label.setPixmap(QPixmap())
            return

        scaled = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setText("")
        label.setPixmap(scaled)

    def _refresh_environment_panel(self):
        status_snapshot, _snapshot, _fan_snapshot, latest_sensor = get_live_monitor_snapshot()

        for field, value_label in self.environment_value_labels.items():
            if field == "fan_pwm_percent":
                if latest_sensor and latest_sensor.get("fan_pwm_percent") is not None:
                    value_label.setText(f"{latest_sensor.get('fan_pwm_percent')} %")
                else:
                    value_label.setText("-- %")
                continue

            value = latest_sensor.get(field) if latest_sensor else None
            value_label.setText(format_metric_value(field, value))

        packet_text = "--"
        if latest_sensor and latest_sensor.get("ts"):
            packet_text = time.strftime("%H:%M:%S", time.localtime(latest_sensor["ts"]))

        sensor_state_text = "--"
        if latest_sensor:
            sensor_state_text = (
                f"风速 {latest_sensor.get('wind', '--')} | "
                f"AHT20 {latest_sensor.get('aht20', '--')} | "
                f"BH1750 {latest_sensor.get('bh1750', '--')} | "
                f"BMP280 {latest_sensor.get('bmp280', '--')}"
            )

        self.environment_status_label.setText(
            "\n".join(
                [
                    f"BLE状态: {status_snapshot.get('state', '--')} | {status_snapshot.get('message', '--')}",
                    f"最后上报: {packet_text}",
                    f"传感器状态: {sensor_state_text}",
                ]
            )
        )

        self._render_alert_cards(
            build_realtime_alert_items(self.last_payload, latest_sensor, self.threshold_settings)
        )

    def _render_alert_cards(self, items):
        if self.alert_cards_layout is None or self.alert_scroll is None or self.alert_empty_label is None:
            return

        while self.alert_cards_layout.count() > 1:
            item = self.alert_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        has_danger = any(item.get("level") == "danger" for item in items)
        self.alert_empty_label.setVisible(not has_danger and len(items) == 1)
        self.alert_scroll.setVisible(has_danger)

        if not has_danger:
            if items:
                self.alert_empty_label.setText(items[0].get("body", "当前无实时报警"))
            return

        for item in items:
            if item.get("level") != "danger":
                continue
            self.alert_cards_layout.insertWidget(self.alert_cards_layout.count() - 1, self._create_alert_card(item))

    def _create_alert_card(self, item):
        frame = QFrame()
        frame.setStyleSheet(
            "background-color: #fff1f1; border: 1px solid #ef9a9a; border-radius: 14px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title_label = QLabel(item.get("title", "实时报警"))
        title_label.setStyleSheet("color: #b42318; font-size: 16px; font-weight: 800;")
        layout.addWidget(title_label)

        body_label = QLabel(item.get("body", ""))
        body_label.setWordWrap(True)
        body_label.setStyleSheet("color: #7a271a; font-size: 14px; line-height: 1.6;")
        layout.addWidget(body_label)
        return frame

    def update_status(self, text):
        pass

    def _update_time(self):
        if self.time_label is not None:
            self.time_label.setText(time.strftime("%Y-%m-%d %H:%M:%S"))

    def resizeEvent(self, event):
        if self.latest_display_image is not None and self.result_image_label is not None:
            self._show_image_on_label(self.result_image_label, self.latest_display_image)
        super().resizeEvent(event)

    def shutdown(self):
        self.environment_timer.stop()
        self.time_timer.stop()
        self.stop_stream()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(5000)
        if self.worker is not None:
            self.worker.shutdown()


class EnvironmentControlPanel(QWidget):
    fan_command_finished = pyqtSignal(bool, str, int)

    def __init__(self, threshold_settings: dict):
        super().__init__()
        self.threshold_settings = threshold_settings
        self.panel_active = False
        self.threshold_widgets = {}
        self.current_value_labels = {}
        self.status_label = None
        self.fan_slider = None
        self.fan_slider_value = None
        self.fan_hint_label = None
        self.apply_fan_button = None
        self.stop_fan_button = None
        self.fan_request_pending = False
        self.fan_slider_dirty = False
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)

        self._build_ui()
        self.fan_command_finished.connect(self._handle_fan_command_finished)
        self.refresh_data()

    def set_active(self, active: bool):
        self.panel_active = active
        if active:
            self.refresh_data()
            if not self.refresh_timer.isActive():
                self.refresh_timer.start(1000)
        else:
            self.refresh_timer.stop()

    def _build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.setObjectName("environmentControlRoot")
        self.setStyleSheet(
            f"""
            QWidget#environmentControlRoot,
            QWidget#environmentControlViewport,
            QWidget#environmentControlContent {{
                background-color: {DASHBOARD_COLORS['bg']};
            }}
            QScrollArea#environmentControlScroll {{
                background-color: {DASHBOARD_COLORS['bg']};
                border: 0;
            }}
            """
        )

        scroll = QScrollArea()
        scroll.setObjectName("environmentControlScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.viewport().setObjectName("environmentControlViewport")

        content = QWidget()
        content.setObjectName("environmentControlContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("环境阈值与管控控制")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']};")
        layout.addWidget(title)

        self.status_label = QLabel("BLE状态: 等待连接...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 14px;")
        layout.addWidget(self.status_label)

        threshold_group = QGroupBox("环境报警阈值设置")
        threshold_layout = QVBoxLayout(threshold_group)
        threshold_hint = QLabel("主画面中的实时报警会根据这里设置的上下限判断环境是否超标。")
        threshold_hint.setWordWrap(True)
        threshold_hint.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 13px;")
        threshold_layout.addWidget(threshold_hint)

        threshold_grid = QGridLayout()
        threshold_grid.setHorizontalSpacing(12)
        threshold_grid.setVerticalSpacing(12)
        threshold_grid.addWidget(QLabel("环境项"), 0, 0)
        threshold_grid.addWidget(QLabel("当前值"), 0, 1)
        threshold_grid.addWidget(QLabel("下限"), 0, 2)
        threshold_grid.addWidget(QLabel("上限"), 0, 3)

        for row, spec in enumerate(ENVIRONMENT_THRESHOLD_SPECS, start=1):
            title_label = QLabel(spec["title"])
            title_label.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']}; font-size: 14px; font-weight: 700;")
            threshold_grid.addWidget(title_label, row, 0)

            current_value_label = QLabel(format_metric_value(spec["field"], None))
            current_value_label.setStyleSheet(f"color: {spec['card_color']}; font-size: 14px; font-weight: 700;")
            threshold_grid.addWidget(current_value_label, row, 1)
            self.current_value_labels[spec["field"]] = current_value_label

            min_spin = QDoubleSpinBox()
            min_spin.setDecimals(spec["decimals"])
            min_spin.setSingleStep(spec["step"])
            min_spin.setRange(spec["spin_min"], spec["spin_max"])
            min_spin.setValue(float(self.threshold_settings[spec["field"]]["min"]))
            min_spin.valueChanged.connect(
                lambda value, field=spec["field"]: self._update_threshold(field, "min", value)
            )

            max_spin = QDoubleSpinBox()
            max_spin.setDecimals(spec["decimals"])
            max_spin.setSingleStep(spec["step"])
            max_spin.setRange(spec["spin_min"], spec["spin_max"])
            max_spin.setValue(float(self.threshold_settings[spec["field"]]["max"]))
            max_spin.valueChanged.connect(
                lambda value, field=spec["field"]: self._update_threshold(field, "max", value)
            )

            threshold_grid.addWidget(min_spin, row, 2)
            threshold_grid.addWidget(max_spin, row, 3)
            self.threshold_widgets[spec["field"]] = {
                "min": min_spin,
                "max": max_spin,
            }

        threshold_layout.addLayout(threshold_grid)

        reset_button = QPushButton("恢复默认阈值")
        reset_button.setProperty("variant", "secondary")
        reset_button.clicked.connect(self._reset_thresholds)
        threshold_layout.addWidget(reset_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(threshold_group)

        control_group = QGroupBox("管控控制")
        control_layout = QHBoxLayout(control_group)
        control_layout.setSpacing(14)
        control_layout.addWidget(self._create_fan_control_card(), 2)
        control_layout.addWidget(self._create_reserved_control_card(), 1)
        layout.addWidget(control_group)
        layout.addStretch(1)

        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

    def _create_fan_control_card(self):
        frame = QFrame()
        frame.setStyleSheet(
            f"background-color: {DASHBOARD_COLORS['panel']}; border: 1px solid {DASHBOARD_COLORS['border']}; border-radius: 12px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title_label = QLabel("风扇速度控制")
        title_label.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']}; font-size: 16px; font-weight: 700;")
        layout.addWidget(title_label)

        self.fan_slider_value = QLabel("0 %")
        self.fan_slider_value.setFont(QFont("Microsoft YaHei", 28, QFont.Weight.Bold))
        self.fan_slider_value.setStyleSheet(f"color: {DASHBOARD_COLORS['secondary']};")
        layout.addWidget(self.fan_slider_value)

        self.fan_hint_label = QLabel("通过 BLE 设置风扇转速")
        self.fan_hint_label.setWordWrap(True)
        self.fan_hint_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 14px;")
        layout.addWidget(self.fan_hint_label)

        self.fan_slider = QSlider(Qt.Orientation.Horizontal)
        self.fan_slider.setRange(0, 100)
        self.fan_slider.setSingleStep(1)
        self.fan_slider.setValue(0)
        self.fan_slider.valueChanged.connect(self._handle_slider_change)
        layout.addWidget(self.fan_slider)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.apply_fan_button = QPushButton("应用风扇速度")
        self.stop_fan_button = QPushButton("停止风扇")
        self.stop_fan_button.setProperty("variant", "secondary")
        for btn in (self.apply_fan_button, self.stop_fan_button):
            btn.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']};")
        self.apply_fan_button.clicked.connect(lambda: self._send_fan_pwm(self.fan_slider.value()))
        self.stop_fan_button.clicked.connect(self._stop_fan)
        button_row.addWidget(self.apply_fan_button)
        button_row.addWidget(self.stop_fan_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        return frame

    def _create_reserved_control_card(self):
        frame = QFrame()
        frame.setStyleSheet(
            f"background-color: {DASHBOARD_COLORS['panel']}; border: 1px solid {DASHBOARD_COLORS['border']}; border-radius: 12px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title_label = QLabel("扩展控制预留")
        title_label.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']}; font-size: 16px; font-weight: 700;")
        layout.addWidget(title_label)

        info_label = QLabel("后续可以在这里扩展补光、喷淋、遮阳或其他执行器控制。")
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 14px; line-height: 1.6;")
        layout.addWidget(info_label)
        layout.addStretch(1)
        return frame

    def _update_threshold(self, field, boundary, value):
        widgets = self.threshold_widgets[field]
        min_spin = widgets["min"]
        max_spin = widgets["max"]

        if boundary == "min" and float(value) > float(max_spin.value()):
            max_spin.setValue(float(value))
        if boundary == "max" and float(value) < float(min_spin.value()):
            min_spin.setValue(float(value))

        self.threshold_settings[field]["min"] = float(min_spin.value())
        self.threshold_settings[field]["max"] = float(max_spin.value())

    def _reset_thresholds(self):
        defaults = build_default_threshold_settings()
        for field, widgets in self.threshold_widgets.items():
            widgets["min"].blockSignals(True)
            widgets["max"].blockSignals(True)
            widgets["min"].setValue(float(defaults[field]["min"]))
            widgets["max"].setValue(float(defaults[field]["max"]))
            widgets["min"].blockSignals(False)
            widgets["max"].blockSignals(False)
            self.threshold_settings[field]["min"] = float(defaults[field]["min"])
            self.threshold_settings[field]["max"] = float(defaults[field]["max"])

    def _handle_slider_change(self, value):
        self.fan_slider_dirty = True
        if self.fan_slider_value is not None:
            self.fan_slider_value.setText(f"{value} %")

    def _sync_slider(self, value):
        if self.fan_slider is None or self.fan_slider_value is None:
            return
        self.fan_slider.blockSignals(True)
        self.fan_slider.setValue(value)
        self.fan_slider.blockSignals(False)
        self.fan_slider_value.setText(f"{value} %")

    def _send_fan_pwm(self, percent):
        if self.fan_request_pending:
            return
        if env_monitor.ble_loop is None:
            if self.fan_hint_label is not None:
                self.fan_hint_label.setText("BLE 循环尚未就绪")
            return

        _status_snapshot, _snapshot, fan_snapshot, _latest = get_live_monitor_snapshot()
        if not fan_snapshot.get("supported", False):
            if self.fan_hint_label is not None:
                self.fan_hint_label.setText(f"风扇控制不可用: {fan_snapshot.get('message', '--')}")
            return

        self.fan_request_pending = True
        if self.fan_hint_label is not None:
            self.fan_hint_label.setText(f"正在发送 {percent}%...")
        if self.apply_fan_button is not None:
            self.apply_fan_button.setEnabled(False)
        if self.stop_fan_button is not None:
            self.stop_fan_button.setEnabled(False)

        future = asyncio.run_coroutine_threadsafe(env_monitor.write_fan_pwm(percent), env_monitor.ble_loop)
        future.add_done_callback(lambda done, target=percent: self._fan_future_done(target, done))

    def _fan_future_done(self, percent, future):
        try:
            future.result()
            self.fan_command_finished.emit(True, "", percent)
        except Exception as exc:  # pylint: disable=broad-except
            self.fan_command_finished.emit(False, str(exc), percent)

    def _handle_fan_command_finished(self, ok, message, percent):
        self.fan_request_pending = False
        if self.apply_fan_button is not None:
            self.apply_fan_button.setEnabled(True)
        if self.stop_fan_button is not None:
            self.stop_fan_button.setEnabled(True)

        if ok:
            self.fan_slider_dirty = False
            if self.fan_hint_label is not None:
                self.fan_hint_label.setText(f"风扇已设置为 {percent}%")
            env_monitor.update_status("connected", f"fan set to {percent}%")
        else:
            if self.fan_hint_label is not None:
                self.fan_hint_label.setText(f"控制失败: {message}")

    def _stop_fan(self):
        self._sync_slider(0)
        self.fan_slider_dirty = True
        self._send_fan_pwm(0)

    def refresh_data(self):
        status_snapshot, _snapshot, fan_snapshot, latest_sensor = get_live_monitor_snapshot()
        if self.status_label is not None:
            self.status_label.setText(
                f"BLE状态: {status_snapshot.get('state', '--')} | {status_snapshot.get('message', '--')}"
            )

        for spec in ENVIRONMENT_THRESHOLD_SPECS:
            label = self.current_value_labels.get(spec["field"])
            if label is not None:
                value = latest_sensor.get(spec["field"]) if latest_sensor else None
                label.setText(format_metric_value(spec["field"], value))

        fan_controls_enabled = (
            status_snapshot.get("state") == "connected"
            and fan_snapshot.get("supported", False)
            and not self.fan_request_pending
        )
        if self.apply_fan_button is not None:
            self.apply_fan_button.setEnabled(fan_controls_enabled)
        if self.stop_fan_button is not None:
            self.stop_fan_button.setEnabled(fan_controls_enabled)

        if self.fan_hint_label is not None and not self.fan_request_pending:
            if not fan_snapshot.get("supported", False):
                self.fan_hint_label.setText(f"风扇控制不可用: {fan_snapshot.get('message', '--')}")
            elif status_snapshot.get("state") != "connected":
                self.fan_hint_label.setText("等待 BLE 连接后可控制风扇")

        fan_pwm = latest_sensor.get("fan_pwm_percent") if latest_sensor else None
        if (
            fan_pwm is not None
            and self.fan_slider is not None
            and not self.fan_slider_dirty
            and not self.fan_request_pending
        ):
            self._sync_slider(int(fan_pwm))

    def shutdown(self):
        self.refresh_timer.stop()


class HistoryPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.panel_active = False
        self.history_chart_panel = None
        self.chart_refresh_single_shot = QTimer(self)
        self.chart_refresh_single_shot.setSingleShot(True)
        self.chart_refresh_single_shot.timeout.connect(self._refresh_history_chart)
        self.status_label = None
        self.start_datetime = None
        self.end_datetime = None
        self.sensor_export_button = None
        self.recognition_export_button = None
        self.sensor_table = None
        self.recognition_table = None
        self.detail_label = None
        self.detail_image_label = None
        self.detail_text_view = None
        self.current_sensor_records = []
        self.current_recognition_records = []
        self.detail_pixmap = QPixmap()
        self._build_ui()
        self._init_time_range()
        self.refresh_tables()

    def set_active(self, active: bool):
        self.panel_active = active
        if active:
            self.schedule_chart_refresh()
        else:
            self.chart_refresh_single_shot.stop()

    def schedule_chart_refresh(self, delay_ms: int = 120):
        if not self.panel_active:
            return
        self.chart_refresh_single_shot.stop()
        self.chart_refresh_single_shot.start(delay_ms)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        header_row = QHBoxLayout()

        title = QLabel("历史记录查询")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']};")

        refresh_button = QPushButton("刷新记录")
        refresh_button.clicked.connect(self.refresh_tables)

        self.start_datetime = QDateTimeEdit()
        self.start_datetime.setCalendarPopup(True)
        self.start_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")

        self.end_datetime = QDateTimeEdit()
        self.end_datetime.setCalendarPopup(True)
        self.end_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")

        filter_button = QPushButton("按时间筛选")
        filter_button.clicked.connect(self.refresh_tables)

        self.sensor_export_button = QPushButton("导出传感器记录")
        self.sensor_export_button.clicked.connect(self.export_sensor_records)

        self.recognition_export_button = QPushButton("导出识别记录")
        self.recognition_export_button.clicked.connect(self.export_recognition_records)

        header_row.addWidget(title)
        header_row.addSpacing(12)
        header_row.addWidget(QLabel("开始时间"))
        header_row.addWidget(self.start_datetime)
        header_row.addWidget(QLabel("结束时间"))
        header_row.addWidget(self.end_datetime)
        header_row.addWidget(filter_button)
        header_row.addStretch(1)
        header_row.addWidget(self.sensor_export_button)
        header_row.addWidget(self.recognition_export_button)
        header_row.addWidget(refresh_button)
        layout.addLayout(header_row)

        self.status_label = QLabel("正在加载历史记录...")
        self.status_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 15px;")
        layout.addWidget(self.status_label)

        self.setStyleSheet(
            f"""
            QTableWidget {{
                background-color: {DASHBOARD_COLORS['bg']};
                alternate-background-color: {DASHBOARD_COLORS['panel']};
            }}
            """
        )

        self.history_chart_panel = SensorPanel(
            show_header=False,
            show_controls=True,
            show_cards=False,
            show_charts=True,
            live_refresh=False,
        )
        chart_group = QGroupBox("环境历史图表")
        chart_layout = QVBoxLayout(chart_group)
        chart_layout.addWidget(self.history_chart_panel)
        layout.addWidget(chart_group)

        sensor_group = QGroupBox("传感器历史记录")
        sensor_layout = QVBoxLayout(sensor_group)
        self.sensor_table = QTableWidget()
        self.sensor_table.setColumnCount(10)
        self.sensor_table.setHorizontalHeaderLabels(
            [
                "时间",
                "土壤",
                "温度",
                "湿度",
                "风速",
                "光照",
                "气压",
                "风扇PWM",
                "风速状态",
                "传感器在线状态",
            ]
        )
        self.sensor_table.verticalHeader().setVisible(False)
        self.sensor_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sensor_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.sensor_table.setAlternatingRowColors(True)
        sensor_layout.addWidget(self.sensor_table)

        recognition_group = QGroupBox("识别历史记录")
        recognition_layout = QVBoxLayout(recognition_group)
        self.recognition_table = QTableWidget()
        self.recognition_table.setColumnCount(6)
        self.recognition_table.setHorizontalHeaderLabels(
            [
                "时间",
                "用户",
                "结果",
                "植物类型",
                "病虫害",
                "识别间隔(秒)",
            ]
        )
        self.recognition_table.verticalHeader().setVisible(False)
        self.recognition_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.recognition_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.recognition_table.setAlternatingRowColors(True)
        self.recognition_table.itemSelectionChanged.connect(self.show_selected_recognition_detail)
        recognition_layout.addWidget(self.recognition_table)

        detail_group = QGroupBox("识别详情与图片")
        detail_layout = QHBoxLayout(detail_group)
        detail_layout.setSpacing(14)

        left_layout = QVBoxLayout()
        self.detail_label = QLabel("请选择一条识别记录查看详情")
        self.detail_label.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']}; font-size: 17px; font-weight: 700;")
        self.detail_text_view = QTextEdit()
        self.detail_text_view.setReadOnly(True)
        self.detail_text_view.setMinimumWidth(260)
        self.detail_text_view.setPlainText("识别详情将在这里显示")
        left_layout.addWidget(self.detail_label)
        left_layout.addWidget(self.detail_text_view, 1)

        right_layout = QVBoxLayout()
        image_title = QLabel("历史识别图片")
        image_title.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']}; font-size: 17px; font-weight: 700;")
        self.detail_image_label = QLabel("识别结果图片将在这里显示")
        self.detail_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_image_label.setMinimumSize(200, 150)
        self.detail_image_label.setStyleSheet(
            f"background-color: {DASHBOARD_COLORS['panel']}; color: {DASHBOARD_COLORS['muted_dark']}; "
            f"border-radius: 14px; border: 1px solid {DASHBOARD_COLORS['border']};"
        )
        right_layout.addWidget(image_title)
        right_layout.addWidget(self.detail_image_label, 1)

        detail_layout.addLayout(left_layout, 1)
        detail_layout.addLayout(right_layout, 3)

        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.addWidget(sensor_group)
        left_splitter.addWidget(recognition_group)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(chart_group)
        right_splitter.addWidget(detail_group)
        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(1, 2)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.addWidget(left_splitter)
        content_splitter.addWidget(right_splitter)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)
        layout.addWidget(content_splitter, 1)

    def _init_time_range(self):
        min_ts, max_ts = fetch_history_bounds()
        self.start_datetime.setDateTime(timestamp_to_qdatetime(min_ts))
        self.end_datetime.setDateTime(timestamp_to_qdatetime(max_ts))

    def _selected_range(self):
        start_ts = qdatetime_to_timestamp(self.start_datetime)
        end_ts = qdatetime_to_timestamp(self.end_datetime)
        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts
        return start_ts, end_ts

    def refresh_tables(self):
        try:
            start_ts, end_ts = self._selected_range()
            sensor_records = fetch_sensor_records(limit=None, start_ts=start_ts, end_ts=end_ts)
            recognition_records = fetch_recognition_records(limit=None, start_ts=start_ts, end_ts=end_ts)
        except Exception as exc:  # pylint: disable=broad-except
            self.status_label.setText(f"历史记录加载失败: {exc}")
            return

        self.current_sensor_records = sensor_records
        self.current_recognition_records = recognition_records
        self._fill_sensor_table(self.current_sensor_records)
        self._fill_recognition_table(self.current_recognition_records)
        self.status_label.setText(
            f"已加载 {len(sensor_records)} 条传感器记录，{len(recognition_records)} 条识别记录"
        )
        self._reset_detail_panel()
        self.schedule_chart_refresh(delay_ms=0)

    def _history_sensor_snapshot(self):
        records = list(reversed(self.current_sensor_records))
        if len(records) > HISTORY_CHART_MAX_POINTS:
            step = max(1, len(records) // HISTORY_CHART_MAX_POINTS)
            sampled = records[::step]
            if sampled[-1] is not records[-1]:
                sampled.append(records[-1])
            records = sampled[-HISTORY_CHART_MAX_POINTS:]

        snapshot = []
        for record in records:
            snapshot.append(
                {
                    "ts": record.get("ts", 0),
                    "soil_humidity": record.get("soil_humidity"),
                    "soil_state": record.get("soil_state"),
                    "air_temperature": record.get("air_temperature"),
                    "air_humidity": record.get("air_humidity"),
                    "wind_speed": record.get("wind_speed"),
                    "light_lux": record.get("light_lux"),
                    "pressure_hpa": record.get("pressure_hpa"),
                    "fan_pwm_percent": record.get("fan_pwm_percent"),
                    "wind": record.get("wind"),
                    "aht20": record.get("aht20"),
                    "bh1750": record.get("bh1750"),
                    "bmp280": record.get("bmp280"),
                }
            )
        return snapshot

    def _refresh_history_chart(self):
        if self.history_chart_panel is None:
            return
        snapshot = self._history_sensor_snapshot()
        self.history_chart_panel.set_snapshot(snapshot)

    def _fill_sensor_table(self, records):
        self.sensor_table.setRowCount(len(records))
        for row_index, record in enumerate(records):
            online_text = (
                f"AHT20 {record.get('aht20', '--')} | "
                f"BH1750 {record.get('bh1750', '--')} | "
                f"BMP280 {record.get('bmp280', '--')}"
            )
            values = [
                record.get("recorded_at", "--"),
                "--" if record.get("soil_humidity") is None else str(record.get("soil_humidity")),
                "--" if record.get("air_temperature") is None else f"{record.get('air_temperature'):.1f}",
                "--" if record.get("air_humidity") is None else f"{record.get('air_humidity'):.0f}",
                "--" if record.get("wind_speed") is None else f"{record.get('wind_speed'):.2f}",
                "--" if record.get("light_lux") is None else f"{record.get('light_lux'):.0f}",
                "--" if record.get("pressure_hpa") is None else f"{record.get('pressure_hpa'):.1f}",
                "--" if record.get("fan_pwm_percent") is None else f"{record.get('fan_pwm_percent')}%",
                record.get("wind", "--"),
                online_text,
            ]
            for col_index, value in enumerate(values):
                self.sensor_table.setItem(row_index, col_index, QTableWidgetItem(value))
        self.sensor_table.resizeColumnsToContents()

    def _fill_recognition_table(self, records):
        self.recognition_table.setRowCount(len(records))
        for row_index, record in enumerate(records):
            values = [
                record.get("recorded_at", "--"),
                record.get("username") or "--",
                record.get("summary", "--"),
                record.get("plant_type") or "--",
                format_label_for_display(record.get("pest_label")) if record.get("pest_label") else "--",
                "--"
                if record.get("interval_seconds") is None
                else f"{record.get('interval_seconds'):.1f}",
            ]
            for col_index, value in enumerate(values):
                self.recognition_table.setItem(row_index, col_index, QTableWidgetItem(value))
        self.recognition_table.resizeColumnsToContents()

    def _reset_detail_panel(self):
        self.detail_label.setText("请选择一条识别记录查看详情")
        self.detail_text_view.setPlainText("识别详情将在这里显示")
        self.detail_image_label.setPixmap(QPixmap())
        self.detail_image_label.setText("识别结果图片将在这里显示")
        self.detail_pixmap = QPixmap()

    def show_selected_recognition_detail(self):
        row = self.recognition_table.currentRow()
        if row < 0 or row >= len(self.current_recognition_records):
            self._reset_detail_panel()
            return

        record = self.current_recognition_records[row]
        detail_lines = [
            f"时间: {record.get('recorded_at', '--')}",
            f"用户: {record.get('username') or '--'}",
            f"结果: {record.get('summary', '--')}",
            f"植物类型: {record.get('plant_type') or '--'}",
            f"病虫害: {format_label_for_display(record.get('pest_label')) if record.get('pest_label') else '--'}",
            "识别间隔: --"
            if record.get("interval_seconds") is None
            else f"识别间隔: {record.get('interval_seconds'):.1f} 秒",
        ]

        try:
            detections = json.loads(record.get("detections_json") or "[]")
        except Exception:  # pylint: disable=broad-except
            detections = []
        try:
            top_candidates = json.loads(record.get("top_candidates_json") or "[]")
        except Exception:  # pylint: disable=broad-except
            top_candidates = []

        detail_lines.append("")
        detail_lines.append("Top 3 候选:")
        detail_lines.extend(format_top3_lines(top_candidates))
        detail_lines.append("")
        detail_lines.append("检测详情:")
        detail_lines.extend(format_detection_lines(detections))

        self.detail_label.setText(f"识别详情 #{record.get('id', '--')}")
        self.detail_text_view.setPlainText("\n".join(detail_lines))
        self._load_detail_image(record.get("image_path"))

    def _load_detail_image(self, image_path):
        if not image_path:
            self.detail_image_label.setPixmap(QPixmap())
            self.detail_image_label.setText("该记录没有保存识别图片")
            self.detail_pixmap = QPixmap()
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.detail_image_label.setPixmap(QPixmap())
            self.detail_image_label.setText(f"图片不存在或无法加载: {image_path}")
            self.detail_pixmap = QPixmap()
            return

        self.detail_pixmap = pixmap
        self._refresh_detail_image()

    def _refresh_detail_image(self):
        if self.detail_pixmap.isNull():
            return
        scaled = self.detail_pixmap.scaled(
            self.detail_image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.detail_image_label.setText("")
        self.detail_image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        self._refresh_detail_image()
        super().resizeEvent(event)

    def export_sensor_records(self):
        self._export_records(
            title="导出传感器历史记录",
            default_name=f"sensor_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            headers=["时间", "土壤", "温度", "湿度", "风速", "光照", "气压", "风扇PWM", "风速状态", "传感器在线状态"],
            rows=[
                [
                    record.get("recorded_at", "--"),
                    "--" if record.get("soil_humidity") is None else str(record.get("soil_humidity")),
                    "--" if record.get("air_temperature") is None else f"{record.get('air_temperature'):.1f}",
                    "--" if record.get("air_humidity") is None else f"{record.get('air_humidity'):.0f}",
                    "--" if record.get("wind_speed") is None else f"{record.get('wind_speed'):.2f}",
                    "--" if record.get("light_lux") is None else f"{record.get('light_lux'):.0f}",
                    "--" if record.get("pressure_hpa") is None else f"{record.get('pressure_hpa'):.1f}",
                    "--" if record.get("fan_pwm_percent") is None else f"{record.get('fan_pwm_percent')}%",
                    record.get("wind", "--"),
                    f"AHT20 {record.get('aht20', '--')} | BH1750 {record.get('bh1750', '--')} | BMP280 {record.get('bmp280', '--')}",
                ]
                for record in self.current_sensor_records
            ],
            sheet_name="传感器记录",
        )

    def export_recognition_records(self):
        self._export_records(
            title="导出识别历史记录",
            default_name=f"recognition_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            headers=["时间", "用户", "结果", "植物类型", "植物置信度", "病虫害", "病虫害置信度", "识别间隔(秒)", "图片路径"],
            rows=[
                [
                    record.get("recorded_at", "--"),
                    record.get("username") or "--",
                    record.get("summary", "--"),
                    record.get("plant_type") or "--",
                    "--" if record.get("plant_confidence") is None else f"{record.get('plant_confidence'):.2f}",
                    format_label_for_display(record.get("pest_label")) if record.get("pest_label") else "--",
                    "--" if record.get("pest_confidence") is None else f"{record.get('pest_confidence'):.2f}",
                    "--" if record.get("interval_seconds") is None else f"{record.get('interval_seconds'):.1f}",
                    record.get("image_path") or "--",
                ]
                for record in self.current_recognition_records
            ],
            sheet_name="识别记录",
        )

    def _export_records(self, title, default_name, headers, rows, sheet_name):
        if not rows:
            self.status_label.setText("当前筛选范围内没有可导出的记录")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            title,
            str(BASE_DIR / default_name),
            "CSV Files (*.csv);;Excel Files (*.xlsx)",
        )
        if not file_path:
            return

        try:
            wants_xlsx = file_path.lower().endswith(".xlsx") or "*.xlsx" in selected_filter
            if wants_xlsx:
                if not file_path.lower().endswith(".xlsx"):
                    file_path += ".xlsx"
                export_rows_to_xlsx(file_path, sheet_name, headers, rows)
            else:
                if not file_path.lower().endswith(".csv"):
                    file_path += ".csv"
                export_rows_to_csv(file_path, headers, rows)
            self.status_label.setText(f"导出成功: {file_path}")
        except Exception as exc:  # pylint: disable=broad-except
            self.status_label.setText(f"导出失败: {exc}")


class PlantSystemWindow(QMainWindow):
    def __init__(self, username: str):
        super().__init__()
        self.username = username
        self.threshold_settings = build_default_threshold_settings()
        self.tabs = None
        self.detection_panel = None
        self.control_panel = None
        self.history_panel = None
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setWindowTitle("智能植物管护系统")
        screen_geom = QApplication.primaryScreen().availableGeometry()
        target_w = min(screen_geom.width(), 1600)
        target_h = min(screen_geom.height(), 980)
        self.resize(target_w, target_h)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("智能植物管护系统")
        title.setFont(QFont("Microsoft YaHei", 22, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']};")
        layout.addWidget(title)

        subtitle = QLabel(f"主画面识别监控 + 环境阈值管控 + 历史记录追溯 | 当前用户: {self.username}")
        subtitle.setFont(QFont("Microsoft YaHei", 13))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']};")
        layout.addWidget(subtitle)

        tabs = QTabWidget()
        self.tabs = tabs
        self.detection_panel = VideoDetectionPanel(self.username, self.threshold_settings)
        self.control_panel = EnvironmentControlPanel(self.threshold_settings)
        self.history_panel = HistoryPanel()
        tabs.addTab(self.detection_panel, "主画面监控")
        tabs.addTab(self.control_panel, "环境阈值与管控控制")
        tabs.addTab(self.history_panel, "历史记录")
        tabs.currentChanged.connect(self._handle_tab_changed)
        layout.addWidget(tabs, 1)

        self.setStyleSheet(build_dashboard_stylesheet(include_tabs=True))
        QTimer.singleShot(0, lambda: self._handle_tab_changed(self.tabs.currentIndex()))

    def _handle_tab_changed(self, index):
        if self.detection_panel is not None:
            self.detection_panel.set_active(index == 0)
        if self.control_panel is not None:
            self.control_panel.set_active(index == 1)
        if self.history_panel is not None:
            self.history_panel.set_active(index == 2)

    def closeEvent(self, event):
        if self.detection_panel is not None:
            self.detection_panel.shutdown()
        if self.control_panel is not None:
            self.control_panel.shutdown()
        if self.history_panel is not None:
            self.history_panel.set_active(False)
        super().closeEvent(event)


def main():
    init_database()

    app = QApplication(sys.argv)
    app.setApplicationName("智能植物管护系统")
    app.setFont(QFont("Microsoft YaHei", 12))

    login_dialog = LoginDialog()
    if login_dialog.exec() != QDialog.DialogCode.Accepted:
        return

    start_ble_thread()

    window = PlantSystemWindow(login_dialog.username)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
