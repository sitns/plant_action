import asyncio
import json
import threading
import time
from collections import deque
from pathlib import Path

from data_store import record_sensor_data

from PyQt6.QtCharts import QChart, QChartView, QDateTimeAxis, QLineSeries, QValueAxis
from PyQt6.QtCore import QDateTime, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

try:
    from bleak import BleakClient, BleakScanner
except ImportError as exc:
    raise SystemExit("Missing dependency: bleak. Install with: pip install bleak") from exc


DEVICE_NAME = "ESP32-EnvSensor"
SERVICE_UUID = "12345678-1234-1234-1234-1234567890ab"
CHAR_UUID = "abcdefab-1234-5678-9abc-def012345678"
FAN_CHAR_UUID = "fedcba98-7654-3210-fedc-ba9876543210"
CURTAIN_CHAR_UUID = "a1b2c3d4-1234-5678-9abc-def012345678"
PUMP_CHAR_UUID = "b2c3d4e5-1234-5678-9abc-def012345678"
MAX_POINTS = 240
WRITE_PROPERTIES = ("write", "write-without-response")
BASE_DIR = Path(__file__).resolve().parent
COMBO_ARROW_ICON = (BASE_DIR / "combo_arrow_dark.svg").as_posix()

DASHBOARD_COLORS = {
    "bg": "#f2efe8",
    "panel": "#fffaf3",
    "card": "#ffffff",
    "ink": "#1f2a24",
    "muted": "#6f7a73",
    "muted_dark": "#47554d",
    "grid": "#d6d1c4",
    "border": "#ece6db",
    "border_strong": "#e8e1d5",
    "accent": "#2f6f5f",
    "accent_hover": "#275b4e",
    "secondary": "#b16b2d",
    "secondary_hover": "#945522",
    "header_a": "#ecf8f1",
    "header_b": "#fff8eb",
}

METRIC_CONFIG = {
    "soil": {
        "field": "soil_humidity",
        "checkbox": "土壤",
        "card_title": "土壤模拟值",
        "series_name": "土壤模拟值",
        "axis_title": "Soil ADC",
        "axis_format": "%.0f",
        "color": "#7b5c2e",
        "side": "left",
        "default_range": (0.0, 4095.0),
        "include_zero": True,
    },
    "humidity": {
        "field": "air_humidity",
        "checkbox": "湿度",
        "card_title": "空气湿度",
        "series_name": "空气湿度 (%)",
        "axis_title": "Humidity %",
        "axis_format": "%.0f",
        "color": "#4b78c7",
        "side": "left",
        "default_range": (0.0, 100.0),
        "fixed_range": (0.0, 100.0),
    },
    "temperature": {
        "field": "air_temperature",
        "checkbox": "温度",
        "card_title": "空气温度",
        "series_name": "空气温度 (C)",
        "axis_title": "Temperature C",
        "axis_format": "%.1f",
        "color": "#d14f45",
        "side": "right",
        "default_range": (0.0, 40.0),
    },
    "wind": {
        "field": "wind_speed",
        "checkbox": "风速",
        "card_title": "风速",
        "series_name": "风速 (m/s)",
        "axis_title": "Wind m/s",
        "axis_format": "%.1f",
        "color": "#7c4cc9",
        "side": "right",
        "default_range": (0.0, 15.0),
        "include_zero": True,
    },
    "light": {
        "field": "light_lux",
        "checkbox": "光照",
        "card_title": "光照强度",
        "series_name": "光照强度 (lx)",
        "axis_title": "Light lx",
        "axis_format": "%.0f",
        "color": "#8b6a16",
        "side": "left",
        "default_range": (0.0, 2000.0),
        "include_zero": True,
    },
    "pressure": {
        "field": "pressure_hpa",
        "checkbox": "气压",
        "card_title": "气压",
        "series_name": "气压 (hPa)",
        "axis_title": "Pressure hPa",
        "axis_format": "%.1f",
        "color": "#2f8f67",
        "side": "right",
        "default_range": (980.0, 1030.0),
    },
}

METRIC_ORDER = list(METRIC_CONFIG.keys())

history = deque(maxlen=MAX_POINTS)
history_lock = threading.Lock()
ble_status = {"state": "idle", "message": "not started", "last_update": 0.0}
fan_control_state = {
    "supported": False,
    "message": "waiting for BLE connection",
    "char_uuid": None,
    "response": True,
}
ble_loop = None
ble_client = None
desired_fan_percent = 0
desired_curtain = False
desired_pump = False
actuator_control_state = {
    "curtain_supported": False,
    "pump_supported": False,
    "curtain_message": "waiting for BLE connection",
    "pump_message": "waiting for BLE connection",
    "curtain_char_uuid": None,
    "pump_char_uuid": None,
}
_ble_thread = None
_ble_thread_lock = threading.Lock()


def build_dashboard_stylesheet(include_tabs: bool = False) -> str:
    tab_styles = ""
    if include_tabs:
        tab_styles = f"""
            QTabWidget::pane {{
                border: 1px solid {DASHBOARD_COLORS['border']};
                border-radius: 12px;
                background: {DASHBOARD_COLORS['panel']};
            }}
            QTabBar::tab {{
                background-color: {DASHBOARD_COLORS['header_a']};
                color: {DASHBOARD_COLORS['ink']};
                min-width: 140px;
                padding: 12px 22px;
                font-size: 15px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 6px;
            }}
            QTabBar::tab:selected {{
                background-color: {DASHBOARD_COLORS['panel']};
                font-weight: 700;
            }}
        """

    return f"""
        QMainWindow {{
            background-color: {DASHBOARD_COLORS['bg']};
        }}
        QWidget {{
            color: {DASHBOARD_COLORS['ink']};
            font-family: "Microsoft YaHei";
            font-size: 15px;
        }}
        QGroupBox {{
            font-size: 18px;
            font-weight: 600;
            color: {DASHBOARD_COLORS['ink']};
            border: 1px solid {DASHBOARD_COLORS['border']};
            border-radius: 12px;
            margin-top: 10px;
            padding-top: 12px;
            background-color: {DASHBOARD_COLORS['panel']};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }}
        QPushButton {{
            background-color: {DASHBOARD_COLORS['accent']};
            color: {DASHBOARD_COLORS['panel']};
            border: none;
            border-radius: 999px;
            padding: 10px 18px;
            font-size: 15px;
            font-weight: 700;
        }}
        QPushButton:hover {{
            background-color: {DASHBOARD_COLORS['accent_hover']};
        }}
        QPushButton[variant="secondary"] {{
            background-color: {DASHBOARD_COLORS['secondary']};
        }}
        QPushButton[variant="secondary"]:hover {{
            background-color: {DASHBOARD_COLORS['secondary_hover']};
        }}
        QPushButton:disabled {{
            background-color: #c6c0b4;
            color: #f8f3ea;
        }}
        QTextEdit {{
            background-color: {DASHBOARD_COLORS['card']};
            border: 1px solid {DASHBOARD_COLORS['border']};
            border-radius: 12px;
            color: {DASHBOARD_COLORS['ink']};
            font-size: 15px;
            selection-background-color: {DASHBOARD_COLORS['accent']};
        }}
        QLineEdit, QDoubleSpinBox, QSpinBox {{
            background-color: {DASHBOARD_COLORS['card']};
            border: 1px solid {DASHBOARD_COLORS['border']};
            border-radius: 12px;
            color: {DASHBOARD_COLORS['ink']};
            padding: 8px 12px;
            font-size: 15px;
            selection-background-color: {DASHBOARD_COLORS['accent']};
        }}
        QDateTimeEdit, QComboBox {{
            background-color: {DASHBOARD_COLORS['panel']};
            border: 1px solid {DASHBOARD_COLORS['border']};
            border-radius: 12px;
            color: {DASHBOARD_COLORS['ink']};
            padding: 8px 12px;
            font-size: 15px;
            selection-background-color: {DASHBOARD_COLORS['accent']};
        }}
        QDateTimeEdit:focus, QComboBox:focus, QDateTimeEdit:on, QComboBox:on {{
            border: 1px solid {DASHBOARD_COLORS['accent']};
            outline: 0;
        }}
        QComboBox, QDateTimeEdit {{
            padding-right: 32px;
        }}
        QComboBox::drop-down, QDateTimeEdit::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 28px;
            border: 0;
            background-color: {DASHBOARD_COLORS['panel']};
            border-top-right-radius: 12px;
            border-bottom-right-radius: 12px;
        }}
        QComboBox::down-arrow, QDateTimeEdit::down-arrow {{
            image: url({COMBO_ARROW_ICON});
            width: 12px;
            height: 8px;
        }}
        QComboBox QAbstractItemView, QCalendarWidget QAbstractItemView {{
            background-color: {DASHBOARD_COLORS['panel']};
            border: 1px solid {DASHBOARD_COLORS['border']};
            color: {DASHBOARD_COLORS['ink']};
            selection-background-color: {DASHBOARD_COLORS['header_a']};
            selection-color: {DASHBOARD_COLORS['ink']};
            outline: 0;
        }}
        QCalendarWidget QWidget {{
            alternate-background-color: {DASHBOARD_COLORS['panel']};
        }}
        QCalendarWidget QWidget#qt_calendar_navigationbar {{
            background-color: {DASHBOARD_COLORS['panel']};
        }}
        QCalendarWidget QToolButton {{
            color: {DASHBOARD_COLORS['ink']};
            background-color: {DASHBOARD_COLORS['panel']};
            border: 0;
            padding: 6px 10px;
        }}
        QCalendarWidget QMenu {{
            background-color: {DASHBOARD_COLORS['panel']};
            color: {DASHBOARD_COLORS['ink']};
        }}
        QCalendarWidget QSpinBox {{
            background-color: {DASHBOARD_COLORS['panel']};
            border: 1px solid {DASHBOARD_COLORS['border']};
            border-radius: 8px;
            color: {DASHBOARD_COLORS['ink']};
            padding: 4px 8px;
        }}
        QTableWidget {{
            background-color: {DASHBOARD_COLORS['bg']};
            alternate-background-color: {DASHBOARD_COLORS['panel']};
            border: 1px solid {DASHBOARD_COLORS['border']};
            border-radius: 12px;
            color: {DASHBOARD_COLORS['ink']};
            gridline-color: {DASHBOARD_COLORS['border']};
        }}
        QTableWidget::item {{
            padding: 6px;
        }}
        QTableWidget::item:selected {{
            background-color: {DASHBOARD_COLORS['header_a']};
            color: {DASHBOARD_COLORS['ink']};
        }}
        QHeaderView::section {{
            background-color: {DASHBOARD_COLORS['header_a']};
            color: {DASHBOARD_COLORS['ink']};
            border: 0;
            border-bottom: 1px solid {DASHBOARD_COLORS['border']};
            border-right: 1px solid {DASHBOARD_COLORS['border']};
            padding: 8px 10px;
            font-size: 14px;
            font-weight: 700;
        }}
        QTableCornerButton::section {{
            background-color: {DASHBOARD_COLORS['header_a']};
            border: 1px solid {DASHBOARD_COLORS['border']};
        }}
        QScrollArea {{
            border: 0;
            background: transparent;
        }}
        QScrollBar:vertical {{
            background-color: {DASHBOARD_COLORS['header_b']};
            width: 14px;
            margin: 2px 2px 2px 0;
            border-radius: 7px;
        }}
        QScrollBar::handle:vertical {{
            background-color: #c9c1b2;
            min-height: 42px;
            border-radius: 7px;
            border: 1px solid #bdb3a2;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: #aa9b84;
            border: 1px solid #96866d;
        }}
        QScrollBar::handle:vertical:pressed {{
            background-color: {DASHBOARD_COLORS['secondary']};
            border: 1px solid {DASHBOARD_COLORS['secondary_hover']};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
            border: none;
            background: transparent;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: transparent;
            border-radius: 7px;
        }}
        QScrollBar:horizontal {{
            background-color: {DASHBOARD_COLORS['header_b']};
            height: 14px;
            margin: 0 2px 2px 2px;
            border-radius: 7px;
        }}
        QScrollBar::handle:horizontal {{
            background-color: #c9c1b2;
            min-width: 42px;
            border-radius: 7px;
            border: 1px solid #bdb3a2;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: #aa9b84;
            border: 1px solid #96866d;
        }}
        QScrollBar::handle:horizontal:pressed {{
            background-color: {DASHBOARD_COLORS['secondary']};
            border: 1px solid {DASHBOARD_COLORS['secondary_hover']};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
            border: none;
            background: transparent;
        }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: transparent;
            border-radius: 7px;
        }}
        QCheckBox, QRadioButton {{
            spacing: 6px;
            color: {DASHBOARD_COLORS['ink']};
            font-size: 14px;
            font-weight: 600;
        }}
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 16px;
            height: 16px;
        }}
        QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {{
            background-color: {DASHBOARD_COLORS['card']};
            border: 1px solid #d9d1c4;
            border-radius: 8px;
        }}
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background-color: {DASHBOARD_COLORS['accent']};
            border: 1px solid {DASHBOARD_COLORS['accent']};
            border-radius: 8px;
        }}
        QSlider::groove:horizontal {{
            height: 6px;
            background: #ded7ca;
            border-radius: 3px;
        }}
        QSlider::sub-page:horizontal {{
            background: {DASHBOARD_COLORS['secondary']};
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            width: 18px;
            margin: -6px 0;
            border-radius: 9px;
            background: {DASHBOARD_COLORS['secondary']};
            border: 1px solid {DASHBOARD_COLORS['card']};
        }}
        {tab_styles}
    """


def update_status(state: str, message: str) -> None:
    with history_lock:
        ble_status["state"] = state
        ble_status["message"] = message
        ble_status["last_update"] = time.time()


def set_fan_control_state(supported: bool, message: str, char_uuid: str | None = None, response: bool = True) -> None:
    with history_lock:
        fan_control_state["supported"] = supported
        fan_control_state["message"] = message
        fan_control_state["char_uuid"] = char_uuid
        fan_control_state["response"] = response


def resolve_fan_control(client: BleakClient) -> tuple[str | None, bool, str]:
    preferred_uuids = [FAN_CHAR_UUID, CHAR_UUID]

    for service in client.services:
        if service.uuid.lower() != SERVICE_UUID.lower():
            continue

        for preferred_uuid in preferred_uuids:
            for characteristic in service.characteristics:
                if characteristic.uuid.lower() != preferred_uuid.lower():
                    continue

                if "write" in characteristic.properties:
                    return characteristic.uuid, True, f"using writable characteristic {characteristic.uuid}"
                if "write-without-response" in characteristic.properties:
                    return characteristic.uuid, False, f"using writable characteristic {characteristic.uuid}"

        for characteristic in service.characteristics:
            if "write" in characteristic.properties:
                return characteristic.uuid, True, f"using writable characteristic {characteristic.uuid}"
            if "write-without-response" in characteristic.properties:
                return characteristic.uuid, False, f"using writable characteristic {characteristic.uuid}"

        break

    return None, True, "device does not expose a writable fan-control characteristic"


def set_actuator_control_state(actuator: str, supported: bool, message: str, char_uuid: str | None = None) -> None:
    with history_lock:
        actuator_control_state[f"{actuator}_supported"] = supported
        actuator_control_state[f"{actuator}_message"] = message
        actuator_control_state[f"{actuator}_char_uuid"] = char_uuid


async def write_curtain(on: bool) -> None:
    global desired_curtain
    client = ble_client
    if client is None or not client.is_connected:
        raise RuntimeError("BLE device not connected")
    await client.write_gatt_char(CURTAIN_CHAR_UUID, b"1" if on else b"0", response=True)
    desired_curtain = on
    with history_lock:
        if history:
            history[-1]["curtain"] = on


async def write_pump(on: bool) -> None:
    global desired_pump
    client = ble_client
    if client is None or not client.is_connected:
        raise RuntimeError("BLE device not connected")
    await client.write_gatt_char(PUMP_CHAR_UUID, b"1" if on else b"0", response=True)
    desired_pump = on
    with history_lock:
        if history:
            history[-1]["pump"] = on


def _as_float(value):
    if value is None:
        return None
    return float(value)


def _as_int(value):
    if value is None:
        return None
    return int(float(value))


def handle_notification(_sender, data: bytearray) -> None:
    text = data.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)

        if isinstance(payload, list):
            curtain_on = False
            pump_on = False
            if len(payload) >= 9:
                soil_humidity = payload[0] if len(payload) > 0 else None
                air_temperature = payload[1] if len(payload) > 1 else None
                air_humidity = payload[2] if len(payload) > 2 else None
                wind_speed = payload[3] if len(payload) > 3 else None
                light_lux = payload[4] if len(payload) > 4 else None
                pressure_hpa = payload[5] if len(payload) > 5 else None
                fan_pwm_percent = payload[6] if len(payload) > 6 else 0
                curtain_on = bool(int(payload[7])) if len(payload) > 7 else False
                pump_on = bool(int(payload[8])) if len(payload) > 8 else False
            elif len(payload) >= 7:
                soil_humidity = payload[0] if len(payload) > 0 else None
                air_temperature = payload[1] if len(payload) > 1 else None
                air_humidity = payload[2] if len(payload) > 2 else None
                wind_speed = payload[3] if len(payload) > 3 else None
                light_lux = payload[4] if len(payload) > 4 else None
                pressure_hpa = payload[5] if len(payload) > 5 else None
                fan_pwm_percent = payload[6] if len(payload) > 6 else 0
            elif len(payload) >= 6:
                soil_humidity = payload[0] if len(payload) > 0 else None
                air_temperature = payload[1] if len(payload) > 1 else None
                air_humidity = payload[2] if len(payload) > 2 else None
                wind_speed = payload[3] if len(payload) > 3 else None
                light_lux = payload[4] if len(payload) > 4 else None
                pressure_hpa = payload[5] if len(payload) > 5 else None
                fan_pwm_percent = 0
            elif len(payload) >= 5:
                soil_humidity = payload[0] if len(payload) > 0 else None
                air_temperature = payload[1] if len(payload) > 1 else None
                air_humidity = payload[2] if len(payload) > 2 else None
                wind_speed = None
                light_lux = payload[3] if len(payload) > 3 else None
                pressure_hpa = payload[4] if len(payload) > 4 else None
                fan_pwm_percent = 0
            else:
                soil_humidity = None
                air_temperature = payload[0] if len(payload) > 0 else None
                air_humidity = payload[1] if len(payload) > 1 else None
                wind_speed = None
                light_lux = payload[2] if len(payload) > 2 else None
                pressure_hpa = payload[3] if len(payload) > 3 else None
                fan_pwm_percent = 0

            millis = 0
            soil_state = str(_as_int(soil_humidity)) if soil_humidity is not None else "offline"
            wind_state = "online" if wind_speed is not None else "offline"
            aht20_state = "online" if air_temperature is not None else "offline"
            bh1750_state = "online" if light_lux is not None else "offline"
            bmp280_state = "online" if pressure_hpa is not None else "offline"
        elif isinstance(payload, dict):
            soil_humidity = payload.get("soilHumidity")
            air_temperature = payload.get("airTemperature")
            air_humidity = payload.get("airHumidity")
            wind_speed = payload.get("windSpeed")
            light_lux = payload.get("lightLux")
            pressure_hpa = payload.get("pressureHpa")
            fan_pwm_percent = payload.get("fanPwmPercent", 0)
            curtain_on = bool(int(payload.get("curtain", 0)))
            pump_on = bool(int(payload.get("pump", 0)))
            millis = int(payload.get("millis", 0))
            soil_state = payload.get(
                "soilState",
                str(_as_int(soil_humidity)) if soil_humidity is not None else "offline",
            )
            wind_state = payload.get("wind", "online" if wind_speed is not None else "offline")
            aht20_state = payload.get(
                "aht20",
                "online" if air_temperature is not None else "offline",
            )
            bh1750_state = payload.get(
                "bh1750",
                "online" if light_lux is not None else "offline",
            )
            bmp280_state = payload.get(
                "bmp280",
                "online" if pressure_hpa is not None else "offline",
            )
        else:
            raise ValueError("unsupported payload type")

        point = {
            "ts": time.time(),
            "soil_humidity": _as_int(soil_humidity),
            "soil_state": soil_state,
            "air_temperature": _as_float(air_temperature),
            "air_humidity": _as_float(air_humidity),
            "wind_speed": _as_float(wind_speed),
            "light_lux": _as_float(light_lux),
            "pressure_hpa": _as_float(pressure_hpa),
            "fan_pwm_percent": _as_int(fan_pwm_percent) if fan_pwm_percent is not None else 0,
            "curtain": curtain_on,
            "pump": pump_on,
            "wind": wind_state,
            "aht20": aht20_state,
            "bh1750": bh1750_state,
            "bmp280": bmp280_state,
            "millis": millis,
        }
        with history_lock:
            history.append(point)
        try:
            record_sensor_data(point)
        except Exception as exc:
            update_status("warn", f"database write failed: {exc}")
    except (TypeError, ValueError, json.JSONDecodeError):
        update_status("warn", f"invalid payload: {text[:50]}")


async def write_fan_pwm(percent: int) -> None:
    global desired_fan_percent

    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")

    client = ble_client
    if client is None or not client.is_connected:
        raise RuntimeError("BLE device not connected")

    with history_lock:
        fan_char_uuid = fan_control_state["char_uuid"]
        fan_response = fan_control_state["response"]
        fan_message = fan_control_state["message"]

    if fan_char_uuid is None:
        raise RuntimeError(fan_message)

    await client.write_gatt_char(fan_char_uuid, str(percent).encode("utf-8"), response=fan_response)
    desired_fan_percent = percent

    with history_lock:
        if history:
            history[-1]["fan_pwm_percent"] = percent


async def ble_worker() -> None:
    global ble_client, desired_curtain, desired_pump

    while True:
        try:
            update_status("scan", f"searching {DEVICE_NAME}")
            device = await BleakScanner.find_device_by_filter(
                lambda d, adv: d.name == DEVICE_NAME
                or (
                    adv is not None
                    and SERVICE_UUID.lower() in [uuid.lower() for uuid in (adv.service_uuids or [])]
                ),
                timeout=10.0,
            )
            if device is None:
                update_status("scan", "device not found, retrying")
                await asyncio.sleep(2.0)
                continue

            update_status("connect", f"connecting {device.address}")
            async with BleakClient(device) as client:
                ble_client = client
                await client.start_notify(CHAR_UUID, handle_notification)

                fan_char_uuid, fan_response, fan_message = resolve_fan_control(client)
                if fan_char_uuid is None:
                    set_fan_control_state(False, fan_message)
                    fan_ready = False
                else:
                    set_fan_control_state(True, fan_message, fan_char_uuid, fan_response)

                set_actuator_control_state("curtain", True, "ok", CURTAIN_CHAR_UUID)
                set_actuator_control_state("pump", True, "ok", PUMP_CHAR_UUID)

                status_parts = ["receiving notifications"]
                try:
                    await write_fan_pwm(desired_fan_percent) if fan_char_uuid else None
                except Exception as exc:
                    set_fan_control_state(False, f"fan init failed: {exc}")
                    status_parts.append(f"fan init failed: {exc}")

                if desired_curtain:
                    try:
                        await write_curtain(desired_curtain)
                    except Exception as exc:
                        status_parts.append(f"curtain init: {exc}")

                if desired_pump:
                    try:
                        await write_pump(desired_pump)
                    except Exception as exc:
                        status_parts.append(f"pump init: {exc}")

                update_status("connected", " | ".join(status_parts))

                while client.is_connected:
                    await asyncio.sleep(1.0)

            ble_client = None
            set_fan_control_state(False, "waiting for BLE connection")
            set_actuator_control_state("curtain", False, "waiting for BLE connection")
            set_actuator_control_state("pump", False, "waiting for BLE connection")
            update_status("scan", "disconnected, retrying")
        except Exception as exc:  # pylint: disable=broad-except
            ble_client = None
            set_fan_control_state(False, str(exc))
            set_actuator_control_state("curtain", False, str(exc))
            set_actuator_control_state("pump", False, str(exc))
            update_status("error", str(exc))
            await asyncio.sleep(2.0)


def run_ble_loop() -> None:
    global ble_loop

    ble_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ble_loop)
    ble_loop.run_until_complete(ble_worker())


def start_ble_thread() -> threading.Thread:
    global _ble_thread

    with _ble_thread_lock:
        if _ble_thread is not None and _ble_thread.is_alive():
            return _ble_thread

        _ble_thread = threading.Thread(target=run_ble_loop, daemon=True, name="ble-monitor")
        _ble_thread.start()
        return _ble_thread


def _format_packet_time(ts: float | None) -> str:
    if not ts:
        return "--"
    return QDateTime.fromSecsSinceEpoch(int(ts)).toString("HH:mm:ss")


def _format_sensor_status(latest: dict) -> str:
    soil_value = latest.get("soil_humidity")
    soil_text = str(soil_value) if soil_value is not None else "--"
    fan_value = latest.get("fan_pwm_percent")
    fan_text = str(fan_value) if fan_value is not None else "--"
    return (
        f"Soil ADC {soil_text} | Fan {fan_text}% | Wind {latest.get('wind', '--')} | "
        f"AHT20 {latest.get('aht20', '--')} | BH1750 {latest.get('bh1750', '--')} | "
        f"BMP280 {latest.get('bmp280', '--')}"
    )


def _value_range(values, default_range, include_zero=False, fixed_range=None):
    if fixed_range is not None:
        return fixed_range

    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return default_range

    lower = min(clean_values)
    upper = max(clean_values)

    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)

    if lower == upper:
        padding = max(abs(lower) * 0.1, 1.0)
    else:
        padding = max((upper - lower) * 0.12, 1.0)

    return lower - padding, upper + padding


def get_live_monitor_snapshot():
    with history_lock:
        status_snapshot = dict(ble_status)
        snapshot = list(history)
        fan_snapshot = dict(fan_control_state)
    latest = snapshot[-1] if snapshot else None
    return status_snapshot, snapshot, fan_snapshot, latest


class SensorPanel(QWidget):
    fan_command_finished = pyqtSignal(bool, str, int)

    def __init__(
        self,
        show_header: bool = True,
        show_controls: bool = True,
        show_cards: bool = True,
        show_charts: bool = True,
        live_refresh: bool = True,
    ):
        super().__init__()
        self.show_header = show_header
        self.show_controls = show_controls
        self.show_cards = show_cards
        self.show_charts = show_charts
        self.live_refresh = live_refresh
        self.metric_checkboxes = {}
        self.metric_cards = {}
        self.combined_series = {}
        self.single_axes = {}
        self.split_cards = {}
        self._last_snapshot = []
        self.external_snapshot = None
        self.fan_request_pending = False
        self.fan_slider_dirty = False

        self.header_frame = None
        self.controls_frame = None
        self.chart_section_frame = None
        self.status_label = None
        self.air_temperature_value = None
        self.air_humidity_value = None
        self.soil_humidity_value = None
        self.wind_speed_value = None
        self.light_lux_value = None
        self.pressure_hpa_value = None
        self.fan_pwm_value = None
        self.sensor_state_value = None
        self.packet_time_value = None
        self.fan_slider = None
        self.fan_slider_value = None
        self.fan_hint_label = None
        self.apply_fan_button = None
        self.stop_fan_button = None
        self.single_chart_frame = None
        self.split_chart_frame = None
        self.single_chart = None
        self.single_axis_x = None
        self.single_chart_radio = None
        self.split_chart_radio = None
        self.climate_humidity_axis = None
        self.climate_temperature_axis = None
        self.timer = None

        self._build_ui()
        self.fan_command_finished.connect(self._handle_fan_command_finished)
        if self.live_refresh:
            self.start_refresh()
        else:
            status_snapshot, snapshot, fan_snapshot, _latest = get_live_monitor_snapshot()
            self._apply_runtime_snapshot(status_snapshot, snapshot, fan_snapshot)

    def _build_ui(self):
        self.setObjectName("sensorPanelRoot")
        self.setStyleSheet(
            f"""
            QWidget#sensorPanelRoot,
            QWidget#sensorScrollViewport,
            QWidget#sensorScrollContent {{
                background-color: {DASHBOARD_COLORS['bg']};
            }}
            QScrollArea#sensorScrollArea {{
                background-color: {DASHBOARD_COLORS['bg']};
                border: 0;
            }}
            """
        )

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setObjectName("sensorScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.viewport().setObjectName("sensorScrollViewport")

        content = QWidget()
        content.setObjectName("sensorScrollContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(16)

        if self.show_header:
            self.header_frame = self._create_header()
            content_layout.addWidget(self.header_frame)

        if self.show_charts and self.show_controls:
            self.controls_frame = self._create_controls()
            content_layout.addWidget(self.controls_frame)

        if self.show_cards:
            content_layout.addLayout(self._create_cards_grid())

        if self.show_charts:
            self.chart_section_frame = self._create_chart_section()
            content_layout.addWidget(self.chart_section_frame)

        content_layout.addStretch(1)

        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

    def _create_header(self):
        frame = QFrame()
        frame.setObjectName("sensorHeader")
        frame.setStyleSheet(
            f"""
            QFrame#sensorHeader {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {DASHBOARD_COLORS['header_a']}, stop:1 {DASHBOARD_COLORS['header_b']});
                border: 1px solid {DASHBOARD_COLORS['border_strong']};
                border-radius: 16px;
            }}
            """
        )

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(6)

        title = QLabel("环境实时监控面板")
        title.setFont(QFont("Microsoft YaHei", 22, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']};")
        layout.addWidget(title)

        self.status_label = QLabel("BLE状态: 等待连接...")
        self.status_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted_dark']}; font-size: 16px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        return frame

    def _create_controls(self):
        frame = QFrame()
        frame.setObjectName("sensorControls")
        frame.setStyleSheet(
            f"""
            QFrame#sensorControls {{
                background-color: {DASHBOARD_COLORS['panel']};
                border: 1px solid {DASHBOARD_COLORS['border']};
                border-radius: 14px;
            }}
            QLabel {{
                color: {DASHBOARD_COLORS['muted']};
                font-size: 14px;
                font-weight: 700;
            }}
            """
        )

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(18)

        chart_group = QHBoxLayout()
        chart_group.setSpacing(12)
        chart_label = QLabel("图表布局")
        chart_group.addWidget(chart_label)

        self.single_chart_radio = QRadioButton("单图表")
        self.split_chart_radio = QRadioButton("分图表")
        self.single_chart_radio.setChecked(True)

        mode_group = QButtonGroup(self)
        mode_group.addButton(self.single_chart_radio)
        mode_group.addButton(self.split_chart_radio)
        self.single_chart_radio.toggled.connect(self._apply_chart_mode)
        self.split_chart_radio.toggled.connect(self._apply_chart_mode)

        chart_group.addWidget(self.single_chart_radio)
        chart_group.addWidget(self.split_chart_radio)
        chart_group.addStretch(1)
        layout.addLayout(chart_group, 1)

        metric_group = QHBoxLayout()
        metric_group.setSpacing(10)
        metric_label = QLabel("显示指标")
        metric_group.addWidget(metric_label)

        for key in METRIC_ORDER:
            checkbox = QCheckBox(METRIC_CONFIG[key]["checkbox"])
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._handle_metric_toggle)
            self.metric_checkboxes[key] = checkbox
            metric_group.addWidget(checkbox)

        metric_group.addStretch(1)
        layout.addLayout(metric_group, 2)
        return frame

    def _create_cards_grid(self):
        layout = QGridLayout()
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)

        temperature_card, self.air_temperature_value = self._create_value_card("空气温度", "-- C", METRIC_CONFIG["temperature"]["color"])
        humidity_card, self.air_humidity_value = self._create_value_card("空气湿度", "-- %", METRIC_CONFIG["humidity"]["color"])
        soil_card, self.soil_humidity_value = self._create_value_card("土壤模拟值", "--", METRIC_CONFIG["soil"]["color"])
        wind_card, self.wind_speed_value = self._create_value_card("风速", "-- m/s", METRIC_CONFIG["wind"]["color"])
        light_card, self.light_lux_value = self._create_value_card("光照强度", "-- lx", METRIC_CONFIG["light"]["color"])
        pressure_card, self.pressure_hpa_value = self._create_value_card("气压", "-- hPa", METRIC_CONFIG["pressure"]["color"])
        fan_pwm_card, self.fan_pwm_value = self._create_value_card("风扇 PWM", "0 %", DASHBOARD_COLORS["secondary"])
        fan_control_card = self._create_fan_control_card()
        sensor_state_card, self.sensor_state_value = self._create_value_card("传感器状态", "--", DASHBOARD_COLORS["ink"], font_size=20)
        packet_time_card, self.packet_time_value = self._create_value_card("最后上报时间", "--", DASHBOARD_COLORS["ink"], font_size=22)

        self.metric_cards = {
            "temperature": temperature_card,
            "humidity": humidity_card,
            "soil": soil_card,
            "wind": wind_card,
            "light": light_card,
            "pressure": pressure_card,
        }

        layout.addWidget(temperature_card, 0, 0)
        layout.addWidget(humidity_card, 0, 1)
        layout.addWidget(soil_card, 0, 2)
        layout.addWidget(wind_card, 1, 0)
        layout.addWidget(light_card, 1, 1)
        layout.addWidget(pressure_card, 1, 2)
        layout.addWidget(fan_pwm_card, 2, 0)
        layout.addWidget(fan_control_card, 2, 1, 1, 2)
        layout.addWidget(sensor_state_card, 3, 0, 1, 2)
        layout.addWidget(packet_time_card, 3, 2)

        for column in range(3):
            layout.setColumnStretch(column, 1)
        return layout

    def _create_card_frame(self):
        frame = QFrame()
        frame.setObjectName("dashboardCard")
        frame.setStyleSheet(
            f"""
            QFrame#dashboardCard {{
                background-color: {DASHBOARD_COLORS['panel']};
                border: 1px solid {DASHBOARD_COLORS['border']};
                border-radius: 12px;
            }}
            """
        )
        return frame

    def _create_value_card(self, title, value, color, font_size=34):
        frame = self._create_card_frame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted']}; font-size: 14px;")
        layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setWordWrap(True)
        value_label.setFont(QFont("Microsoft YaHei", font_size, QFont.Weight.Bold))
        value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(value_label)
        layout.addStretch(1)
        return frame, value_label

    def _create_fan_control_card(self):
        frame = self._create_card_frame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title_label = QLabel("风扇控制")
        title_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted']}; font-size: 14px;")
        layout.addWidget(title_label)

        info_row = QHBoxLayout()
        info_row.setSpacing(12)

        self.fan_slider_value = QLabel("0 %")
        self.fan_slider_value.setFont(QFont("Microsoft YaHei", 24, QFont.Weight.Bold))
        self.fan_slider_value.setStyleSheet(f"color: {DASHBOARD_COLORS['ink']};")
        info_row.addWidget(self.fan_slider_value)

        self.fan_hint_label = QLabel("通过 BLE 设置风扇转速")
        self.fan_hint_label.setWordWrap(True)
        self.fan_hint_label.setStyleSheet(f"color: {DASHBOARD_COLORS['muted']}; font-size: 14px;")
        info_row.addWidget(self.fan_hint_label, 1)
        layout.addLayout(info_row)

        self.fan_slider = QSlider(Qt.Orientation.Horizontal)
        self.fan_slider.setRange(0, 100)
        self.fan_slider.setSingleStep(1)
        self.fan_slider.setValue(0)
        self.fan_slider.valueChanged.connect(self._handle_slider_change)
        layout.addWidget(self.fan_slider)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.apply_fan_button = QPushButton("应用")
        self.stop_fan_button = QPushButton("停止风扇")
        self.stop_fan_button.setProperty("variant", "secondary")

        self.apply_fan_button.clicked.connect(lambda: self._send_fan_pwm(self.fan_slider.value()))
        self.stop_fan_button.clicked.connect(self._stop_fan)

        button_row.addWidget(self.apply_fan_button)
        button_row.addWidget(self.stop_fan_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        return frame

    def _create_chart_section(self):
        frame = QFrame()
        frame.setObjectName("chartSection")
        frame.setStyleSheet(
            f"""
            QFrame#chartSection {{
                background-color: {DASHBOARD_COLORS['panel']};
                border: 1px solid {DASHBOARD_COLORS['border']};
                border-radius: 14px;
            }}
            """
        )

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        self.single_chart_frame = self._create_chart_card("环境多指标总览")
        self.single_chart, self.single_axis_x = self._build_combined_chart(self.single_chart_frame[1])
        layout.addWidget(self.single_chart_frame[0])

        self.split_chart_frame = QFrame()
        split_layout = QVBoxLayout(self.split_chart_frame)
        split_layout.setContentsMargins(0, 0, 0, 0)
        split_layout.setSpacing(14)

        soil_card = self._create_chart_card("土壤模拟值")
        soil_chart = self._build_single_metric_chart(soil_card[1], "soil")
        split_layout.addWidget(soil_card[0])

        climate_card = self._create_chart_card("温度与湿度")
        climate_chart = self._build_climate_chart(climate_card[1])
        split_layout.addWidget(climate_card[0])

        light_card = self._create_chart_card("光照强度")
        light_chart = self._build_single_metric_chart(light_card[1], "light")
        split_layout.addWidget(light_card[0])

        wind_card = self._create_chart_card("风速")
        wind_chart = self._build_single_metric_chart(wind_card[1], "wind")
        split_layout.addWidget(wind_card[0])

        pressure_card = self._create_chart_card("气压")
        pressure_chart = self._build_single_metric_chart(pressure_card[1], "pressure")
        split_layout.addWidget(pressure_card[0])

        self.split_cards = {
            "soil": {"card": soil_card[0], **soil_chart},
            "climate": {"card": climate_card[0], **climate_chart},
            "light": {"card": light_card[0], **light_chart},
            "wind": {"card": wind_card[0], **wind_chart},
            "pressure": {"card": pressure_card[0], **pressure_chart},
        }

        layout.addWidget(self.split_chart_frame)
        self._apply_chart_mode()
        return frame

    def _create_chart_card(self, title):
        frame = self._create_card_frame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"background-color: {DASHBOARD_COLORS['header_a']}; color: {DASHBOARD_COLORS['muted_dark']}; "
            f"font-size: 14px; font-weight: 700; padding: 10px 12px; border-bottom: 1px solid {DASHBOARD_COLORS['border']};"
        )
        layout.addWidget(title_label)

        chart_host = QWidget()
        chart_host.setStyleSheet(f"background-color: {DASHBOARD_COLORS['panel']};")
        chart_layout = QVBoxLayout(chart_host)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(chart_host)
        return frame, chart_layout

    def _create_chart_base(self, chart_layout, legend_visible, min_height):
        chart = QChart()
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setBackgroundBrush(QColor(DASHBOARD_COLORS["panel"]))
        chart.setPlotAreaBackgroundVisible(True)
        chart.setPlotAreaBackgroundBrush(QColor(DASHBOARD_COLORS["panel"]))
        chart.setTitleBrush(QColor(DASHBOARD_COLORS["ink"]))
        legend = chart.legend()
        if legend is not None:
            legend.setVisible(legend_visible)
            legend.setLabelColor(QColor(DASHBOARD_COLORS["muted_dark"]))

        axis_x = QDateTimeAxis()
        axis_x.setFormat("HH:mm:ss")
        axis_x.setTickCount(6)
        axis_x.setTitleText("时间")
        self._style_axis(axis_x)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        chart_view.setMinimumHeight(min_height)
        chart_view.setStyleSheet("background: transparent; border: 0;")
        chart_layout.addWidget(chart_view)
        return chart, axis_x

    def _create_series(self, metric_key):
        series = QLineSeries()
        series.setName(METRIC_CONFIG[metric_key]["series_name"])
        series.setColor(QColor(METRIC_CONFIG[metric_key]["color"]))
        return series

    def _create_value_axis(self, metric_key, title_override=None):
        axis = QValueAxis()
        axis.setTickCount(6)
        axis.setLabelFormat(METRIC_CONFIG[metric_key]["axis_format"])
        axis.setTitleText(title_override or METRIC_CONFIG[metric_key]["axis_title"])
        self._style_axis(axis)
        return axis

    def _style_axis(self, axis):
        axis.setTitleBrush(QColor(DASHBOARD_COLORS["ink"]))
        axis.setLabelsColor(QColor(DASHBOARD_COLORS["ink"]))
        axis.setGridLineColor(QColor(DASHBOARD_COLORS["grid"]))
        axis.setLinePenColor(QColor(DASHBOARD_COLORS["grid"]))

    def _build_combined_chart(self, chart_layout):
        chart, axis_x = self._create_chart_base(chart_layout, legend_visible=True, min_height=340)

        for key in METRIC_ORDER:
            series = self._create_series(key)
            axis = self._create_value_axis(key)
            axis.setVisible(False)
            chart.addSeries(series)
            chart.addAxis(
                axis,
                Qt.AlignmentFlag.AlignLeft
                if METRIC_CONFIG[key]["side"] == "left"
                else Qt.AlignmentFlag.AlignRight,
            )
            series.attachAxis(axis_x)
            series.attachAxis(axis)
            self.combined_series[key] = series
            self.single_axes[key] = axis

        return chart, axis_x

    def _build_single_metric_chart(self, chart_layout, metric_key):
        chart, axis_x = self._create_chart_base(chart_layout, legend_visible=False, min_height=240)
        axis_y = self._create_value_axis(metric_key)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)

        series = self._create_series(metric_key)
        chart.addSeries(series)
        series.attachAxis(axis_x)
        series.attachAxis(axis_y)
        return {"chart": chart, "axis_x": axis_x, "axis_y": axis_y, "series": {metric_key: series}}

    def _build_climate_chart(self, chart_layout):
        chart, axis_x = self._create_chart_base(chart_layout, legend_visible=True, min_height=240)
        humidity_axis = self._create_value_axis("humidity")
        temperature_axis = self._create_value_axis("temperature")
        chart.addAxis(humidity_axis, Qt.AlignmentFlag.AlignLeft)
        chart.addAxis(temperature_axis, Qt.AlignmentFlag.AlignRight)

        humidity_series = self._create_series("humidity")
        temperature_series = self._create_series("temperature")
        chart.addSeries(humidity_series)
        chart.addSeries(temperature_series)
        humidity_series.attachAxis(axis_x)
        humidity_series.attachAxis(humidity_axis)
        temperature_series.attachAxis(axis_x)
        temperature_series.attachAxis(temperature_axis)

        self.climate_humidity_axis = humidity_axis
        self.climate_temperature_axis = temperature_axis
        return {
            "chart": chart,
            "axis_x": axis_x,
            "axis_y": humidity_axis,
            "axis_y_right": temperature_axis,
            "series": {"humidity": humidity_series, "temperature": temperature_series},
        }

    def _set_series_visibility(self, chart, series, visible):
        series.setVisible(visible)
        legend = chart.legend()
        if legend is None:
            return
        for marker in legend.markers(series):
            marker.setVisible(visible)

    def _handle_metric_toggle(self):
        self._apply_metric_visibility()
        self._refresh_charts(self._last_snapshot)

    def _apply_chart_mode(self):
        if (
            self.single_chart_frame is None
            or self.split_chart_frame is None
            or self.single_chart_radio is None
        ):
            return
        use_split = self.split_chart_radio.isChecked()
        self.single_chart_frame[0].setVisible(not use_split)
        self.split_chart_frame.setVisible(use_split)
        self._apply_metric_visibility()

    def _apply_metric_visibility(self):
        if self.single_chart is None or not self.split_cards:
            return
        active_metrics = self.active_metrics()

        for key, series in self.combined_series.items():
            self._set_series_visibility(self.single_chart, series, key in active_metrics)

        climate_series = self.split_cards["climate"]["series"]
        self._set_series_visibility(
            self.split_cards["climate"]["chart"],
            climate_series["humidity"],
            "humidity" in active_metrics,
        )
        self._set_series_visibility(
            self.split_cards["climate"]["chart"],
            climate_series["temperature"],
            "temperature" in active_metrics,
        )

        self.split_cards["soil"]["card"].setVisible("soil" in active_metrics)
        self.split_cards["light"]["card"].setVisible("light" in active_metrics)
        self.split_cards["wind"]["card"].setVisible("wind" in active_metrics)
        self.split_cards["pressure"]["card"].setVisible("pressure" in active_metrics)
        self.split_cards["climate"]["card"].setVisible(
            "humidity" in active_metrics or "temperature" in active_metrics
        )

        self.climate_humidity_axis.setVisible("humidity" in active_metrics)
        self.climate_temperature_axis.setVisible("temperature" in active_metrics)

    def _handle_slider_change(self, value):
        self.fan_slider_dirty = True
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
        if ble_loop is None:
            self.fan_hint_label.setText("BLE 循环尚未就绪")
            return

        with history_lock:
            fan_supported = fan_control_state["supported"]
            fan_message = fan_control_state["message"]

        if not fan_supported:
            self.fan_hint_label.setText(f"风扇控制不可用: {fan_message}")
            return

        self.fan_request_pending = True
        self.fan_hint_label.setText(f"正在发送 {percent}%...")
        self.apply_fan_button.setEnabled(False)
        self.stop_fan_button.setEnabled(False)

        future = asyncio.run_coroutine_threadsafe(write_fan_pwm(percent), ble_loop)
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
            update_status("connected", f"fan set to {percent}%")
        else:
            if self.fan_hint_label is not None:
                self.fan_hint_label.setText(f"控制失败: {message}")

    def _stop_fan(self):
        self._sync_slider(0)
        self.fan_slider_dirty = True
        self._send_fan_pwm(0)

    def start_refresh(self):
        if self.timer is not None:
            self.timer.stop()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(1000)
        self.refresh_data()

    def active_metrics(self):
        if not self.metric_checkboxes:
            return set(METRIC_ORDER)
        return {key for key, checkbox in self.metric_checkboxes.items() if checkbox.isChecked()}

    def refresh_data(self):
        status_snapshot, live_snapshot, fan_snapshot, _latest = get_live_monitor_snapshot()
        snapshot = list(self.external_snapshot) if self.external_snapshot is not None else live_snapshot
        self._apply_runtime_snapshot(status_snapshot, snapshot, fan_snapshot)

    def set_snapshot(self, snapshot, status_snapshot=None, fan_snapshot=None):
        self.external_snapshot = list(snapshot)
        live_status_snapshot, _live_snapshot, live_fan_snapshot, _latest = get_live_monitor_snapshot()
        self._apply_runtime_snapshot(
            status_snapshot or live_status_snapshot,
            self.external_snapshot,
            fan_snapshot or live_fan_snapshot,
        )

    def _apply_runtime_snapshot(self, status_snapshot, snapshot, fan_snapshot):
        self._last_snapshot = snapshot

        if self.status_label is not None:
            self.status_label.setText(
                f"BLE状态: {status_snapshot.get('state', '--')} | {status_snapshot.get('message', '--')}"
            )

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

        if snapshot:
            latest = snapshot[-1]
            if self.air_temperature_value is not None:
                self.air_temperature_value.setText(
                    "-- C" if latest.get("air_temperature") is None else f"{latest['air_temperature']:.1f} C"
                )
            if self.air_humidity_value is not None:
                self.air_humidity_value.setText(
                    "-- %" if latest.get("air_humidity") is None else f"{latest['air_humidity']:.0f} %"
                )
            if self.soil_humidity_value is not None:
                self.soil_humidity_value.setText(
                    "--" if latest.get("soil_humidity") is None else f"{latest['soil_humidity']}"
                )
            if self.wind_speed_value is not None:
                self.wind_speed_value.setText(
                    "-- m/s" if latest.get("wind_speed") is None else f"{latest['wind_speed']:.2f} m/s"
                )
            if self.light_lux_value is not None:
                self.light_lux_value.setText(
                    "-- lx" if latest.get("light_lux") is None else f"{latest['light_lux']:.0f} lx"
                )
            if self.pressure_hpa_value is not None:
                self.pressure_hpa_value.setText(
                    "-- hPa" if latest.get("pressure_hpa") is None else f"{latest['pressure_hpa']:.1f} hPa"
                )

            fan_pwm = latest.get("fan_pwm_percent")
            if self.fan_pwm_value is not None:
                self.fan_pwm_value.setText("-- %" if fan_pwm is None else f"{fan_pwm} %")
            if (
                fan_pwm is not None
                and self.fan_slider is not None
                and not self.fan_slider_dirty
                and not self.fan_request_pending
            ):
                self._sync_slider(fan_pwm)

            if self.sensor_state_value is not None:
                self.sensor_state_value.setText(_format_sensor_status(latest))
            if self.packet_time_value is not None:
                self.packet_time_value.setText(_format_packet_time(latest.get("ts")))
        else:
            if self.air_temperature_value is not None:
                self.air_temperature_value.setText("-- C")
            if self.air_humidity_value is not None:
                self.air_humidity_value.setText("-- %")
            if self.soil_humidity_value is not None:
                self.soil_humidity_value.setText("--")
            if self.wind_speed_value is not None:
                self.wind_speed_value.setText("-- m/s")
            if self.light_lux_value is not None:
                self.light_lux_value.setText("-- lx")
            if self.pressure_hpa_value is not None:
                self.pressure_hpa_value.setText("-- hPa")
            if self.fan_pwm_value is not None:
                self.fan_pwm_value.setText("-- %")
            if self.sensor_state_value is not None:
                self.sensor_state_value.setText("--")
            if self.packet_time_value is not None:
                self.packet_time_value.setText("--")

        if self.show_charts and self.single_chart is not None:
            self._apply_metric_visibility()
            self._refresh_charts(snapshot)

    def _refresh_charts(self, snapshot):
        self._update_combined_chart(snapshot)
        self._update_split_charts(snapshot)

    def _update_combined_chart(self, snapshot):
        active_metrics = self.active_metrics()

        for key in METRIC_ORDER:
            series = self.combined_series[key]
            axis = self.single_axes[key]
            field = METRIC_CONFIG[key]["field"]
            values = []
            series.clear()
            for point in snapshot:
                value = point.get(field)
                if value is None:
                    continue
                values.append(value)
                series.append(point.get("ts", 0) * 1000, float(value))

            lower, upper = _value_range(
                values,
                METRIC_CONFIG[key]["default_range"],
                include_zero=METRIC_CONFIG[key].get("include_zero", False),
                fixed_range=METRIC_CONFIG[key].get("fixed_range"),
            )
            axis.setRange(lower, upper)
            axis.setVisible(key in active_metrics)

        self._set_time_axis(self.single_axis_x, snapshot)

    def _update_split_charts(self, snapshot):
        self._update_single_metric_chart(self.split_cards["soil"], snapshot, "soil")
        self._update_single_metric_chart(self.split_cards["light"], snapshot, "light")
        self._update_single_metric_chart(self.split_cards["wind"], snapshot, "wind")
        self._update_single_metric_chart(self.split_cards["pressure"], snapshot, "pressure")

        climate_card = self.split_cards["climate"]
        humidity_values = []
        temperature_values = []
        humidity_series = climate_card["series"]["humidity"]
        temperature_series = climate_card["series"]["temperature"]
        humidity_series.clear()
        temperature_series.clear()

        for point in snapshot:
            timestamp = point.get("ts", 0) * 1000
            humidity = point.get(METRIC_CONFIG["humidity"]["field"])
            temperature = point.get(METRIC_CONFIG["temperature"]["field"])
            if humidity is not None:
                humidity_series.append(timestamp, float(humidity))
                humidity_values.append(humidity)
            if temperature is not None:
                temperature_series.append(timestamp, float(temperature))
                temperature_values.append(temperature)

        self._set_time_axis(climate_card["axis_x"], snapshot)
        humidity_range = _value_range(
            humidity_values,
            METRIC_CONFIG["humidity"]["default_range"],
            fixed_range=METRIC_CONFIG["humidity"].get("fixed_range"),
        )
        temperature_range = _value_range(
            temperature_values,
            METRIC_CONFIG["temperature"]["default_range"],
        )
        climate_card["axis_y"].setRange(*humidity_range)
        climate_card["axis_y_right"].setRange(*temperature_range)

    def _update_single_metric_chart(self, chart_def, snapshot, metric_key):
        series = chart_def["series"][metric_key]
        field = METRIC_CONFIG[metric_key]["field"]
        values = []
        series.clear()

        for point in snapshot:
            value = point.get(field)
            if value is None:
                continue
            values.append(value)
            series.append(point.get("ts", 0) * 1000, float(value))

        self._set_time_axis(chart_def["axis_x"], snapshot)
        lower, upper = _value_range(
            values,
            METRIC_CONFIG[metric_key]["default_range"],
            include_zero=METRIC_CONFIG[metric_key].get("include_zero", False),
            fixed_range=METRIC_CONFIG[metric_key].get("fixed_range"),
        )
        chart_def["axis_y"].setRange(lower, upper)

    def _set_time_axis(self, axis, snapshot):
        if not snapshot:
            now = int(time.time() * 1000)
            axis.setMin(QDateTime.fromMSecsSinceEpoch(now - 60_000))
            axis.setMax(QDateTime.fromMSecsSinceEpoch(now))
            return

        min_ts = int(snapshot[0].get("ts", 0) * 1000)
        max_ts = int(snapshot[-1].get("ts", 0) * 1000)
        if min_ts == max_ts:
            min_ts -= 60_000
            max_ts += 60_000

        axis.setMin(QDateTime.fromMSecsSinceEpoch(min_ts))
        axis.setMax(QDateTime.fromMSecsSinceEpoch(max_ts))
