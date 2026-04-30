import hashlib
import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "plant_system.db"
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin123"

_db_lock = threading.Lock()
_db_ready = False


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _format_recorded_at(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_sql: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")


def _build_time_filter_clause(start_ts: float | None, end_ts: float | None) -> tuple[str, list[float]]:
    clauses = []
    params: list[float] = []
    if start_ts is not None:
        clauses.append("ts >= ?")
        params.append(start_ts)
    if end_ts is not None:
        clauses.append("ts <= ?")
        params.append(end_ts)
    if not clauses:
        return "", params
    return f"WHERE {' AND '.join(clauses)}", params


def init_database() -> None:
    global _db_ready

    with _db_lock:
        if _db_ready:
            return

        with _connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sensor_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    ts REAL NOT NULL,
                    soil_humidity INTEGER,
                    soil_state TEXT,
                    air_temperature REAL,
                    air_humidity REAL,
                    wind_speed REAL,
                    light_lux REAL,
                    pressure_hpa REAL,
                    fan_pwm_percent INTEGER,
                    wind TEXT,
                    aht20 TEXT,
                    bh1750 TEXT,
                    bmp280 TEXT,
                    millis INTEGER,
                    raw_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recognition_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    ts REAL NOT NULL,
                    username TEXT,
                    plant_type TEXT,
                    plant_confidence REAL,
                    pest_detected INTEGER NOT NULL,
                    pest_label TEXT,
                    pest_confidence REAL,
                    summary TEXT NOT NULL,
                    interval_seconds REAL,
                    detections_json TEXT,
                    top_candidates_json TEXT,
                    image_path TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_sensor_records_ts ON sensor_records(ts);
                CREATE INDEX IF NOT EXISTS idx_recognition_records_ts ON recognition_records(ts);
                """
            )

            _ensure_column(conn, "recognition_records", "image_path", "TEXT")

            existing_user = conn.execute(
                "SELECT 1 FROM users WHERE username = ?",
                (DEFAULT_USERNAME,),
            ).fetchone()
            if existing_user is None:
                conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (DEFAULT_USERNAME, _hash_password(DEFAULT_PASSWORD)),
                )
            conn.commit()

        _db_ready = True


def verify_login(username: str, password: str) -> bool:
    init_database()

    with _db_lock:
        with _connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()

    return row is not None and row[0] == _hash_password(password)


def record_sensor_data(point: dict) -> None:
    init_database()

    timestamp = float(point.get("ts") or time.time())
    values = (
        _format_recorded_at(timestamp),
        timestamp,
        point.get("soil_humidity"),
        point.get("soil_state"),
        point.get("air_temperature"),
        point.get("air_humidity"),
        point.get("wind_speed"),
        point.get("light_lux"),
        point.get("pressure_hpa"),
        point.get("fan_pwm_percent"),
        point.get("wind"),
        point.get("aht20"),
        point.get("bh1750"),
        point.get("bmp280"),
        point.get("millis"),
        json.dumps(point, ensure_ascii=False),
    )

    with _db_lock:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO sensor_records (
                    recorded_at,
                    ts,
                    soil_humidity,
                    soil_state,
                    air_temperature,
                    air_humidity,
                    wind_speed,
                    light_lux,
                    pressure_hpa,
                    fan_pwm_percent,
                    wind,
                    aht20,
                    bh1750,
                    bmp280,
                    millis,
                    raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            conn.commit()


def record_recognition_result(payload: dict) -> None:
    init_database()

    timestamp = float(payload.get("ts") or time.time())
    values = (
        _format_recorded_at(timestamp),
        timestamp,
        payload.get("username"),
        payload.get("plant_type"),
        payload.get("plant_confidence"),
        1 if payload.get("pest_detected") else 0,
        payload.get("pest_label"),
        payload.get("pest_confidence"),
        payload.get("summary", ""),
        payload.get("interval_seconds"),
        json.dumps(payload.get("detections", []), ensure_ascii=False),
        json.dumps(payload.get("top_candidates", []), ensure_ascii=False),
        payload.get("image_path"),
    )

    with _db_lock:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO recognition_records (
                    recorded_at,
                    ts,
                    username,
                    plant_type,
                    plant_confidence,
                    pest_detected,
                    pest_label,
                    pest_confidence,
                    summary,
                    interval_seconds,
                    detections_json,
                    top_candidates_json,
                    image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            conn.commit()


def fetch_sensor_records(
    limit: int | None = None,
    start_ts: float | None = None,
    end_ts: float | None = None,
) -> list[dict]:
    init_database()

    where_clause, params = _build_time_filter_clause(start_ts, end_ts)
    sql = f"""
        SELECT
            id,
            recorded_at,
            ts,
            soil_humidity,
            soil_state,
            air_temperature,
            air_humidity,
            wind_speed,
            light_lux,
            pressure_hpa,
            fan_pwm_percent,
            wind,
            aht20,
            bh1750,
            bmp280,
            millis
        FROM sensor_records
        {where_clause}
        ORDER BY id DESC
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(float(limit))

    with _db_lock:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def fetch_recognition_records(
    limit: int | None = None,
    start_ts: float | None = None,
    end_ts: float | None = None,
) -> list[dict]:
    init_database()

    where_clause, params = _build_time_filter_clause(start_ts, end_ts)
    sql = f"""
        SELECT
            id,
            recorded_at,
            ts,
            username,
            plant_type,
            plant_confidence,
            pest_detected,
            pest_label,
            pest_confidence,
            summary,
            interval_seconds,
            detections_json,
            top_candidates_json,
            image_path
        FROM recognition_records
        {where_clause}
        ORDER BY id DESC
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(float(limit))

    with _db_lock:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def fetch_history_bounds() -> tuple[float, float]:
    init_database()

    with _db_lock:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts
                FROM (
                    SELECT ts FROM sensor_records
                    UNION ALL
                    SELECT ts FROM recognition_records
                )
                """
            ).fetchone()

    now = time.time()
    min_ts = row[0] if row and row[0] is not None else now - 86400
    max_ts = row[1] if row and row[1] is not None else now
    return float(min_ts), float(max_ts)
