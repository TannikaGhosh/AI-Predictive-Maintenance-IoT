import sqlite3
from datetime import datetime

import pandas as pd

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DB_PATH = str(BASE_DIR / "predictions.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  air_temp REAL,
                  process_temp REAL,
                  speed REAL,
                  torque REAL,
                  tool_wear REAL,
                  probability REAL,
                  risk TEXT)"""
    )
    conn.commit()
    conn.close()


def log_prediction(sensor_data, probability, risk):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO predictions 
                 (timestamp, air_temp, process_temp, speed, torque, tool_wear, probability, risk)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            sensor_data["Air temperature [K]"],
            sensor_data["Process temperature [K]"],
            sensor_data["Rotational speed [rpm]"],
            sensor_data["Torque [Nm]"],
            sensor_data["Tool wear [min]"],
            probability,
            risk,
        ),
    )
    conn.commit()
    conn.close()


def get_history(limit=1000):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {int(limit)}", conn
    )
    conn.close()
    return df
