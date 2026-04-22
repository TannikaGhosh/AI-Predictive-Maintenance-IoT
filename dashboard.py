import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.database import get_history, init_db
@st.cache_data
def get_cached_prediction(air_temp, process_temp, speed, torque, tool_wear):
    test_data = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }
    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict", json=test_data, timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            return float(result.get("probability", 0.0)), result.get("failure_risk", "Low")
    except Exception as e:
        st.warning(f"Could not reach Flask API: {e}")
    return 0.0, "Low"



@st.cache_data
def get_cached_prediction(air_temp, process_temp, speed, torque, tool_wear):
    test_data = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }
    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict", json=test_data, timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            return float(result.get("probability", 0.0)), result.get("failure_risk", "Low")
    except Exception as e:
        st.warning(f"Could not reach Flask API: {e}")
    return 0.0, "Low"


def check_critical_condition(sensor_data, probability):
    """Check sensor values against critical thresholds."""
    return (
        probability > 0.7
        or sensor_data["Process temperature [K]"] > 320
        or (
            sensor_data["Rotational speed [rpm]"] < 1100
            and sensor_data["Torque [Nm]"] > 60
        )
        or sensor_data["Tool wear [min]"] > 200
        and sensor_data["Torque [Nm]"] > 65
    )


st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("Predictive Maintenance Dashboard")

init_db()
history_df = get_history(limit=500)

if "manual_prediction" not in st.session_state:
    st.session_state["manual_prediction"] = None

if st.session_state["manual_prediction"]:
    history_df = pd.concat([history_df, pd.DataFrame([st.session_state["manual_prediction"]])], ignore_index=True)

if history_df.empty:
    st.info("No predictions logged yet. Start `src/flask_api.py` and send /predict requests.")
else:
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], errors="coerce")
    history_df = history_df.dropna(subset=["timestamp"])
    history_df = history_df.sort_values("timestamp")
    history_df = history_df.drop_duplicates(subset="timestamp", keep="last")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", len(history_df))
    c2.metric("High Risk Count", int((history_df["risk"] == "High").sum()))
    c3.metric("Latest Probability", f"{history_df["probability"].iloc[-1]:.3f}")

    # --- Container for Top Charts (Pie and Bar) ---
    st.divider()
    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.subheader("📊 Risk Distribution Overview")
        if len(history_df) > 0:
            risk_counts = history_df["risk"].value_counts()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4,
                marker_colors=["#ff6b6b", "#4ecdc4"],
                textinfo="label+percent",
                textposition="auto"
            )])
            fig_pie.update_layout(
                title="High Risk vs Low Risk Predictions",
                height=400,
                annotations=[dict(text=f"Total: {len(history_df)}", x=0.5, y=0.5, font_size=14, showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data yet. Start simulation to see charts.")

    with col_bar:
        st.subheader("📈 Alert Triggers Breakdown")
        if len(history_df) > 0:
            alert_counts = {
                "AI Prediction (>70%)": 0,
                "Overheating (>320K)": 0,
                "Mechanical Stress": 0,
                "Tool Wear Failure": 0
            }
            
            for _, row in history_df.iterrows():
                prob = row["probability"]
                temp = row["process_temp"]
                speed = row["speed"]
                torque = row["torque"]
                wear = row["tool_wear"]
                
                if prob > 0.7:
                    alert_counts["AI Prediction (>70%)"] += 1
                if temp > 320:
                    alert_counts["Overheating (>320K)"] += 1
                if speed < 1100 and torque > 60:
                    alert_counts["Mechanical Stress"] += 1
                if wear > 200 and torque > 65:
                    alert_counts["Tool Wear Failure"] += 1
            
            fig_bar = go.Figure(data=[go.Bar(
                x=list(alert_counts.keys()),
                y=list(alert_counts.values()),
                marker_color="#ff6b6b"
            )])
            fig_bar.update_layout(
                title="Alert Triggers Breakdown",
                xaxis_title="Alert Type",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No data yet. Start simulation to see charts.")

    # --- Gauge for Latest Probability ---
    st.subheader("Current Machine Status")
    if len(history_df) > 0:
        latest_prob = history_df["probability"].iloc[-1]
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_prob * 100,
            title={"text": "Failure Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgreen"},
                    {"range": [50, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 70
                }
            }
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("No data yet. Start simulation to see gauge.")


    st.subheader("Failure Probability Over Time")
    fig_line = go.Figure(data=[go.Scatter(x=history_df["timestamp"], y=history_df["probability"], mode="lines")])
    fig_line.update_layout(title="Failure Probability Over Time", xaxis_title="Time", yaxis_title="Probability")
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Recent Predictions")
    st.dataframe(history_df.tail(100), use_container_width=True)

    st.subheader("Past Alerts")
    alerts_df = history_df[history_df["probability"] > 0.7]
    st.dataframe(
        alerts_df[["timestamp", "probability", "risk"]].sort_values("timestamp", ascending=False),
        use_container_width=True,
    )

    # --- Manual Input Section for Testing Critical Conditions ---
    with st.expander("Manual Test - Enter Custom Sensor Values", expanded=False):
        st.markdown(
            "Use this panel to simulate any machine condition and verify risk and alert behavior."
        )

        preset = st.selectbox(
            "Pre-defined critical scenario",
            ["None", "Overheating", "Mechanical stress", "Tool wear failure"],
        )
        preset_values = {
            "air": 300.0,
            "process": 310.0,
            "speed": 1500.0,
            "torque": 40.0,
            "wear": 100.0,
        }
        if preset == "Overheating":
            preset_values["process"] = 325.0
        elif preset == "Mechanical stress":
            preset_values["speed"] = 1050.0
            preset_values["torque"] = 65.0
        elif preset == "Tool wear failure":
            preset_values["wear"] = 210.0
            preset_values["torque"] = 70.0

        col1, col2, col3 = st.columns(3)
        with col1:
            manual_air_temp = st.number_input(
                "Air temperature [K]",
                min_value=290.0,
                max_value=310.0,
                value=float(preset_values["air"]),
                step=1.0,
            )
            manual_process_temp = st.number_input(
                "Process temperature [K]",
                min_value=300.0,
                max_value=330.0,
                value=float(preset_values["process"]),
                step=1.0,
            )
        with col2:
            manual_speed = st.number_input(
                "Rotational speed [rpm]",
                min_value=1000.0,
                max_value=1600.0,
                value=float(preset_values["speed"]),
                step=10.0,
            )
            manual_torque = st.number_input(
                "Torque [Nm]",
                min_value=30.0,
                max_value=80.0,
                value=float(preset_values["torque"]),
                step=1.0,
            )
        with col3:
            manual_tool_wear = st.number_input(
                "Tool wear [min]",
                min_value=0.0,
                max_value=300.0,
                value=float(preset_values["wear"]),
                step=5.0,
            )

        if st.button("Get Prediction & Check Alert", type="primary"):
            test_data = {
                "Air temperature [K]": manual_air_temp,
                "Process temperature [K]": manual_process_temp,
                "Rotational speed [rpm]": manual_speed,
                "Torque [Nm]": manual_torque,
                "Tool wear [min]": manual_tool_wear,
            }
            try:
                response = requests.post(
                    "http://127.0.0.1:5000/predict", json=test_data, timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    prob = float(result.get("probability", 0.0))
                    risk = result.get("failure_risk", "Low")

                    st.subheader("Prediction Result")
                    col_res1, col_res2 = st.columns(2)
                    col_res1.metric("Failure Risk", risk)
                    col_res2.metric("Probability", f"{prob:.2%}")

                    # Store the prediction for graph update
                    new_row = {
                        "timestamp": pd.Timestamp.now(),
                        "air_temp": manual_air_temp,
                        "process_temp": manual_process_temp,
                        "speed": manual_speed,
                        "torque": manual_torque,
                        "tool_wear": manual_tool_wear,
                        "probability": prob,
                        "risk": risk
                    }
                    st.session_state["manual_prediction"] = new_row

                    if check_critical_condition(test_data, prob):
                        st.error(
                            "CRITICAL ALERT! This condition would trigger an immediate alert!",
                            icon="🚨",
                        )
                    else:
                        st.info(
                            "No critical condition - machine is operating within safe limits."
                        )
                else:
                    st.error(f"API error: {response.status_code}")
            except Exception as e:
                st.error(f"Could not reach Flask API. Make sure it is running. Error: {e}")

    latest = history_df.iloc[-1]
    latest_sensor_data = {
        "Air temperature [K]": float(latest["air_temp"]),
        "Process temperature [K]": float(latest["process_temp"]),
        "Rotational speed [rpm]": float(latest["speed"]),
        "Torque [Nm]": float(latest["torque"]),
        "Tool wear [min]": float(latest["tool_wear"]),
    }
    latest_prob = float(latest["probability"])
    latest_ts = str(latest["timestamp"])

    if check_critical_condition(latest_sensor_data, latest_prob):
        st.error("⚠️ CRITICAL ALERT! Immediate maintenance required! ⚠️", icon="🚨")

