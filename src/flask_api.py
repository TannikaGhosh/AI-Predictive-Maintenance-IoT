import os
import smtplib
from email.mime.text import MIMEText

from flask import Flask, jsonify, request
import joblib
import numpy as np

from database import init_db, log_prediction

app = Flask(__name__)
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.7"))


def send_email_alert(probability, sensor_data):
    """Send an email alert when failure probability crosses threshold."""
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")
    email_to = os.getenv("ALERT_RECIPIENT", email_user)

    if not email_user or not email_pass or not email_to:
        print("Email alert skipped: set EMAIL_USER, EMAIL_PASS, and ALERT_RECIPIENT env vars.")
        return

    subject = "Predictive Maintenance Alert: High Failure Risk"
    body = f"""
Machine failure probability: {probability:.2%}

Sensor readings:
- Air temperature: {sensor_data['Air temperature [K]']} K
- Process temperature: {sensor_data['Process temperature [K]']} K
- Rotational speed: {sensor_data['Rotational speed [rpm]']} rpm
- Torque: {sensor_data['Torque [Nm]']} Nm
- Tool wear: {sensor_data['Tool wear [min]']} min

Immediate maintenance recommended.
"""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = email_user
    msg["To"] = email_to

    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(email_user, email_pass)
        server.send_message(msg)
    print("Alert email sent.")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "GET":
        return jsonify({"message": "This endpoint requires POST with JSON data. Use the dashboard for predictions."}), 200
    data = request.get_json()
    # Expected keys: 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    features = np.array(
        [
            [
                data["Air temperature [K]"],
                data["Process temperature [K]"],
                data["Rotational speed [rpm]"],
                data["Torque [Nm]"],
                data["Tool wear [min]"],
            ]
        ]
    )
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]
    risk = "High" if pred == 1 else "Low"

    log_prediction(data, float(prob), risk)
    if prob > ALERT_THRESHOLD:
        send_email_alert(prob, data)

    return jsonify({"failure_risk": risk, "probability": round(float(prob), 3)})


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
