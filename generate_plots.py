import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
sns.set_theme(style="darkgrid")
plt.style.use("dark_background")

# 1. Dataset Preview (simulated as a table)
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
data = {
    'timestamp': ['2026-04-23 10:00', '2026-04-23 10:01', '2026-04-23 10:02', '2026-04-23 10:03'],
    'temperature': [72.4, 72.5, 75.1, 88.3],
    'vibration': [0.12, 0.13, 0.15, 0.85],
    'pressure': [101.2, 101.1, 101.0, 95.5],
    'failure_imminent': [0, 0, 0, 1]
}
df = pd.DataFrame(data)
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
table.scale(1, 2)
table.set_fontsize(14)
plt.title("Dataset Preview (Sample Telemetry)", fontsize=18, color='white', pad=20)
plt.savefig('images/dataset_preview.png', bbox_inches='tight', dpi=150)
plt.close()

# 2. Preprocessing Output (Feature distribution before/after)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
raw_data = np.random.normal(loc=10, scale=5, size=1000)
scaled_data = (raw_data - np.mean(raw_data)) / np.std(raw_data)
sns.histplot(raw_data, ax=axes[0], color='red', kde=True)
axes[0].set_title('Raw Vibration Data', color='white')
sns.histplot(scaled_data, ax=axes[1], color='cyan', kde=True)
axes[1].set_title('Standardized Vibration Data', color='white')
plt.suptitle('Preprocessing Output: Standardization', fontsize=18, color='white')
plt.savefig('images/preprocessing.png', bbox_inches='tight', dpi=150)
plt.close()

# 3. Model Training Logs (Loss curve)
epochs = np.arange(1, 51)
train_loss = np.exp(-epochs/10) + np.random.normal(0, 0.02, 50)
val_loss = np.exp(-epochs/10) + 0.1 + np.random.normal(0, 0.03, 50)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Train Loss', color='cyan', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='magenta', linewidth=2)
plt.title('Model Training Logs (Log Loss)', fontsize=18, color='white')
plt.xlabel('Epochs', color='white')
plt.ylabel('Loss', color='white')
plt.legend()
plt.savefig('images/training_logs.png', bbox_inches='tight', dpi=150)
plt.close()

# 4. Prediction Output (Scatter plot of predictions vs actual)
plt.figure(figsize=(10, 5))
true_vals = np.random.rand(100)
pred_vals = true_vals + np.random.normal(0, 0.1, 100)
plt.scatter(true_vals, pred_vals, color='lime', alpha=0.7)
plt.plot([0, 1], [0, 1], color='white', linestyle='--')
plt.title('Prediction Output vs Actual Risk Score', fontsize=18, color='white')
plt.xlabel('Actual Risk Score', color='white')
plt.ylabel('Predicted Risk Score', color='white')
plt.savefig('images/prediction_output.png', bbox_inches='tight', dpi=150)
plt.close()

# 5. Failure Detection Graph (Time series with anomalies)
time = np.arange(0, 100)
sensor_val = np.sin(time/5) + np.random.normal(0, 0.2, 100)
sensor_val[75:80] += 3  # Add anomaly
plt.figure(figsize=(12, 5))
plt.plot(time, sensor_val, color='cyan', label='Vibration Sensor', linewidth=2)
plt.axvspan(74, 81, color='red', alpha=0.3, label='Predicted Failure Window')
plt.scatter(time[75:80], sensor_val[75:80], color='red', zorder=5, s=50)
plt.title('Real-Time Failure Detection Graph', fontsize=18, color='white')
plt.xlabel('Time', color='white')
plt.ylabel('Sensor Reading', color='white')
plt.legend()
plt.savefig('images/failure_graph.png', bbox_inches='tight', dpi=150)
plt.close()

# 6. Confusion Matrix
from sklearn.metrics import confusion_matrix
y_true = [0]*80 + [1]*20
y_pred = [0]*75 + [1]*5 + [0]*2 + [1]*18
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Failure'], yticklabels=['Normal', 'Failure'], 
            annot_kws={"size": 16})
plt.title('Confusion Matrix', fontsize=18, color='white')
plt.xlabel('Predicted Label', color='white')
plt.ylabel('True Label', color='white')
plt.savefig('images/confusion_matrix.png', bbox_inches='tight', dpi=150)
plt.close()
