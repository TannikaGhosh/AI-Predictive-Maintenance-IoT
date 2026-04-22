import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from data_preprocessing import load_data, preprocess_data, split_data

# Load and prepare data
df = load_data()
X, y, scaler = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Train XGBoost (handles imbalance well)
model = XGBClassifier(scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]))
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("outputs/figures/confusion_matrix.png")
print("Model saved and confusion matrix generated.")
