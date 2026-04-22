from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def load_data():
    csv_path = Path("data/raw/ai4i2020.csv")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print("Data loaded from CSV. Shape:", df.shape)
    else:
        print("Local CSV not found. Fetching dataset from UCI...")
        dataset = fetch_ucirepo(id=601)
        X = dataset.data.features
        y = dataset.data.targets

        if "Machine failure" not in y.columns and len(y.columns) > 0:
            y = y.rename(columns={y.columns[0]: "Machine failure"})

        df = pd.concat([X, y], axis=1)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print("Data fetched from UCI and saved locally. Shape:", df.shape)

    df = df.rename(
        columns={
            "Air temperature": "Air temperature [K]",
            "Process temperature": "Process temperature [K]",
            "Rotational speed": "Rotational speed [rpm]",
            "Torque": "Torque [Nm]",
            "Tool wear": "Tool wear [min]",
        }
    )
    print("Data loaded. Shape:", df.shape)
    return df


def preprocess_data(df):
    # Drop unnecessary columns when present
    df = df.drop(["UDI", "Product ID"], axis=1, errors="ignore")

    # Use the same core sensor features expected by the API/simulator
    feature_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]
    y = df["Machine failure"]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
