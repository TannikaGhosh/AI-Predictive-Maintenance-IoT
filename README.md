# AI-Powered Predictive Maintenance for IoT 🛠️🤖

## A. Project Explanation 📖

### Overview
This project implements an AI-driven Predictive Maintenance system for IoT sensors. It processes telemetry data from simulated industrial machines to detect anomalies and predict impending failures before they occur, reducing downtime and maintenance costs.

### Problem Statement
Unplanned downtime in industrial environments leads to significant financial losses. Traditional maintenance is either reactive (fix when broken) or preventive (fix on a schedule, often unnecessarily). We need a predictive approach that leverages machine learning to anticipate failures based on real-time sensor data.

### Industry Relevance
Predictive maintenance is a critical component of Industry 4.0. It is heavily utilized in manufacturing, aviation, energy, and logistics to optimize operations, enhance safety, and extend equipment lifespan.

## B. Tech Stack Options ⚙️
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, TensorFlow/Keras
- **Web Dashboard**: Streamlit, Dash, Gradio
- **Database**: SQLite, PostgreSQL, InfluxDB (Time Series)
- **Containerization**: Docker

## C. Selected Approach 🎯
- **Core ML**: Random Forest & XGBoost (for robust tabular data classification).
- **Backend/Data Processing**: Python, Pandas.
- **Frontend/Dashboard**: Streamlit (for rapid, interactive data visualization).
- **Storage**: SQLite (lightweight, zero-configuration local database for prediction storage).

## D. Architecture 🏗️
1. **IoT Sensor Simulation**: Real-time generation of machine telemetry (temperature, vibration, pressure).
2. **Data Ingestion**: Streaming data into the processing pipeline.
3. **Feature Engineering**: Creating rolling averages and anomaly flags.
4. **Model Inference**: Evaluating data against a trained ML model to predict failures.
5. **Database Storage**: Saving predictions and raw data to SQLite.
6. **Dashboard**: Streamlit UI fetching data from SQLite to display real-time metrics, risk scores, and alerts.

## E. Folder Structure 📁

```text
AI-Predictive-Maintenance-IoT/
│
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for EDA and model training
├── src/                   # Source code modules
│   ├── database.py        # Database operations
│   ├── data_processor.py  # Data cleaning and feature engineering
│   └── model.py           # Model training and prediction logic
├── models/                # Saved, trained models (.pkl or .joblib)
├── outputs/               # Logs, generated reports
├── images/                # Output plots and diagrams
├── docs/                  # Additional documentation
├── .streamlit/            # Streamlit configuration secrets
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore file
├── dashboard.py           # Main Streamlit dashboard application
├── simulate_realtime.py   # Script to simulate real-time sensor data
└── Dockerfile             # Docker configuration
```

## F. Installation 💻

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AI-Predictive-Maintenance-IoT.git
   cd AI-Predictive-Maintenance-IoT
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## G. Code & H. Simulation & I. Execution 🚀

1. **Initialize the Database & Train Model (if needed via Notebooks/src)**
   *(Ensure your model is saved in the `models/` directory)*

2. **Run the Real-time Sensor Simulation:**
   In a new terminal, run:
   ```bash
   python simulate_realtime.py
   ```
   *This script generates synthetic IoT data and stores it in the SQLite database.*

3. **Launch the Streamlit Dashboard:**
   In another terminal, run:
   ```bash
   streamlit run dashboard.py
   ```
   *Access the dashboard at `http://localhost:8501` to view real-time predictions and analytics.*

## J. GitHub Steps & L. Commit Plan (Step-by-Step GitHub Proof Plan) 📅

- **Day 1 – Setup:** Initialize repository, `.gitignore`, virtual environment, `requirements.txt`.
  - *Proof: First commit with base files.*
- **Day 2 – Dataset:** Add raw dataset (or script to generate it) and load functions.
  - *Proof: `data/` folder populated, initial load scripts.*
- **Day 3 – Preprocessing:** Data cleaning, handling missing values, feature engineering.
  - *Proof: `src/data_processor.py` completed.*
- **Day 4 – Model:** Train and save the predictive ML model.
  - *Proof: `models/` folder populated with saved model, training notebook added.*
- **Day 5 – Evaluation:** Add evaluation metrics (confusion matrix, accuracy).
  - *Proof: Output logs and classification reports generated.*
- **Day 6 – Visualization:** Build Streamlit dashboard.
  - *Proof: `dashboard.py` running successfully.*
- **Day 7 – Upload:** Finalize README, clean code, push all changes.
  - *Proof: Completed GitHub repository.*

## K. Full Project Implementation Plan 📝

- **Phase 1 – Setup:** Environment creation and repo initialization. (Why: Foundation. Output: Repo ready.)
- **Phase 2 – Dataset Loading:** Read data from CSV or simulator. (Why: Need data. Output: Pandas DataFrame.)
- **Phase 3 – Data Cleaning:** Handle nulls, duplicates. (Why: Garbage in, garbage out. Output: Clean DataFrame.)
- **Phase 4 – Feature Engineering:** Rolling means, variance. (Why: Helps model find patterns. Output: Enriched features.)
- **Phase 5 – Model Building:** Train Random Forest/XGBoost. (Why: Prediction engine. Output: Trained model object.)
- **Phase 6 – Evaluation:** Precision, Recall, F1. (Why: Ensure model accuracy. Output: Metrics report.)
- **Phase 7 – Failure Prediction:** Run inference on new data. (Why: Core goal. Output: 0/1 failure flag.)
- **Phase 8 – Visualization:** Streamlit UI. (Why: User accessibility. Output: Interactive web app.)
- **Phase 9 – GitHub Publishing:** Commit and push. (Why: Sharing and version control. Output: Live repo.)
- **Phase 10 – Final Output:** Project presentation and documentation.


### Learning Outcomes 🧠
- Mastering time-series and sensor data processing.
- Implementing End-to-End Machine Learning pipelines.
- Building real-time interactive web applications with Streamlit.
- Understanding database integration for ML applications.
