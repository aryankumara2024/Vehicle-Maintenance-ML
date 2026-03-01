# 🚗 Vehicle Maintenance Risk Prediction

A machine learning system that predicts whether an electric vehicle is at **high risk of failure** based on real-time telemetry data. Built using classical ML techniques (Scikit-learn) as part of the ML Capstone – Milestone 1.

---

## 📋 Problem Statement

Electric vehicles generate vast amounts of telemetry data from batteries, motors, brakes, tires, and environmental sensors. This project uses supervised machine learning to **predict failure risk** (binary classification) from that data, enabling proactive maintenance scheduling and reducing breakdowns.

**Target:** `Failure_Probability` (0 = No risk, 1 = At risk)

---

## 📊 Dataset

- **Source:** `EV_Predictive_Maintenance_Dataset.csv`
- **Records:** 175,393
- **Time span:** Jan 2020 – Jan 2025
- **Sampling:** Every 15 minutes

### Feature Groups

| Category | Features |
|----------|----------|
| Battery | SoC, SoH, Voltage, Current, Temperature, Charge Cycles |
| Motor & Drivetrain | Temperature, Vibration, Torque, RPM, Power Consumption |
| Brake System | Pad Wear, Pressure, Regenerative Braking Efficiency |
| Tire & Suspension | Tire Pressure, Tire Temperature, Suspension Load |
| Environment & Usage | Ambient Temp/Humidity, Load Weight, Speed, Distance, Idle Time, Route Roughness |
| Maintenance | Maintenance Type (0=None, 1=Preventive, 2=Corrective, 3=Predictive) |

### Data Leakage Columns (Removed)

- `RUL` (Remaining Useful Life)
- `TTF` (Time to Failure)
- `Component_Health_Score`

These directly encode failure information and were removed to prevent data leakage.

---

## 🧠 Models

### 1. Logistic Regression
- Linear binary classifier using sigmoid function
- Feature scaling required (StandardScaler applied)
- `class_weight='balanced'` to handle class imbalance

### 2. Decision Tree Classifier
- Non-linear model that splits features hierarchically
- Provides feature importance rankings
- Hyperparameter tuning via GridSearchCV
- `class_weight='balanced'` to handle class imbalance

### Feature Engineering
Three derived features were created:
- **Battery_Stress** = `Battery_Current × Battery_Temperature`
- **Motor_Load** = `Motor_Torque × Motor_RPM`
- **Wear_Index** = `Distance_Traveled / Charge_Cycles`

---

## 📈 Results

| Metric | Logistic Regression | Decision Tree |
|--------|-------------------|---------------|
| Accuracy | 0.5058 | 0.3489 |
| Precision | 0.0986 | 0.0981 |
| Recall | 0.4916 | 0.6821 |
| F1-Score | 0.1643 | 0.1715 |
| ROC AUC | 0.4978 | 0.4961 |

**Best Model:** Decision Tree (selected by highest F1-Score)

> **Note:** The dataset's `Failure_Probability` target has weak correlation with the telemetry features, which is realistic for predictive maintenance scenarios. The models are trained with balanced class weights to address the ~90/10 class imbalance.

---

## 🖥️ User Interface

Built with **Streamlit**, the UI provides:

- **CSV Upload** — Upload vehicle telemetry data for batch prediction
- **Manual Input** — Enter individual feature values for single prediction
- **Risk Display** — Shows HIGH RISK 🚨 or LOW RISK ✅
- **Model Performance** — Side-by-side comparison of both models with confusion matrices
- **Feature Importance** — Top 15 features from the Decision Tree

---

## 🚀 Deployment

**Live App:** [Deployed on Streamlit Community Cloud](https://vehicle-maintenance-ml.streamlit.app)

---

## 🗂️ Project Structure

```
vehicle-maintenance-ml/
├── data/
│   └── EV_Predictive_Maintenance_Dataset.csv
├── notebooks/
│   └── eda_training.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   └── train.py
├── app.py
├── model.pkl
├── requirements.txt
├── context.md
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Models
```bash
python src/train.py
```
This preprocesses the data, trains both models, and saves `model.pkl`.

### Run the App Locally
```bash
streamlit run app.py
```

### Run the Notebook
Open `notebooks/eda_training.ipynb` in Jupyter or VS Code and run all cells.

---

## 🔧 Technologies

- **Python 3.12**
- **Scikit-learn** — ML models, preprocessing, evaluation
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
- **Streamlit** — Web UI
- **Joblib** — Model serialization

---

## 📄 License

This project is for academic purposes (ML Capstone – Milestone 1).
