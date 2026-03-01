# 🚗 Vehicle Maintenance Risk Prediction

> **ML Capstone Project – Milestone 1**

---

## 1. Problem Statement

The objective of this project is to develop a machine learning system that predicts vehicle maintenance risk using historical telemetry and operational data.

This is formulated as a binary classification problem:

| Label | Meaning |
|-------|---------|
| `0` | No Failure Risk |
| `1` | High Failure Risk |

---

## 2. Dataset

This project supports any vehicle telemetry dataset containing operational, environmental, and maintenance-related features.

### Example Input Features

| Category | Features |
|----------|----------|
| Usage | Mileage / Distance Traveled, Engine Hours |
| Battery | Battery Temperature |
| Motor | Motor Vibration, Motor RPM |
| Environment | Load Weight, Ambient Temperature |
| Diagnostics | Fault Code Count |
| Brake & Tire | Brake Wear, Tire Pressure |

### Target Variable

- `Failure Probability` (0/1)
  or
- `Maintenance Required` (0/1)

### Data Leakage Prevention

To prevent data leakage, the following columns (if present) are removed:

- `Remaining Useful Life (RUL)`
- `Time to Failure (TTF)`
- `Component Health Score`

---

## 3. Methodology

### Data Preprocessing

- Missing value imputation (median strategy)
- Feature scaling using `StandardScaler`
- Encoding of categorical variables (if applicable)
- Train–test split (80–20)

### Models Implemented

1. **Logistic Regression**
2. **Decision Tree Classifier**

---

## 4. Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

A comparative analysis is performed between Logistic Regression and Decision Tree to determine the better-performing model.

---

## 5. Feature Importance

Feature importance from the Decision Tree model is analyzed to identify key predictors contributing to maintenance risk.

---

## 6. Project Structure

```
vehicle-maintenance-ml/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── train.py
│   ├── preprocessing.py
│   └── evaluate.py
│
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

## 7. Running the Project

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Train the model:**

```bash
python src/train.py
```

**Run the application:**

```bash
streamlit run app.py
```

---

## 8. Deployment

The application is publicly deployed at:

🔗 **Live Demo:** [Insert Deployment Link]

> Localhost-only demonstrations are not accepted.

---

## 9. Conclusion

This project demonstrates the use of classical machine learning techniques for predictive maintenance in vehicle systems. The comparison between linear and nonlinear models provides insights into modeling maintenance risk using telemetry data.

---

|---|---|
| **Author** | Shreya, Aryan Kumar, Pranjal, Ritesh |
| **Course** | ML Capstone |
| **Milestone** | 1 |\n