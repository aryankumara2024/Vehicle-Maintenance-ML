# 🚗 ML Capstone Project – Milestone 1
## Project 6: Vehicle Maintenance Risk Prediction

---

# 1️⃣ PROJECT OVERVIEW

This project involves designing and implementing a machine learning-based vehicle maintenance prediction system.

Milestone 1 focuses strictly on **classical machine learning techniques** applied to electric vehicle telemetry and maintenance data.

⚠ IMPORTANT:
- NO LLMs
- NO Agentic AI
- NO GenAI-based methods
- ONLY traditional ML using Scikit-Learn

This milestone is evaluated strictly on classical ML workflows.

---

# 2️⃣ OBJECTIVE

Design and implement a supervised machine learning system that:

• Accepts vehicle telemetry & maintenance CSV data  
• Performs preprocessing & feature engineering  
• Predicts maintenance failure risk  
• Evaluates model performance  
• Displays predictions in a user interface  
• Is publicly deployed (NOT localhost)

---

# 3️⃣ DATASET INFORMATION

Dataset: EV_Predictive_Maintenance_Dataset.csv

Records: 175,393  
Time span: Jan 2020 – Jan 2025  
Sampling: Every 15 minutes  

Features include:

## Battery Monitoring
- SoC
- SoH
- Battery Voltage
- Battery Current
- Battery Temperature
- Charge Cycles

## Motor & Drivetrain
- Motor Temperature
- Motor Vibration
- Motor Torque
- Motor RPM
- Power Consumption

## Brake System
- Brake Pad Wear
- Brake Pressure
- Regenerative Braking Efficiency

## Tire & Suspension
- Tire Pressure
- Tire Temperature
- Suspension Load

## Environmental & Usage
- Ambient Temperature
- Ambient Humidity
- Load Weight
- Driving Speed
- Distance Traveled
- Idle Time
- Route Roughness

## Maintenance Records
- Maintenance Type (0=None, 1=Preventive, 2=Corrective, 3=Predictive)

## Target Labels Available
- Remaining Useful Life (RUL)
- Failure Probability (0/1)
- Time to Failure (TTF)
- Component Health Score (0–1)

---

# 4️⃣ TARGET FOR MILESTONE 1

Use:

TARGET = Failure Probability (Binary Classification)

Problem formulation:

Predict whether the vehicle is at HIGH RISK of failure.

Classification:
0 → No failure risk
1 → Failure risk

---

# 5️⃣ IMPORTANT: REMOVE DATA LEAKAGE

The following columns MUST be removed:

- Remaining Useful Life
- Time to Failure
- Component Health Score

These directly encode failure information.

---

# 6️⃣ REQUIRED ML MODELS

Must implement:

1. Logistic Regression
2. Decision Tree Classifier

Framework:
Scikit-learn

Optional (bonus):
- Cross-validation
- Hyperparameter tuning
- ROC Curve

---

# 7️⃣ PREPROCESSING REQUIREMENTS

Must include:

• Handling missing values (median or mean imputation)
• Feature scaling (StandardScaler for Logistic Regression)
• Encoding categorical variables if present
• Train-test split (80-20 recommended)

---

# 8️⃣ EVALUATION METRICS

For classification:

• Accuracy
• Precision
• Recall
• F1-score
• Confusion Matrix

Must compare:
Logistic Regression vs Decision Tree

Include interpretation of results.

---

# 9️⃣ FEATURE ENGINEERING (RECOMMENDED)

Create derived features such as:

Battery_Stress = Battery_Current * Battery_Temperature
Motor_Load = Motor_Torque * Motor_RPM
Wear_Index = Distance_Traveled / Charge_Cycles

Document reasoning behind feature creation.

---

# 🔟 EXPECTED OUTPUT

The system must:

• Output predicted failure risk (0 or 1)
• Display model accuracy
• Show confusion matrix
• Show feature importance (Decision Tree)

---

# 1️⃣1️⃣ USER INTERFACE REQUIREMENT

Use:

Streamlit (recommended)

The UI must:

• Allow CSV upload
OR
• Allow manual input of feature values

Display:
• Prediction
• Risk message (High / Low)
• Model performance metrics

---

# 1️⃣2️⃣ DEPLOYMENT (MANDATORY)

Must deploy publicly on one of:

• Streamlit Community Cloud
• HuggingFace Spaces
• Render (Free Tier)

⚠ Localhost-only demos will NOT be accepted.

---

# 1️⃣3️⃣ GITHUB REQUIREMENTS

Repository must include:

vehicle-maintenance-ml/
│
├── data/
├── notebooks/
├── src/
├── app.py
├── requirements.txt
├── model.pkl
└── README.md

README must contain:
• Problem statement
• Dataset description
• Model explanation
• Results
• Deployment link
• Setup instructions

---

# 1️⃣4️⃣ REPORT REQUIREMENTS (LATEX)

Sections:

1. Abstract
2. Introduction
3. Dataset Description
4. Methodology
5. Preprocessing
6. Model Implementation
7. Results & Comparison
8. Feature Importance Analysis
9. Conclusion

Include:
• Accuracy table
• Confusion matrix image
• Feature importance graph

---

# 1️⃣5️⃣ VIVA PREPARATION NOTES

Be able to explain:

• Why Logistic Regression?
• Why Decision Tree?
• Why scaling is required?
• What is overfitting?
• What is data leakage?
• Why remove RUL & TTF?
• Why 80-20 split?
• Why Decision Tree performed better/worse?

---

# 1️⃣6️⃣ CODING CONSTRAINTS

• Use Python
• Use Scikit-learn
• No deep learning frameworks
• No LLM APIs
• No Agent frameworks
• No LangGraph
• No RAG

This milestone is strictly traditional ML.

---

# 1️⃣7️⃣ SUCCESS CRITERIA

The project is considered successful if:

✓ Code runs without errors  
✓ Both models are implemented  
✓ Performance metrics are shown  
✓ UI works  
✓ Hosted link is public  
✓ GitHub repository is clean & structured  

---

# END OF CONTEXT