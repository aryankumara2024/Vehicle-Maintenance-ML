"""
Streamlit UI for Vehicle Maintenance Risk Prediction.

Features:
- Upload CSV or enter values manually
- Predict failure risk (High / Low)
- Display model performance metrics
- Show feature importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# ── Page configuration ─────────────────────────────────────
st.set_page_config(
    page_title="🚗 Vehicle Maintenance Risk Predictor",
    page_icon="🚗",
    layout="wide",
)

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


@st.cache_resource
def load_artefacts():
    """Load saved model artefacts."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            "model.pkl not found. Run `python src/train.py` first to train and save the models."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


artefacts = load_artefacts()
best_model = artefacts["best_model"]
scaler = artefacts["scaler"]
feature_names = artefacts["feature_names"]
lr_metrics = artefacts["lr_metrics"]
dt_metrics = artefacts["dt_metrics"]
best_name = artefacts["best_model_name"]

# Leakage columns to strip from uploaded CSVs
LEAKAGE_COLS = [
    "Remaining Useful Life", "RUL",
    "Time to Failure", "TTF",
    "Component Health Score", "Component_Health_Score",
]

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Predict", "📊 Model Performance", "🌳 Feature Importance"])

# ══════════════════════════════════════════════════════════
# PAGE 1: Prediction
# ══════════════════════════════════════════════════════════
if page == "🏠 Predict":
    st.title("🚗 Vehicle Maintenance Risk Prediction")
    st.markdown(
        "Predict whether a vehicle is at **HIGH RISK** of failure based on telemetry data."
    )

    input_mode = st.radio("Input method", ["📂 Upload CSV", "✏️ Manual Input"], horizontal=True)

    # ── CSV Upload ─────────────────────────────────────────
    if input_mode == "📂 Upload CSV":
        uploaded = st.file_uploader("Upload a CSV file with vehicle telemetry data", type="csv")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Drop leakage & non-feature columns
            for col in LEAKAGE_COLS + ["Timestamp", "Failure_Probability"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            # Feature engineering
            if "Battery_Current" in df.columns and "Battery_Temperature" in df.columns:
                df["Battery_Stress"] = df["Battery_Current"] * df["Battery_Temperature"]
            if "Motor_Torque" in df.columns and "Motor_RPM" in df.columns:
                df["Motor_Load"] = df["Motor_Torque"] * df["Motor_RPM"]
            if "Distance_Traveled" in df.columns and "Charge_Cycles" in df.columns:
                df["Wear_Index"] = df["Distance_Traveled"] / df["Charge_Cycles"].replace(0, np.nan)

            # Align columns
            missing_cols = set(feature_names) - set(df.columns)
            for c in missing_cols:
                df[c] = 0
            df = df[feature_names]

            # Fill NaN with 0 for prediction
            df = df.fillna(0)

            # Scale & predict
            X_scaled = scaler.transform(df)
            preds = best_model.predict(X_scaled)

            df_out = df.copy()
            df_out["Prediction"] = preds
            df_out["Risk"] = df_out["Prediction"].map({0: "✅ Low Risk", 1: "🚨 High Risk"})

            st.subheader("Predictions")
            st.dataframe(df_out[["Risk"]].join(df_out.drop(columns=["Prediction", "Risk"]).head()), use_container_width=True)

            high = int((preds == 1).sum())
            low = int((preds == 0).sum())
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Samples", len(preds))
            col2.metric("🚨 High Risk", high)
            col3.metric("✅ Low Risk", low)

    # ── Manual Input ───────────────────────────────────────
    else:
        st.subheader("Enter Vehicle Telemetry Values")
        cols = st.columns(3)

        input_fields = {
            "SoC": (0.0, 1.0, 0.8),
            "SoH": (0.0, 1.0, 0.9),
            "Battery_Voltage": (100.0, 500.0, 350.0),
            "Battery_Current": (-50.0, 50.0, -20.0),
            "Battery_Temperature": (-10.0, 80.0, 30.0),
            "Charge_Cycles": (0.0, 500.0, 150.0),
            "Motor_Temperature": (20.0, 120.0, 55.0),
            "Motor_Vibration": (0.0, 5.0, 1.0),
            "Motor_Torque": (0.0, 300.0, 120.0),
            "Motor_RPM": (0.0, 5000.0, 2000.0),
            "Power_Consumption": (0.0, 100.0, 25.0),
            "Brake_Pad_Wear": (0.0, 1.0, 0.2),
            "Brake_Pressure": (0.0, 100.0, 40.0),
            "Reg_Brake_Efficiency": (0.0, 1.0, 0.85),
            "Tire_Pressure": (20.0, 45.0, 32.0),
            "Tire_Temperature": (10.0, 80.0, 30.0),
            "Suspension_Load": (0.0, 500.0, 150.0),
            "Ambient_Temperature": (-20.0, 50.0, 20.0),
            "Ambient_Humidity": (0.0, 100.0, 50.0),
            "Load_Weight": (0.0, 2000.0, 700.0),
            "Driving_Speed": (0.0, 200.0, 60.0),
            "Distance_Traveled": (0.0, 500.0, 50.0),
            "Idle_Time": (0.0, 1.0, 0.5),
            "Route_Roughness": (0.0, 1.0, 0.2),
            "Maintenance_Type": (0.0, 3.0, 1.0),
        }

        values = {}
        for i, (name, (lo, hi, default)) in enumerate(input_fields.items()):
            with cols[i % 3]:
                values[name] = st.number_input(name, min_value=lo, max_value=hi, value=default, step=0.01)

        if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
            row = pd.DataFrame([values])

            # Feature engineering
            row["Battery_Stress"] = row["Battery_Current"] * row["Battery_Temperature"]
            row["Motor_Load"] = row["Motor_Torque"] * row["Motor_RPM"]
            row["Wear_Index"] = row["Distance_Traveled"] / row["Charge_Cycles"].replace(0, np.nan)
            row = row.fillna(0)

            # Align columns
            missing_cols = set(feature_names) - set(row.columns)
            for c in missing_cols:
                row[c] = 0
            row = row[feature_names]

            X_scaled = scaler.transform(row)
            pred = best_model.predict(X_scaled)[0]

            st.markdown("---")
            if pred == 1:
                st.error("## 🚨 HIGH RISK — Maintenance Required!")
                st.markdown("The model predicts that this vehicle is **at risk of failure**.")
            else:
                st.success("## ✅ LOW RISK — Vehicle is Healthy")
                st.markdown("The model predicts that this vehicle is **not at risk of failure**.")

# ══════════════════════════════════════════════════════════
# PAGE 2: Model Performance
# ══════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Comparison")
    st.markdown(f"**Best model selected:** {best_name}")

    # Metrics table
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"],
            "Logistic Regression": [
                lr_metrics["accuracy"],
                lr_metrics["precision"],
                lr_metrics["recall"],
                lr_metrics["f1"],
                lr_metrics["roc_auc"],
            ],
            "Decision Tree": [
                dt_metrics["accuracy"],
                dt_metrics["precision"],
                dt_metrics["recall"],
                dt_metrics["f1"],
                dt_metrics["roc_auc"],
            ],
        }
    )
    st.dataframe(metrics_df.style.format({"Logistic Regression": "{:.4f}", "Decision Tree": "{:.4f}"}),
                 use_container_width=True)

    # Confusion matrices side by side
    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Logistic Regression")
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.heatmap(lr_metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax1)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        st.pyplot(fig1)

    with col2:
        st.markdown("#### Decision Tree")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.heatmap(dt_metrics["confusion_matrix"], annot=True, fmt="d", cmap="Greens", ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)

# ══════════════════════════════════════════════════════════
# PAGE 3: Feature Importance
# ══════════════════════════════════════════════════════════
elif page == "🌳 Feature Importance":
    st.title("🌳 Feature Importance (Decision Tree)")

    dt_model = artefacts["dt_model"]
    importances = dt_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    top_names = [feature_names[i] for i in sorted_idx]
    top_vals = importances[sorted_idx]
    ax.barh(top_names[::-1], top_vals[::-1], color="teal")
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("All Feature Importances")
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    st.dataframe(fi_df, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Scikit-learn & Streamlit")
