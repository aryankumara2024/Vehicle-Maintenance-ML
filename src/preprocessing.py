"""
Preprocessing module for Vehicle Maintenance Risk Prediction.

Handles:
- Loading data
- Removing leakage columns
- Handling missing values (median imputation)
- Feature engineering
- Encoding categorical variables
- Feature scaling (StandardScaler)
- Train-test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Columns that directly encode failure information → data leakage
LEAKAGE_COLUMNS = [
    "Remaining Useful Life",
    "RUL",
    "Time to Failure",
    "TTF",
    "Component Health Score",
    "Component_Health_Score",
]

# Non-feature columns to drop
DROP_COLUMNS = ["Timestamp"]

# Target column
TARGET = "Failure_Probability"


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    df = pd.read_csv(filepath)
    return df


def remove_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that cause data leakage."""
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df


def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-feature columns like Timestamp."""
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features:
    - Battery_Stress = Battery_Current * Battery_Temperature
    - Motor_Load = Motor_Torque * Motor_RPM
    - Wear_Index = Distance_Traveled / Charge_Cycles
    """
    if "Battery_Current" in df.columns and "Battery_Temperature" in df.columns:
        df["Battery_Stress"] = df["Battery_Current"] * df["Battery_Temperature"]

    if "Motor_Torque" in df.columns and "Motor_RPM" in df.columns:
        df["Motor_Load"] = df["Motor_Torque"] * df["Motor_RPM"]

    if "Distance_Traveled" in df.columns and "Charge_Cycles" in df.columns:
        # Avoid division by zero
        df["Wear_Index"] = df["Distance_Traveled"] / df["Charge_Cycles"].replace(0, np.nan)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode any remaining categorical columns (excluding target)."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if TARGET in cat_cols:
        cat_cols.remove(TARGET)
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with median for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


def preprocess_pipeline(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline.

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # 1. Load
    df = load_data(filepath)

    # 2. Remove leakage columns
    df = remove_leakage(df)

    # 3. Drop non-feature columns
    df = drop_non_features(df)

    # 4. Feature engineering
    df = engineer_features(df)

    # 5. Encode categoricals
    df = encode_categoricals(df)

    # 6. Impute missing values
    df = impute_missing(df)

    # 7. Split features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    feature_names = X.columns.tolist()

    # 8. Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 9. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
