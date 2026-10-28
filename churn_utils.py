"""Shared constants and helper functions for the Telco churn project."""
import pandas as pd

TARGET_COL = "Churn"
ID_COL = "customerID"
POSITIVE_LABEL = "Yes"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the identifier column and convert TotalCharges to numeric (blank strings become NaN)."""
    cleaned = df.copy()
    cleaned.columns = [name.strip() for name in cleaned.columns]
    if ID_COL in cleaned.columns:
        cleaned = cleaned.drop(columns=[ID_COL])
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    return cleaned


def encode_target(labels: pd.Series) -> pd.Series:
    """Convert Yes/No churn labels (or 1/0) into integers, with churn = 1."""
    text = labels.astype(str).str.strip()
    return text.isin([POSITIVE_LABEL, "1", "True", "true"]).astype(int)
