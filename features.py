"""Shared feature engineering, used by BOTH the notebook and the Streamlit app.

Keeping this in one file guarantees the app transforms inputs exactly the
way the model was trained (a very common deployment bug otherwise).
"""
import pandas as pd

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is text in the raw file; blanks belong to brand-new customers (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # How many services does the customer use?
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int)

    # Average spend per month of tenure (falls back to MonthlyCharges for new customers)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].where(df["tenure"] > 0)
    df["avg_monthly_spend"] = df["avg_monthly_spend"].fillna(df["MonthlyCharges"])

    # Protective services reduce churn
    df["has_support"] = ((df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")).astype(int)

    # Automatic payment methods
    df["auto_pay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)

    # Tenure buckets (as plain strings so they one-hot encode)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    ).astype(str)

    return df
