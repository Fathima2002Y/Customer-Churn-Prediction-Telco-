import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import features  # noqa: F401  (needed so joblib can unpickle the pipeline)

BASE = Path(__file__).parent

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    pipe = joblib.load(BASE / "model" / "churn_pipeline.joblib")
    meta = json.loads((BASE / "model" / "metadata.json").read_text())
    return pipe, meta


pipe, meta = load_artifacts()
threshold = meta["threshold"]

st.title("📉 Customer Churn Predictor")
st.caption(
    f"Model: {meta['model_name']} | Test ROC-AUC: {meta['test_roc_auc']:.2f} | "
    f"Recall: {meta['test_recall']:.2f} | Decision threshold: {threshold:.2f}"
)

with st.sidebar:
    st.header("Customer details")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has partner", ["Yes", "No"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)

col1, col2, col3 = st.columns(3)
phone = col1.selectbox("Phone service", ["Yes", "No"])
internet = col2.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
multiple = col3.selectbox("Multiple lines", ["No", "Yes"]) if phone == "Yes" else "No phone service"

if internet == "No":
    sec = backup = protect = tech = tv = movies = "No internet service"
    st.info("No internet service selected, so internet add-ons are not applicable.")
else:
    c1, c2, c3 = st.columns(3)
    sec = c1.selectbox("Online security", ["No", "Yes"])
    backup = c2.selectbox("Online backup", ["No", "Yes"])
    protect = c3.selectbox("Device protection", ["No", "Yes"])
    c4, c5, c6 = st.columns(3)
    tech = c4.selectbox("Tech support", ["No", "Yes"])
    tv = c5.selectbox("Streaming TV", ["No", "Yes"])
    movies = c6.selectbox("Streaming movies", ["No", "Yes"])

row = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
    "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
    "MultipleLines": multiple, "InternetService": internet, "OnlineSecurity": sec,
    "OnlineBackup": backup, "DeviceProtection": protect, "TechSupport": tech,
    "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": tenure * monthly,
}])

if st.button("Predict churn risk", type="primary"):
    prob = float(pipe.predict_proba(row)[0, 1])
    will_churn = prob >= threshold

    left, right = st.columns([1, 2])
    left.metric("Churn probability", f"{prob:.1%}")
    if will_churn:
        right.error("⚠️ High churn risk: this customer is likely to leave.")
    else:
        right.success("✅ Low churn risk: this customer is likely to stay.")
    st.progress(min(prob, 1.0))

    if will_churn:
        st.subheader("Suggested retention actions")
        tips = []
        if contract == "Month-to-month":
            tips.append("Offer a discounted 1-year or 2-year contract.")
        if internet != "No" and tech == "No":
            tips.append("Bundle free tech support or online security for a few months.")
        if payment == "Electronic check":
            tips.append("Nudge toward automatic payment with a small incentive.")
        if tenure <= 12:
            tips.append("Run an onboarding / early-life engagement campaign.")
        if internet == "Fiber optic" and monthly > 85:
            tips.append("Review pricing; high fiber bills are a known churn driver.")
        for t in tips or ["Assign a retention agent for a personal check-in call."]:
            st.write("•", t)
