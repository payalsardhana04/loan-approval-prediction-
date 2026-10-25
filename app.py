import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦"
)

st.title("🏦 Loan Approval Prediction System")
st.write("Enter the applicant details to predict loan status.")

model = joblib.load("loan_approval_model (5).pkl")
feature_columns = joblib.load("loan_feature_columns.pkl")

with st.form("loan_form"):
    loan_id = st.number_input("Loan ID", min_value=1, value=5000)
    dependents = st.selectbox(
        "Number of Dependents",
        ["0", "1", "2", "3+"]
    )
    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )
    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"]
    )
    income = st.number_input("Annual Income", min_value=0, value=500000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)
    loan_term = st.number_input("Loan Term", min_value=1, value=12)
    cibil = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
    residential = st.number_input("Residential Assets Value", min_value=0, value=100000)
    commercial = st.number_input("Commercial Assets Value", min_value=0, value=0)
    luxury = st.number_input("Luxury Assets Value", min_value=0, value=0)
    bank_assets = st.number_input("Bank Asset Value", min_value=0, value=100000)

    submitted = st.form_submit_button("Predict Loan Status")

if submitted:
    applicant = pd.DataFrame([{
        "loan_id": loan_id,
        "no_of_dependents": dependents,
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil,
        "residential_assets_value": residential,
        "commercial_assets_value": commercial,
        "luxury_assets_value": luxury,
        "bank_asset_value": bank_assets,
        "education": education,
        "self_employed": self_employed
    }])

    applicant["no_of_dependents"] = (
        applicant["no_of_dependents"].replace("3+", "3").astype(int)
    )

    applicant = pd.get_dummies(
        applicant, columns=["education", "self_employed"],
        drop_first=True, dtype=int
    )

    applicant = applicant.reindex(
        columns=feature_columns, fill_value=0
    )

    prediction = model.predict(applicant)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Loan Status: {prediction}")
    st.caption(
        "This is a machine-learning prediction, not an actual bank approval."
    )
