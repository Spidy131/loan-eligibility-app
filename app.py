import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Loan Eligibility Checker")

st.title("🏦 Loan Eligibility Prediction System")
st.write("Enter applicant details to check loan eligibility")

# ---------------- INPUTS ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

applicant_income = st.number_input("Applicant Income (Monthly ₹)", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income (Monthly ₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands ₹)", min_value=0)
loan_term = st.number_input("Loan Term (months)", value=360)
credit_history = st.selectbox("Credit History", ["Good", "Bad"])

# ---------------- RULE BASED CHECK ----------------
def rule_based_check(income, loan_amt, credit):
    total_income = income + coapplicant_income

    # Basic banking rules
    min_income = 15000
    max_loan_multiple = 20
    emi_ratio = 0.4

    loan_rupees = loan_amt * 1000
    max_loan_allowed = total_income * max_loan_multiple
    estimated_emi = loan_rupees / loan_term
    max_emi_allowed = total_income * emi_ratio

    if credit == 0:
        return False, "Bad credit history"

    if total_income < min_income:
        return False, "Income below minimum requirement"

    if loan_rupees > max_loan_allowed:
        return False, "Loan amount too high compared to income"

    if estimated_emi > max_emi_allowed:
        return False, "EMI exceeds 40% of income"

    return True, "Rule-based eligibility passed"

# ---------------- ENCODING ----------------
def encode_inputs():
    return [
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        int(dependents.replace("+","")),
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        1 if credit_history == "Good" else 0,
        {"Urban":2,"Semiurban":1,"Rural":0}[property_area]
    ]

# ---------------- PREDICTION ----------------
if st.button("Check Eligibility"):

    credit_val = 1 if credit_history == "Good" else 0
    is_eligible, reason = rule_based_check(applicant_income, loan_amount, credit_val)

    if not is_eligible:
        st.error(f"❌ Loan Rejected (Rule-Based): {reason}")

    else:
        input_data = np.array(encode_inputs()).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.success("✅ Loan Approved (ML Decision)")
        else:
            st.error("❌ Loan Rejected (ML Decision)")
