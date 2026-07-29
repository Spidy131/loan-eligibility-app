import streamlit as st
import numpy as np
import joblib

# ---------------- LOAD MODEL ----------------
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

# User enters amount in Rupees
loan_amount = st.number_input("Loan Amount (₹)", min_value=0)

# Convert to thousands because the model was trained that way
loan_amount_thousands = loan_amount / 1000

loan_term = st.number_input("Loan Term (months)", value=360)
interest_rate = st.number_input(
    "Annual Interest Rate (%)",
    min_value=1.0,
    max_value=25.0,
    value=9.5,
    step=0.1
)

credit_history = st.selectbox("Credit History", ["Good", "Bad"])


# ---------------- RULE BASED CHECK ----------------
def rule_based_check(income, co_income, loan_amt_rupees, credit, term):

    total_income = income + co_income

    min_income = 15000
    max_loan_multiple = 20
    emi_ratio = 0.40

    max_loan_allowed = total_income * max_loan_multiple
    estimated_emi = loan_amt_rupees / term
    max_emi_allowed = total_income * emi_ratio

    if credit == 0:
        return False, "Bad Credit History"

    if total_income < min_income:
        return False, "Income below minimum requirement"

    if loan_amt_rupees > max_loan_allowed:
        return False, "Loan amount too high compared to income"

    if estimated_emi > max_emi_allowed:
        return False, "Estimated EMI exceeds 40% of monthly income"

    return True, "Rule-based eligibility passed"


# ---------------- ENCODING ----------------
def encode_inputs():

    return [[
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        int(dependents.replace("+", "")),
        0 if education == "Graduate" else 1,
        1 if self_employed == "Yes" else 0,
        applicant_income,
        coapplicant_income,
        loan_amount_thousands,
        loan_term,
        1 if credit_history == "Good" else 0,
        {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]
    ]]
#---------------------EMI FUNCTION-----------------
def calculate_emi(principal, annual_rate, months):

    monthly_rate = annual_rate / (12 * 100)

    emi = (
        principal
        * monthly_rate
        * (1 + monthly_rate) ** months
    ) / (
        (1 + monthly_rate) ** months - 1
    )

    return emi

# ---------------- PREDICTION ----------------
if st.button("Check Eligibility"):

    credit_val = 1 if credit_history == "Good" else 0

    eligible, reason = rule_based_check(
        applicant_income,
        coapplicant_income,
        loan_amount,
        credit_val,
        loan_term,
    )

    if eligible:
        st.success(f"Rule Check: {reason}")
    else:
        st.warning(f"Rule Check: {reason}")

    input_data = np.array(encode_inputs())

    st.write("Encoded Input:", input_data)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.write("Prediction Probability:", probability)

    if prediction[0] == 1:
        emi = calculate_emi(
            loan_amount,
            interest_rate,
            loan_term
        )
    
        total_payment = emi * loan_term
        total_interest = total_payment - loan_amount
    
        st.success(
            f"✅ Loan Approved ({probability[0][1]*100:.2f}% confidence)"
        )
    
        st.metric("Monthly EMI", f"₹{emi:,.2f}")
        st.metric("Total Interest", f"₹{total_interest:,.2f}")
        st.metric("Total Payment", f"₹{total_payment:,.2f}")
    else:
        st.error(
            f"❌ Loan Rejected ({probability[0][0]*100:.2f}% confidence)"
        )
