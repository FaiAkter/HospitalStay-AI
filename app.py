# ============================================================
# HOSPITALSTAY AI - STREAMLIT WEB APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# -------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="HospitalStay AI",
    page_icon="🏥",
    layout="centered"
)

# -------------------------------
# 2. LOAD THE TRAINED MODEL
# -------------------------------

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "hospitalstay_model.joblib"
    return joblib.load(model_path)

try:
    model = load_model()

    with open(Path(__file__).parent / "categories.json") as f:
        categories = json.load(f)

except Exception as e:
    st.error("Unable to load the prediction model.")
    st.exception(e)
    st.stop()

# -------------------------------
# 3. APPLICATION HEADER
# -------------------------------

st.title("🏥 HospitalStay AI")

st.subheader("Hospital Length of Stay Prediction")

st.write(
    "An interactive Machine Learning application designed "
    "to estimate hospital length of stay using patient information."
)

st.divider()

# -------------------------------
# 4. PATIENT INFORMATION
# -------------------------------

st.header("Patient Information")

st.write(
    "Enter the patient's information below to generate "
    "an estimated hospital stay duration."
)

# Display input fields inside a form.
with st.form("patient_form"):

    rcount = st.selectbox(
        "Previous readmissions (last 180 days)",
        options=categories["rcount"],
        help="Number of previous readmissions within 180 days."
    )

    gender = st.selectbox(
        "Patient gender",
        options=categories["gender"]
    )

    submitted = st.form_submit_button(
        "Predict Length of Stay",
        use_container_width=True
    )

# -------------------------------
# 5. GENERATE PREDICTION
# -------------------------------

if submitted:

    # Create an input DataFrame matching the training features.
    patient_data = pd.DataFrame([{
        "rcount": str(rcount),
        "gender": str(gender)
    }])

    try:
        prediction = float(
            model.predict(patient_data)[0]
        )

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            label="Estimated Hospital Stay",
            value=f"{prediction:.2f} days"
        )

        st.success("Prediction generated successfully!")

        st.info(
            "This estimate is based on patterns learned from "
            "synthetic hospital data. It does not represent "
            "a clinically validated prediction."
        )

    except Exception as e:
        st.error("The prediction could not be generated.")
        st.exception(e)

# -------------------------------
# 6. ABOUT THE APPLICATION
# -------------------------------

st.divider()

with st.expander("About HospitalStay AI"):

    st.write(
        "HospitalStay AI uses a Gradient Boosting Regressor "
        "trained on synthetic hospital admission records."
    )

    st.write(
        "The current version uses two input variables: "
        "previous readmissions and gender."
    )

    st.write(
        "The application demonstrates how Machine Learning "
        "can be applied to hospital resource planning."
    )

st.caption(
    "Educational university project | "
    "Not intended for clinical decision-making."
)
