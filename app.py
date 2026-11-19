"""HospitalStay AI — role-based educational Streamlit demonstration.

All roles access the same conservative two-feature Gradient Boosting pipeline.
No real patient identifiers or data should be entered. Roles are UI modes, NOT authentication.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="HospitalStay AI | Role-Based Demo", page_icon="🏥", layout="wide")

st.markdown("""<style>
.block-container {max-width: 1180px; padding-top: 1.5rem; padding-bottom: 3rem;}
h1,h2,h3 {letter-spacing: -.025em;}
div[data-testid="stMetric"] {padding: 0.9rem; border: 1px solid #d8e3ed; border-radius: 12px;}
.smallnote {font-size: .9rem; color: #64748b;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Only load our own trusted local Joblib file. Never load uploaded model files.
    return joblib.load(ROOT / "hospitalstay_model.joblib")

@st.cache_data
def load_categories():
    with (ROOT / "categories.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("rcount", "gender"):
        if not isinstance(data.get(key), list) or not data[key]:
            raise ValueError(f"Missing input options: {key}")
    return data

try:
    model = load_model()
    categories = load_categories()
except Exception:
    st.error("The application cannot load the model or input categories. Check the application files and Python dependencies.")
    st.stop()


def predict(records):
    """Validate categorical inputs, return a list of nonnegative prediction estimates."""
    if not records:
        raise ValueError("At least one demo profile is required")
    frame = pd.DataFrame.from_records(records, columns=["rcount", "gender"])
    if frame.isnull().any().any():
        raise ValueError("Inputs cannot be empty")
    frame = frame.astype(str)
    for col in ("rcount", "gender"):
        if not frame[col].isin(categories[col]).all():
            raise ValueError(f"Unsupported {col} category")
    return [max(0.0, float(value)) for value in model.predict(frame)]


def single_form(key):
    with st.form(key):
        c1, c2 = st.columns(2)
        with c1:
            readmissions = st.selectbox(
                "Readmissions in the previous 180 days",
                categories["rcount"], key=f"{key}_rcount",
                help="Select an existing category from the synthetic source dataset."
            )
        with c2:
            gender = st.selectbox(
                "Recorded gender category", categories["gender"],
                key=f"{key}_gender",
                format_func=lambda x: {"F":"F (dataset category)","M":"M (dataset category)"}.get(x,x)
            )
        submitted = st.form_submit_button("Estimate hospital stay", type="primary", use_container_width=True)
    if submitted:
        try:
            result = predict([{"rcount":readmissions, "gender":gender}])[0]
            st.metric("Model-estimated length of stay", f"{result:.2f} days")
            st.info("This is a statistical estimate from synthetic data. It is not a discharge date, an individualized clinical assessment, or medical advice.")
        except ValueError:
            st.error("Please choose supported input categories.")
        except Exception:
            st.error("Unable to calculate this prediction. Please review the deployed model.")


def scenario_chart(key):
    st.caption("Illustrative profiles only. Changing an input does not establish a causal effect.")
    gender = st.selectbox("Fixed recorded gender category", categories["gender"], key=f"{key}_gender")
    data = [{"rcount":r, "gender":gender} for r in categories["rcount"]]
    values = predict(data)
    table = pd.DataFrame({"Readmissions":categories["rcount"], "Estimated days":values})
    st.bar_chart(table.set_index("Readmissions"))
    st.dataframe(table.style.format({"Estimated days":"{:.2f}"}), use_container_width=True, hide_index=True)


def demo_cohort():
    """Create fictional input profiles, not source data or real patient records."""
    return pd.DataFrame([
        {"Demo ID":f"SIM-{idx:02d}", "rcount":str(r), "gender":g}
        for idx,(r,g) in enumerate([
            ("0","F"),("0","M"),("1","F"),("2","M"),
            ("3","F"),("4","M"),("5+","F"),("5+","M")
        ], start=1)
    ])


def show_cohort(label):
    st.subheader(label)
    st.warning("These are eight fictional demonstration profiles. They are not real patients, hospital admissions, or a validated bed-capacity forecast.")
    cohort = demo_cohort()
    cohort["Estimated stay (days)"] = predict(cohort[["rcount","gender"]].to_dict("records"))
    c1,c2 = st.columns(2)
    c1.metric("Fictional demo profiles", str(len(cohort)))
    c2.metric("Mean estimate across demo profiles", f"{cohort['Estimated stay (days)'].mean():.2f} days")
    st.dataframe(cohort.style.format({"Estimated stay (days)":"{:.2f}"}), hide_index=True, use_container_width=True)
    st.bar_chart(cohort.set_index("Demo ID")["Estimated stay (days)"])
    st.caption("These figures compare modeled outputs; they cannot determine clinical priorities, staffing requirements or actual bed occupancy.")


st.title("🏥 HospitalStay AI")
st.write("**Role-based demonstration of a Machine Learning hospital length-of-stay predictor**")
st.caption("University prototype • Microsoft synthetic dataset • Two-feature Gradient Boosting model")
st.warning("Educational demo only. Do not enter real patient data or use predictions for diagnosis, discharge, or allocation decisions.")

role = st.sidebar.radio(
    "Choose a demonstration view",
    ["Patient", "Doctor", "Nurse", "Hospital administrator"],
    help="This changes the interface only; it does not authenticate or verify a professional role."
)
st.sidebar.info("Role selection is not a login or access-control system. All users can view the same demo data.")
st.sidebar.caption("Model inputs: previous readmissions and recorded gender. No patient records are stored by this app.")

if role == "Patient":
    st.header("Patient view")
    st.write("Explore an example prediction using two simple input categories. The estimate is not a personalized medical prognosis.")
    single_form("patient")
    with st.expander("How should I understand this estimate?"):
        st.write("It is a statistical output from a synthetic-data model. It cannot take account of symptoms, illness severity, treatment response or your care team’s assessment.")
        st.write("For individual questions about hospitalization, speak with your healthcare team.")

elif role == "Doctor":
    st.header("Doctor view — model interpretation")
    st.write("Review the same experimental model with technical context. It uses **only two categorical inputs**; it does not assess medical severity or discharge readiness.")
    single_form("doctor")
    st.subheader("Hypothetical scenario comparison")
    scenario_chart("doctor_compare")
    st.info("Model performance: the deployed two-feature Gradient Boosting model achieved a 5-fold cross-validation MAE of 1.230 days (SD 0.029) on synthetic training data. This is not a clinical confidence interval.")

elif role == "Nurse":
    st.header("Nurse view — demonstration coordination board")
    st.write("Compare modeled length-of-stay estimates for fictional profiles. No care tasks, escalation priorities, or discharge instructions are generated.")
    show_cohort("Simulated patient board")
    with st.expander("Individual example"):
        single_form("nurse")

elif role == "Hospital administrator":
    st.header("Administrator view — research analytics")
    st.write("Explore scenario outputs and the study methodology. These are educational simulations, not validated hospital demand forecasts.")
    show_cohort("Hypothetical portfolio of profiles")
    st.subheader("Readmission scenario comparison")
    scenario_chart("admin_compare")

st.divider()
with st.expander("📈 Model evaluation — view methodology and metrics"):
    st.write("**Important:** the four-model holdout comparison used eight features, but the deployed Streamlit model uses two features.")
    st.dataframe(pd.DataFrame([
        ["Dummy Regressor",1.843,2.292,-0.003],
        ["Linear Regression",1.083,1.322,0.666],
        ["Random Forest",1.069,1.317,0.669],
        ["Gradient Boosting",1.071,1.316,0.669],
    ], columns=["Eight-feature model", "Test MAE (days)","Test RMSE (days)","Test R²"]), hide_index=True, use_container_width=True)
    st.dataframe(pd.DataFrame([
        ["Gradient Boosting — eight features",1.099,0.028],
        ["Random Forest — eight features",1.101,0.026],
        ["Linear Regression — eight features",1.111,0.028],
        ["Gradient Boosting — two features (deployed)",1.230,0.029],
    ], columns=["Model", "5-fold CV MAE (days)","Fold SD"]), hide_index=True, use_container_width=True)
    st.caption("The independent test metrics of the deployed two-feature model should be reported after computing them. Do not quote eight-feature test metrics as performance of the deployed app.")

with st.expander("🗂️ Dataset sampling, fairness and limitations"):
    st.write("The original source has approximately **100,000 synthetic hospital encounters**. We drew **5,000 source rows** using simple random sampling without replacement.")
    st.code("df = df_original.sample(n=5000, random_state=42).reset_index(drop=True)", language="python")
    st.write("The fixed random seed makes the draw reproducible; it does not ensure a perfectly representative sample. Compare full-dataset and sample distributions to assess variation and rare subgroups.")
    st.write("The research models used additional diagnosis flags whose availability at admission was uncertain. The deployed model uses only readmission category and recorded gender; it still lacks real-world clinical validation.")
    st.write("Gender is included because it is a source-dataset variable, not because gender should guide care. A real deployment would require subgroup bias analysis, privacy safeguards, model calibration, prospective validation and appropriate governance.")
    st.markdown("[Original Microsoft dataset](https://github.com/microsoft/r-server-hospital-length-of-stay) · [Project source code](https://github.com/faiakter/hospitalstay-ai)")

st.caption("HospitalStay AI • University educational demonstration • Role modes are not secure professional accounts")
