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


def nurse_board():
    """An editable in-session board for fictional profiles only."""
    st.subheader("Fictional ward board")
    st.caption("Create, edit and delete demonstration profiles. Data are session-only, not clinical records.")
    if "ward_profiles" not in st.session_state:
        st.session_state.ward_profiles = [
            {"Demo ID": "SIM-001", "rcount": "0", "gender": "M"},
            {"Demo ID": "SIM-002", "rcount": "2", "gender": "F"},
        ]
    with st.form("add_ward_record", clear_on_submit=True):
        st.markdown("**Add a fictional patient profile**")
        a,b,c=st.columns(3)
        new_id=a.text_input("Fictional ID", placeholder="SIM-003", max_chars=25)
        new_r=b.selectbox("Readmissions in 180 days", categories["rcount"])
        new_g=c.selectbox("Recorded gender category", categories["gender"])
        add=st.form_submit_button("Add to board",type="primary")
    if add:
        new_id=new_id.strip()
        if not new_id or any(x["Demo ID"]==new_id for x in st.session_state.ward_profiles):
            st.error("Enter a nonempty, unique fictional ID.")
        elif len(st.session_state.ward_profiles)>=100:
            st.error("Demo limit of 100 profiles reached.")
        else:
            st.session_state.ward_profiles.append({"Demo ID":new_id,"rcount":new_r,"gender":new_g})
            st.success("Fictional profile added.")
    if st.session_state.ward_profiles:
        current=pd.DataFrame(st.session_state.ward_profiles)
        st.markdown("**Edit profiles directly in the table below**")
        edited=st.data_editor(current,hide_index=True,use_container_width=True,num_rows="fixed",
            column_config={
                "Demo ID": st.column_config.TextColumn("Fictional ID",required=True),
                "rcount": st.column_config.SelectboxColumn("Readmissions",options=categories["rcount"],required=True),
                "gender": st.column_config.SelectboxColumn("Gender",options=categories["gender"],required=True),
            },key="ward_editor")
        if st.button("Save edited profiles"):
            entries=edited.to_dict("records")
            ids=[str(x["Demo ID"]).strip() for x in entries]
            if not all(ids) or len(set(ids))!=len(ids):
                st.error("Every fictional ID must be nonempty and unique.")
            else:
                try:
                    predict(entries)
                    st.session_state.ward_profiles=[{"Demo ID":ids[i],"rcount":str(x["rcount"]),"gender":str(x["gender"])} for i,x in enumerate(entries)]
                    st.success("Changes saved. Estimates recalculated below.")
                except Exception:
                    st.error("Invalid entry: choose supported input categories.")
        table=pd.DataFrame(st.session_state.ward_profiles)
        table["Estimated stay (days)"]=predict(table[["rcount","gender"]].to_dict("records"))
        c1,c2,c3=st.columns(3)
        c1.metric("Demo profiles",len(table))
        c2.metric("Mean estimated stay",f"{table['Estimated stay (days)'].mean():.2f} days")
        c3.metric("Sum of estimated bed-days",f"{table['Estimated stay (days)'].sum():.1f}")
        st.dataframe(table.style.format({"Estimated stay (days)":"{:.2f}"}),hide_index=True,use_container_width=True)
        st.bar_chart(table.set_index("Demo ID")["Estimated stay (days)"])
        to_remove=st.selectbox("Remove fictional profile",table["Demo ID"].tolist())
        if st.button("Remove selected profile"):
            st.session_state.ward_profiles=[x for x in st.session_state.ward_profiles if x["Demo ID"]!=to_remove]
            st.rerun()
        csv=table.to_csv(index=False).encode("utf-8")
        st.download_button("Download simulated ward table (CSV)",csv,"hospitalstay_demo_ward.csv","text/csv")
    else:
        st.info("Your demonstration board is empty. Add a fictional profile above.")
    st.warning("These estimates cannot assess patient needs, staffing requirements, discharge readiness, or real bed capacity. Do not enter real patient data.")


def doctor_workspace():
    """In-session fictional case register and model interpretation for the Doctor view."""
    st.subheader("Case workspace — hypothetical patients")
    st.caption("Create and compare fictional cases. This is not an electronic health record or a tool for clinical decisions.")
    state_key = "doctor_cases"
    if state_key not in st.session_state:
        st.session_state[state_key] = [
            {"Case ID": "CASE-001", "rcount": "0", "gender": "M"},
            {"Case ID": "CASE-002", "rcount": "2", "gender": "F"},
        ]

    with st.form("doctor_add_case", clear_on_submit=True):
        st.markdown("**Create a fictional case**")
        col1, col2, col3 = st.columns(3)
        case_id = col1.text_input("Fictional case ID", placeholder="CASE-003", max_chars=25)
        readmissions = col2.selectbox("Previous readmissions", categories["rcount"], key="doc_add_r")
        gender = col3.selectbox("Recorded gender category", categories["gender"], key="doc_add_g")
        add_case = st.form_submit_button("Add case", type="primary")
    if add_case:
        case_id = case_id.strip()
        if not case_id or any(case_id == x["Case ID"] for x in st.session_state[state_key]):
            st.error("Use a nonempty, unique fictional case ID.")
        elif len(st.session_state[state_key]) >= 100:
            st.error("Demo limit of 100 fictional cases reached.")
        else:
            st.session_state[state_key].append({"Case ID":case_id, "rcount":readmissions, "gender":gender})
            st.success("Fictional case added.")
            st.rerun()

    if not st.session_state[state_key]:
        st.info("No cases yet. Add a fictional case to begin.")
        return

    # Editing is explicit; unsaved data editor values are not used for predictions.
    current = pd.DataFrame(st.session_state[state_key])
    st.markdown("**Edit cases**")
    edited = st.data_editor(
        current, hide_index=True, use_container_width=True, num_rows="fixed", key="doctor_case_editor",
        column_config={
            "Case ID": st.column_config.TextColumn("Fictional case ID", required=True),
            "rcount": st.column_config.SelectboxColumn("Previous readmissions", options=categories["rcount"], required=True),
            "gender": st.column_config.SelectboxColumn("Recorded gender", options=categories["gender"], required=True),
        },
    )
    if st.button("Save case edits", key="save_doctor_edits"):
        entries = edited.to_dict("records")
        ids = [str(v["Case ID"]).strip() for v in entries]
        if not all(ids) or len(set(ids)) != len(ids):
            st.error("All case IDs must be nonempty and unique.")
        else:
            try:
                cleaned = [
                    {"Case ID":ids[i], "rcount":str(v["rcount"]), "gender":str(v["gender"])}
                    for i, v in enumerate(entries)
                ]
                predict(cleaned)
                st.session_state[state_key] = cleaned
                st.success("Case edits saved.")
                st.rerun()
            except (ValueError, TypeError):
                st.error("Use only supported readmission and gender categories.")
            except Exception:
                st.error("The model could not validate the modified cases.")

    records = pd.DataFrame(st.session_state[state_key])
    try:
        records["Estimated stay (days)"] = predict(records[["rcount", "gender"]].to_dict("records"))
    except Exception:
        st.error("Unable to predict the fictional cases. Check the saved model.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Fictional cases", len(records))
    c2.metric("Mean predicted stay", f"{records['Estimated stay (days)'].mean():.2f} days")
    c3.metric("5-fold CV MAE", "1.230 days", help="Training-data cross-validation mean MAE for the two-feature model; not a case-specific confidence interval")
    st.dataframe(records.style.format({"Estimated stay (days)":"{:.2f}"}), use_container_width=True, hide_index=True)

    st.subheader("Compare two cases")
    if len(records) >= 2:
        a, b = st.columns(2)
        ids = records["Case ID"].tolist()
        first = a.selectbox("First case", ids, index=0, key="doc_comp_first")
        second = b.selectbox("Second case", ids, index=1, key="doc_comp_second")
        if first == second:
            st.info("Choose two different cases to compare their model estimates.")
        else:
            comparison = records[records["Case ID"].isin([first, second])].copy()
            st.bar_chart(comparison.set_index("Case ID")["Estimated stay (days)"])
            comparison = comparison.set_index("Case ID")
            delta = float(comparison.loc[first, "Estimated stay (days)"] - comparison.loc[second, "Estimated stay (days)"])
            st.write(f"Difference in model estimates: **{abs(delta):.2f} days**.")
            st.caption("Differences are model associations, not evidence that any input causes a longer or shorter stay.")
    else:
        st.info("Add at least two fictional cases to compare them.")

    st.subheader("Manage cases")
    left, right = st.columns([2, 1])
    remove_id = left.selectbox("Select case to remove", records["Case ID"].tolist(), key="doc_remove")
    if right.button("Remove selected case", key="doc_delete"):
        st.session_state[state_key] = [x for x in st.session_state[state_key] if x["Case ID"] != remove_id]
        st.rerun()
    st.download_button(
        "Download fictional case results (CSV)",
        records.to_csv(index=False).encode("utf-8"),
        file_name="hospitalstay_doctor_demo_cases.csv", mime="text/csv", key="doc_export"
    )

    with st.expander("Model interpretation and evidence"):
        st.markdown("**Deployed model:** Gradient Boosting with two input categories (`rcount`, `gender`).")
        st.write("Its 5-fold cross-validation mean MAE on synthetic training data was **1.230 days** (SD 0.029). This is a dataset-level error estimate, not a personal uncertainty interval.")
        st.write("The eight-feature research models performed better, but used clinical-condition flags whose availability at admission could not be established. Their test metrics must not be presented as those of this deployed model.")
        st.write("Only readmission history and recorded gender are modeled; severity, treatments, symptoms and clinical status are not measured.")
    st.warning("All cases must be fictional. Do not enter real identifiers or use these outputs for medical decisions or discharge planning.")


def administrator_planner():
    """Aggregate hypothetical admissions by category without patient data."""
    st.subheader("Configurable admission scenario")
    st.write("Set counts of **fictional arrivals** for each input category. The model estimates length of stay for each category, then totals the modeled bed-days. This is not an occupancy forecast.")
    scenario=[]
    with st.form("admin_scenario"):
        st.markdown("**Number of hypothetical arrivals by readmission category and gender**")
        for r in categories["rcount"]:
            c1,c2=st.columns(2)
            for col,g in [(c1,"F"),(c2,"M")]:
                count=col.number_input(f"Readmissions {r}, category {g}",min_value=0,max_value=1000,value=(3 if r=="0" else 0),step=1,key=f"admin_{r}_{g}")
                scenario.append({"rcount":r,"gender":g,"count":int(count)})
        submit=st.form_submit_button("Calculate simulated planning estimates",type="primary")
    if submit:
        st.session_state.admin_scenario_results=scenario
    if "admin_scenario_results" in st.session_state:
        table=pd.DataFrame(st.session_state.admin_scenario_results)
        estimates=predict(table[["rcount","gender"]].to_dict("records"))
        table["Estimated stay (days)"]=estimates
        table["Estimated bed-days"]=table["count"]*table["Estimated stay (days)"]
        table=table.loc[table["count"]>0].copy()
        if table.empty:
            st.info("Add at least one hypothetical arrival to calculate results.")
        else:
            n=int(table["count"].sum())
            total=float(table["Estimated bed-days"].sum())
            c1,c2,c3=st.columns(3)
            c1.metric("Fictional arrivals",n)
            c2.metric("Sum of model-estimated bed-days",f"{total:.1f}")
            c3.metric("Mean estimated stay",f"{total/n:.2f} days")
            st.dataframe(table.style.format({"Estimated stay (days)":"{:.2f}","Estimated bed-days":"{:.2f}"}),hide_index=True,use_container_width=True)
            chart=table.groupby("rcount",sort=False)["Estimated bed-days"].sum().rename("Modeled bed-days")
            st.bar_chart(chart)
            st.download_button("Download scenario results (CSV)",table.to_csv(index=False).encode("utf-8"),"hospitalstay_demo_scenario.csv","text/csv")
    st.warning("Summed model estimates are not a validated prediction of hospital occupancy, staffing, or actual bed availability. No real admissions are used.")


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
    st.header("Doctor view — case comparison and interpretation")
    st.write("A fictional case workspace for comparing predictions and reviewing technical model evidence. This does not assess clinical severity or discharge readiness.")
    doctor_workspace()
    with st.expander("Quick individual prediction"):
        single_form("doctor")
    with st.expander("Compare readmission categories"):
        scenario_chart("doctor_compare")

elif role == "Nurse":
    st.header("Nurse view — demonstration coordination board")
    st.write("Build an editable board of fictional profiles, compare modeled stays, and export simulation results. No care tasks or discharge instructions are generated.")
    nurse_board()
    with st.expander("Individual example"):
        single_form("nurse")

elif role == "Hospital administrator":
    st.header("Administrator view — research analytics")
    st.write("Create hypothetical admission mixes and compare aggregate model estimates. These are educational simulations, not validated hospital demand forecasts.")
    administrator_planner()
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
