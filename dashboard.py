"""Step 7: Streamlit dashboard.

Doctor view:
  - Upload a CSV/XLSX of many patients' biomarkers.
  - Every patient is run through the full pipeline (critical check -> model ->
    clinical context -> LLM narrative) as soon as they're uploaded.
  - A second tab lists all processed patients by Patient ID so the doctor can
    review each one, edit the plain-English summary if needed, and either:
      Approve            -> sent to the patient's view
      Manual Attention   -> patient sees "on hold", doctor deals with it later

Patient view:
  - Logs in with just their Patient ID.
  - Sees their approved report, or a holding message if it's still pending /
    on hold.

Storage is a single JSON file (outputs/dashboard_store.json) — deliberately
simple, no real database. Login is a hardcoded doctor username/password dict
— fine for a dissertation demo, NOT meant for real clinical deployment.

Run with: streamlit run dashboard.py   (from the repo root)
"""
import json
import os
import datetime as dt

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.pipeline.patient_upload import load_all_patients_from_upload, PatientDataError
from src.pipeline.inference import BloodTestPredictor
from src.pipeline.generate_report_layout import compute_key_findings
from src.main import run_pipeline, approve_and_generate_patient_report


def biomarker_dataframe(biomarker_table):
    """Turns the biomarker_table list-of-dicts into a display-ready DataFrame
    with one combined 'Value' column (value + unit) — avoids the fixed-width
    text alignment breaking when rendered in a browser."""
    rows = [{
        "Biomarker": b["biomarker"],
        "Value": f"{b['value']} {b['unit']}",
        "Reference Range": b["reference_range"],
        "Status": b["flag"],
    } for b in biomarker_table]
    return pd.DataFrame(rows)


def render_report(patient_id, dob, test_date, context, narrative, doctor_view,
                   reviewing_doctor=None, approval_date=None):
    """Renders a report (doctor or patient version) using proper Streamlit
    widgets instead of one big pre-formatted text block, so columns actually
    align and abnormal values stand out."""
    st.write(f"**Patient ref:** {patient_id}  |  **DOB:** {dob}  |  **Test date:** {test_date}")

    st.subheader("1. Results Summary")
    df = biomarker_dataframe(context["biomarker_table"])

    def _highlight_status(row):
        color = "background-color: #ffe3e3" if row["Status"] != "Normal" else ""
        return [color] * len(row)

    st.dataframe(df.style.apply(_highlight_status, axis=1), use_container_width=True, hide_index=True)

    risk = context["risk_assessment"]
    anomaly = context["anomaly_assessment"]

    if doctor_view:
        st.subheader("2. Risk Assessment")
        st.write(f"**Diabetes Risk Screening Result:** {risk['label']} (confidence {risk['probability']:.1%})")
        st.write(f"**Overall Biomarker Deviation Score:** {anomaly['score']:.3f} — {anomaly['interpretation']}")

        key_findings = compute_key_findings(context["biomarker_table"])
        st.subheader("3. Key Contributing Factors")
        if key_findings:
            for f in key_findings:
                st.write(f"- {f['biomarker']}: {f['pct_deviation']:.0f}% {f['direction']} of reference range")
        else:
            st.write("No biomarkers outside reference range.")

        st.subheader("4. Model Transparency Note (Clinician Reference Only)")
        for d in context.get("top_shap_drivers", [])[:5]:
            sign = "+" if d["shap_value"] > 0 else ""
            st.write(f"- {d['biomarker']}: {sign}{d['shap_value']} ({d['direction']}, status: {d['flag']})")
        st.caption("SHAP values reflect the AI model's computed contribution to this prediction. "
                   "They do NOT represent medical causation.")

        st.subheader("5. Summary (plain-English text sent to the patient if approved)")
        st.write(narrative)
    else:
        st.subheader("2. What Your Results Show")
        plain_label = ("Your results suggest a notably higher than average chance of blood sugar problems."
                        if risk["label"] == "AtRisk" and risk["probability"] > 0.8 else
                        "Your results suggest a somewhat higher than average chance of blood sugar problems."
                        if risk["label"] == "AtRisk" else
                        "Your results do not currently suggest an increased risk of blood sugar problems.")
        st.write(plain_label)

        key_findings = compute_key_findings(context["biomarker_table"])
        st.subheader("3. Results That Stand Out")
        if key_findings:
            st.write("These results were the main ones outside the usual healthy range:")
            for f in key_findings:
                direction_word = "higher" if f["direction"] == "high" else "lower"
                st.write(f"- {f['biomarker']}: {direction_word} than the healthy range")
        else:
            st.write("All your results were within the healthy range.")

        st.subheader("4. What This Means For You")
        st.write(narrative)

        st.subheader("5. Doctor Review")
        st.write(f"Reviewed and approved by: {reviewing_doctor or '[Doctor name]'}")
        st.write(f"Approval date: {approval_date or '[Date]'}")
        st.caption("This report has been reviewed and approved by your doctor before being sent to you. "
                   "If you have any questions, please contact your GP surgery.")


# ---------------------------------------------------------------------------
# Config — simple and hardcoded on purpose
# ---------------------------------------------------------------------------
DOCTOR_CREDENTIALS = {
    "doctor": "changeme123",   # <-- change this before showing the demo to anyone
}
STORE_PATH = "outputs/dashboard_store.json"


# ---------------------------------------------------------------------------
# Tiny JSON "database" helpers
# ---------------------------------------------------------------------------
def load_store():
    if not os.path.exists(STORE_PATH):
        return {}
    with open(STORE_PATH, "r") as f:
        return json.load(f)


def save_store(store):
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    with open(STORE_PATH, "w") as f:
        json.dump(store, f, indent=2, default=str)


@st.cache_resource
def get_predictor():
    return BloodTestPredictor()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "role" not in st.session_state:
    st.session_state.role = None
if "doctor_name" not in st.session_state:
    st.session_state.doctor_name = None
if "patient_id" not in st.session_state:
    st.session_state.patient_id = None

st.set_page_config(page_title="Blood Test Dashboard", layout="wide")

# Hide the "running..." spinner that Streamlit shows next to the Deploy button
# while a script executes — purely cosmetic, has no effect on functionality.
st.markdown(
    "<style>[data-testid='stStatusWidget'] {visibility: hidden;}</style>",
    unsafe_allow_html=True,
)


def logout():
    st.session_state.role = None
    st.session_state.doctor_name = None
    st.session_state.patient_id = None


# ---------------------------------------------------------------------------
# LOGIN SCREEN
# ---------------------------------------------------------------------------
if st.session_state.role is None:
    st.title("Blood Test Dashboard")
    choice = st.radio("I am logging in as a:", ["Doctor", "Patient"], horizontal=True)

    if choice == "Doctor":
        with st.form("doctor_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            if DOCTOR_CREDENTIALS.get(username) == password:
                st.session_state.role = "doctor"
                st.session_state.doctor_name = username
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    else:  # Patient
        with st.form("patient_login"):
            patient_id = st.text_input("Patient ID")
            submitted = st.form_submit_button("Log in")
        if submitted:
            if patient_id.strip():
                st.session_state.role = "patient"
                st.session_state.patient_id = patient_id.strip()
                st.rerun()
            else:
                st.error("Please enter your Patient ID.")

    st.stop()


# ---------------------------------------------------------------------------
# DOCTOR VIEW
# ---------------------------------------------------------------------------
if st.session_state.role == "doctor":
    st.sidebar.write(f"Logged in as **{st.session_state.doctor_name}**")
    if st.sidebar.button("Log out"):
        logout()
        st.rerun()

    st.title("Doctor Dashboard")
    tab_upload, tab_review = st.tabs(["Upload Patients", "Review & Approve"])

    # --- Upload tab --------------------------------------------------------
    with tab_upload:
        st.subheader("Upload a biomarker file")
        st.caption("CSV or Excel file with one row per patient. Every patient is "
                   "processed automatically as soon as you upload.")

        uploaded_file = st.file_uploader("Choose file", type=["csv", "xlsx", "xls"])
        skip_existing = st.checkbox(
            "Skip patients already in the system (recommended)", value=True,
            help="If unchecked, re-uploading a file with a Patient ID that's already been "
                 "reviewed/approved will OVERWRITE that record, including any approval.",
        )

        if uploaded_file is not None and st.button("Process file", type="primary"):
            try:
                patients = load_all_patients_from_upload(uploaded_file, filename_hint=uploaded_file.name)
            except PatientDataError as e:
                st.error(f"Could not read file: {e}")
                patients = []

            if patients:
                predictor = get_predictor()
                store = load_store()
                progress = st.progress(0, text="Processing patients...")

                failures = []
                skipped = []
                for i, patient in enumerate(patients):
                    pid = str(patient.get("PatientID", f"UNKNOWN-{i}"))

                    if skip_existing and pid in store and store[pid]["status"] != "processing_failed":
                        skipped.append(pid)
                        progress.progress((i + 1) / len(patients), text=f"Processed {i + 1}/{len(patients)}")
                        continue

                    try:
                        result = run_pipeline(patient, predictor=predictor, patient_id=pid)
                    except Exception as e:
                        # One patient failing (e.g. Gemini quota hit) should NOT
                        # crash the whole batch — record it and keep going.
                        store[pid] = {
                            "status": "processing_failed",
                            "error": str(e),
                            "patient_data": patient,  # kept so "Retry" can re-run just this one
                            "uploaded_at": dt.datetime.now().isoformat(),
                        }
                        failures.append(pid)
                        progress.progress((i + 1) / len(patients), text=f"Processed {i + 1}/{len(patients)}")
                        continue

                    store[pid] = {
                        "status": "pending_review",
                        "dob": result["dob"],
                        "test_date": result["test_date"],
                        "context": result["context"],
                        "narrative": result["narrative"],
                        "doctor_report": result["doctor_report"],
                        "critical_alerts": result["critical_alerts"],
                        "uploaded_at": dt.datetime.now().isoformat(),
                    }
                    progress.progress((i + 1) / len(patients), text=f"Processed {i + 1}/{len(patients)}")

                save_store(store)
                processed_count = len(patients) - len(skipped)
                if skipped:
                    st.info(f"Skipped {len(skipped)} patient(s) already in the system (unchanged): "
                            f"{', '.join(skipped)}")
                if failures:
                    st.warning(
                        f"Processed {processed_count} patient(s), but {len(failures)} failed "
                        f"(often a Gemini API quota limit — the free tier allows 20 requests/day): "
                        f"{', '.join(failures)}. They're marked 'Processing Failed' in the "
                        f"'Review & Approve' tab — retry them individually once quota resets."
                    )
                elif processed_count:
                    st.success(f"Processed {processed_count} patient(s). Go to the 'Review & Approve' tab.")

    # --- Review tab ----------------------------------------------------------
    with tab_review:
        store = load_store()
        if not store:
            st.info("No patients uploaded yet.")
        else:
            status_icon = {
                "pending_review": "🟡 Pending review",
                "approved": "✅ Approved",
                "manual_attention": "🟠 Manual attention (on hold)",
                "processing_failed": "⚫ Processing failed",
            }

            def label_for(p):
                e = store[p]
                base = status_icon.get(e["status"], e["status"])
                if e.get("critical_alerts"):
                    base = f"🔴 {base} (critical value flagged)"
                return f"{p} — {base}"

            pids = sorted(store.keys())
            selected = st.selectbox("Select a patient", pids, format_func=label_for)
            entry = store[selected]

            st.markdown(f"### Patient {selected}")
            st.markdown(f"**Status:** {label_for(selected).split(' — ', 1)[1]}")

            if entry["status"] == "processing_failed":
                st.error(f"This patient failed to process: {entry['error']}")
                st.caption("This is usually the Gemini API's free-tier daily quota (20 requests/day) "
                           "being exceeded, not a bug — wait for it to reset (~24h) or try again later.")
                if st.button("🔁 Retry this patient"):
                    predictor = get_predictor()
                    try:
                        result = run_pipeline(
                            entry["patient_data"], predictor=predictor, patient_id=selected
                        )
                        store[selected] = {
                            "status": "pending_review",
                            "dob": result["dob"],
                            "test_date": result["test_date"],
                            "context": result["context"],
                            "narrative": result["narrative"],
                            "doctor_report": result["doctor_report"],
                            "critical_alerts": result["critical_alerts"],
                            "uploaded_at": dt.datetime.now().isoformat(),
                        }
                        save_store(store)
                        st.success("Retried successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Retry failed again: {e}")

            else:
                render_report(selected, entry["dob"], entry["test_date"],
                               entry["context"], entry["narrative"], doctor_view=True)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve — send to patient", type="primary"):
                        pipeline_result = {
                            "status": "PROCESSED",
                            "patient_id": selected,
                            "dob": entry["dob"],
                            "test_date": entry["test_date"],
                            "context": entry["context"],
                            "narrative": entry["narrative"],
                        }
                        patient_report = approve_and_generate_patient_report(
                            pipeline_result,
                            reviewing_doctor=st.session_state.doctor_name,
                        )
                        entry["status"] = "approved"
                        entry["patient_report"] = patient_report
                        entry["reviewing_doctor"] = st.session_state.doctor_name
                        entry["approval_date"] = dt.date.today().isoformat()
                        save_store(store)
                        st.success("Approved and sent to patient.")
                        st.rerun()

                with col2:
                    if entry["status"] != "manual_attention":
                        if st.button("🟠 Needs Manual Attention"):
                            entry["status"] = "manual_attention"
                            save_store(store)
                            st.rerun()
                    else:
                        st.caption("Currently on hold for the patient.")


# ---------------------------------------------------------------------------
# PATIENT VIEW
# ---------------------------------------------------------------------------
elif st.session_state.role == "patient":
    st.sidebar.write(f"Logged in as Patient **{st.session_state.patient_id}**")
    if st.sidebar.button("Log out"):
        logout()
        st.rerun()

    st.title("Your Blood Test Results")

    store = load_store()
    entry = store.get(st.session_state.patient_id)

    if entry is None:
        st.error("No results found for this Patient ID. Please check with your GP surgery.")
    elif entry["status"] == "approved":
        render_report(st.session_state.patient_id, entry["dob"], entry["test_date"],
                       entry["context"], entry["narrative"], doctor_view=False,
                       reviewing_doctor=entry.get("reviewing_doctor"),
                       approval_date=entry.get("approval_date"))
    elif entry["status"] == "manual_attention":
        st.warning("Your results need a closer look from your doctor before we can share them. "
                   "Your GP surgery will be in touch — no action needed from you right now.")
    else:  # pending_review (including patients with a flagged critical value)
        st.info("Your results are still being reviewed by your doctor. Please check back later.")
