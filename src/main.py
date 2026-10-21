import datetime
from dotenv import load_dotenv

load_dotenv()  # picks up GEMINI_API_KEY from a .env file in the repo root, if present

from src.pipeline.patient_upload import validate_patient, load_patient_from_upload, PatientDataError
from src.pipeline.critical_check import check_critical, escalate
from src.pipeline.inference import BloodTestPredictor
from src.pipeline.clinical_context import build_clinical_context
from src.pipeline.report_generation import generate_clinical_narrative
from src.pipeline.generate_report_layout import build_doctor_report, build_patient_report


def run_pipeline(patient: dict, predictor: BloodTestPredictor = None,
                  patient_id: str = None, dob: str = None, test_date: str = None):
    patient_id = patient_id or patient.get('PatientID', 'UNKNOWN')
    dob = dob or 'N/A'
    test_date = test_date or datetime.date.today().isoformat()

    # Step 2: critical check runs FIRST. 
    critical_flags = check_critical(patient)
    if critical_flags:
        escalate(patient, critical_flags)  # still logged, for audit purposes

    # Step 3: three-headed model (classification, SHAP, reconstruction)
    if predictor is None:
        predictor = BloodTestPredictor()
    model_outputs = predictor.predict(patient)

    # Step 4: clinical context layer
    context = build_clinical_context(patient, model_outputs)

    # Step 5: LLM report generation
    narrative = generate_clinical_narrative(context)
    doctor_report = build_doctor_report(patient_id, dob, test_date, context, narrative)

    return {
        'status': 'PROCESSED',
        'patient_id': patient_id,
        'dob': dob,
        'test_date': test_date,
        'context': context,
        'narrative': narrative,
        'doctor_report': doctor_report,
        'critical_alerts': critical_flags,  # [] if none — dashboard shows red if non-empty
    }


def approve_and_generate_patient_report(pipeline_result: dict, reviewing_doctor: str,
                                          approval_date: str = None):
    """Step 6 (approval action) + patient-facing report generation.
    """
    if pipeline_result['status'] != 'PROCESSED':
        raise ValueError("Cannot approve a report that was not fully processed.")

    approval_date = approval_date or datetime.date.today().isoformat()

    patient_report = build_patient_report(
        pipeline_result['patient_id'], pipeline_result['dob'], pipeline_result['test_date'],
        pipeline_result['context'], pipeline_result['narrative'],
        reviewing_doctor=reviewing_doctor, approval_date=approval_date
    )

    return patient_report


def run_pipeline_from_upload(file_source, filename_hint=None, row_index=0,
                               predictor: BloodTestPredictor = None,
                               patient_id: str = None, dob: str = None):
    """Convenience wrapper: load + validate a patient from an uploaded file,
    then run the full pipeline in one call. This is what the UI's upload
    button should call directly."""
    try:
        patient = load_patient_from_upload(file_source, filename_hint=filename_hint, row_index=row_index)
    except PatientDataError as e:
        return {'status': 'ERROR', 'message': str(e)}

    return run_pipeline(patient, predictor=predictor,
                         patient_id=patient_id or patient.get('PatientID'), dob=dob)


if __name__ == '__main__':
    print("=" * 60)
    print("FULL PIPELINE TEST — using demo holdout patient")
    print("=" * 60)

    predictor = BloodTestPredictor()  # built once, reused

    result = run_pipeline_from_upload(
        'data/sample_upload/demo_patients.csv', row_index=0, predictor=predictor
    )

    if result['status'] == 'ERROR':
        print(f"Upload error: {result['message']}")
    else:
        if result['critical_alerts']:
            print(f"\n*** CRITICAL VALUE(S) FLAGGED — Patient {result['patient_id']} ***")
            for flag in result['critical_alerts']:
                print(f"  - {flag}")
            print("(pipeline continued through the model and report anyway)\n")
        print(f"\nPatient {result['patient_id']} processed successfully.")
        print("\n--- DOCTOR REPORT ---")
        print(result['doctor_report'])

        # Simulate doctor approval
        print("\n\n" + "#" * 60)
        print("# Simulating doctor approval...")
        print("#" * 60)
        patient_report = approve_and_generate_patient_report(
            result, reviewing_doctor="Dr. A. Sharma"
        )
        print("\n--- PATIENT REPORT (post-approval) ---")
        print(patient_report)