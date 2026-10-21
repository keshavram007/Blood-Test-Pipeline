import pandas as pd

REQUIRED_NUMERIC_FIELDS = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST', 'FastingGlucose', 'Glycated Haemoglobin'
]
REQUIRED_FIELDS = REQUIRED_NUMERIC_FIELDS + ['Gender']


class PatientDataError(Exception):
    """Raised when uploaded patient data is missing fields or malformed."""
    pass


def validate_patient(patient: dict) -> dict:
    missing = [f for f in REQUIRED_FIELDS if f not in patient or pd.isna(patient[f])]
    if missing:
        raise PatientDataError(f"Missing required field(s): {missing}")

    if patient['Gender'] not in ('Male', 'Female'):
        raise PatientDataError(f"Gender must be 'Male' or 'Female', got: {patient['Gender']!r}")

    validated = dict(patient)
    for field in REQUIRED_NUMERIC_FIELDS:
        try:
            validated[field] = float(patient[field])
        except (ValueError, TypeError):
            raise PatientDataError(f"Field '{field}' must be numeric, got: {patient[field]!r}")
        if validated[field] < 0:
            raise PatientDataError(f"Field '{field}' cannot be negative: {validated[field]}")

    return validated


def _read_uploaded_file(file_source, filename_hint=None):
    """Reads a CSV or Excel file, whether given as a filepath (str) or an
    in-memory file-like object (e.g. Streamlit's UploadedFile)."""
    name = filename_hint if filename_hint else (
        file_source if isinstance(file_source, str) else getattr(file_source, 'name', '')
    )

    if name.endswith('.csv'):
        return pd.read_csv(file_source)
    elif name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_source)
    else:
        raise PatientDataError(
            f"Unsupported file format ({name}). Please upload a .csv or .xlsx file."
        )


def load_patient_from_upload(file_source, filename_hint=None, row_index=0) -> dict:
    """Reads ONE patient from an uploaded file — filepath (str) or an
    in-memory file object (e.g. from st.file_uploader). Returns a validated
    patient dict ready for BloodTestPredictor.predict()."""
    df = _read_uploaded_file(file_source, filename_hint)

    if len(df) == 0:
        raise PatientDataError("Uploaded file contains no rows.")
    if row_index >= len(df):
        raise PatientDataError(f"Row index {row_index} out of range (file has {len(df)} rows).")

    missing_columns = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing_columns:
        raise PatientDataError(f"Uploaded file is missing required columns: {missing_columns}")

    patient_row = df.iloc[row_index]
    patient = {field: patient_row[field] for field in REQUIRED_FIELDS}
    if 'PatientID' in df.columns:
        patient['PatientID'] = patient_row['PatientID']

    return validate_patient(patient)


def load_all_patients_from_upload(file_source, filename_hint=None) -> list:
    """Reads ALL rows from an uploaded file — filepath or in-memory object.
    Returns a list of validated patient dicts."""
    df = _read_uploaded_file(file_source, filename_hint)

    missing_columns = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing_columns:
        raise PatientDataError(f"Uploaded file is missing required columns: {missing_columns}")

    patients = []
    for i in range(len(df)):
        patient_row = df.iloc[i]
        patient = {field: patient_row[field] for field in REQUIRED_FIELDS}
        if 'PatientID' in df.columns:
            patient['PatientID'] = patient_row['PatientID']
        patients.append(validate_patient(patient))

    return patients


if __name__ == '__main__':
    # Local testing: still works the same way using a filepath
    patients = load_all_patients_from_upload('data/sample_upload/demo_patients.csv')
    print(f"Loaded {len(patients)} patients from demo file")
    print("\nFirst patient:")
    print(patients[0])