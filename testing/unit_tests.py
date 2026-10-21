"""
Unit tests — individual pipeline components tested in isolation, with no
trained model involved. Covers the deterministic, rule-based pieces:
critical value thresholds, diagnosis labelling, unit conversion, and
patient input validation.

Run from the repo root:
    python -m pytest testing/unit_tests.py -v

(or: python testing/unit_tests.py   -- runs without pytest, plain asserts)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pytest

from src.pipeline.critical_check import check_critical, run_critical_check
from src.pipeline.patient_upload import validate_patient, PatientDataError
from src.data_preprocessing.labels import derive_diabetes_label
from src.data_preprocessing.unit_conversion import rename_and_convert


# ---------------------------------------------------------------------------
# critical_check.py
# ---------------------------------------------------------------------------

def _base_patient(**overrides):
    """A patient with every value safely inside normal range, so tests only
    need to override the field(s) they're actually testing."""
    p = {
        'PatientID': 'TEST001',
        'FastingGlucose': 5.0,
        'Glycated Haemoglobin': 38.0,
        'Creatinine': 70.0,
        'Albumin': 42.0,
        'UricAcid': 300.0,
        'Alanine Aminotransferase': 20.0,
        'AST': 20.0,
        'Sodium': 140.0,
        'Triglycerides': 1.2,
        'BUN': 5.0,
        'GGT': 20.0,
    }
    p.update(overrides)
    return p


# A fully normal patient should produce zero critical flags.
def test_critical_check_no_flags_on_normal_patient():
    flags = check_critical(_base_patient())
    assert flags == []


# Glucose above the 27.8 critical threshold should be flagged HIGH.
def test_critical_check_flags_high_glucose():
    flags = check_critical(_base_patient(FastingGlucose=28.5))  # above 27.8 threshold
    assert any('FastingGlucose' in f for f in flags)


# Sodium below the 120.0 critical threshold should be flagged LOW.
def test_critical_check_flags_low_sodium():
    flags = check_critical(_base_patient(Sodium=115.0))  # below 120.0 threshold
    assert any('Sodium' in f for f in flags)


# Two simultaneously abnormal values should both be flagged, not just one.
def test_critical_check_flags_multiple_simultaneously():
    flags = check_critical(_base_patient(FastingGlucose=28.5, Sodium=115.0))
    assert len(flags) == 2


# Missing (optional) biomarker fields must not crash the check.
def test_critical_check_ignores_missing_biomarkers():
    # A patient dict missing some optional fields should not crash the check
    partial = {'PatientID': 'TEST002', 'FastingGlucose': 5.0}
    flags = check_critical(partial)
    assert flags == []


# The wrapper returns a PROCEED status when nothing is critical.
def test_run_critical_check_proceed_status():
    result = run_critical_check(_base_patient())
    assert result['status'] == 'PROCEED'


# escalate() writes a correctly-labelled audit log entry to disk.
def test_run_critical_check_escalates_and_logs(tmp_path):
    log_path = tmp_path / "escalations.jsonl"
    from src.pipeline.critical_check import check_critical, escalate
    flags = check_critical(_base_patient(FastingGlucose=28.5))
    alert = escalate(_base_patient(FastingGlucose=28.5), flags, log_path=str(log_path))
    assert alert['status'] == 'ESCALATED — requires immediate clinical review'
    assert log_path.exists()


# ---------------------------------------------------------------------------
# labels.py — NICE/NHS diagnosis thresholds
# ---------------------------------------------------------------------------

# Both HbA1c and glucose well within range should label Normal.
def test_label_normal():
    df = pd.DataFrame([{'Glycated Haemoglobin': 30.0, 'FastingGlucose': 4.5}])
    result = derive_diabetes_label(df)
    assert result.iloc[0]['DiabetesStatus'] == 'Normal'


# Elevated HbA1c alone (glucose normal) should label Prediabetic.
def test_label_prediabetic_by_hba1c():
    df = pd.DataFrame([{'Glycated Haemoglobin': 44.0, 'FastingGlucose': 4.5}])
    result = derive_diabetes_label(df)
    assert result.iloc[0]['DiabetesStatus'] == 'Prediabetic'


# Elevated glucose alone (HbA1c normal) should label Prediabetic.
def test_label_prediabetic_by_glucose():
    df = pd.DataFrame([{'Glycated Haemoglobin': 30.0, 'FastingGlucose': 5.7}])
    result = derive_diabetes_label(df)
    assert result.iloc[0]['DiabetesStatus'] == 'Prediabetic'


# High HbA1c alone should be enough to label Diabetic.
def test_label_diabetic_by_hba1c():
    df = pd.DataFrame([{'Glycated Haemoglobin': 50.0, 'FastingGlucose': 4.5}])
    result = derive_diabetes_label(df)
    assert result.iloc[0]['DiabetesStatus'] == 'Diabetic'


# High glucose alone should be enough to label Diabetic.
def test_label_diabetic_by_glucose():
    df = pd.DataFrame([{'Glycated Haemoglobin': 30.0, 'FastingGlucose': 7.5}])
    result = derive_diabetes_label(df)
    assert result.iloc[0]['DiabetesStatus'] == 'Diabetic'


# A value exactly AT the Diabetic threshold (48) must classify Diabetic, confirming >= not >.
def test_label_boundary_exactly_at_diabetic_threshold():
    # NICE/NHS thresholds use >=, so the boundary value itself must classify Diabetic
    df = pd.DataFrame([{'Glycated Haemoglobin': 48.0, 'FastingGlucose': 4.5}])
    result = derive_diabetes_label(df)
    assert result.iloc[0]['DiabetesStatus'] == 'Diabetic'


# ---------------------------------------------------------------------------
# unit_conversion.py
# ---------------------------------------------------------------------------

# A textbook-normal glucose value in mg/dL should convert correctly to mmol/L.
def test_unit_conversion_glucose_mgdl_to_mmol():
    # 90 mg/dL is a textbook-normal fasting glucose -> should convert to ~5.0 mmol/L
    df = pd.DataFrame([{
        'SEQN': 1, 'RIDAGEYR': 40, 'RIAGENDR': 1, 'CYCLE': '2017-2018',
        'LBXGLU': 90.0, 'LBXGH': 5.5, 'LBDHDD': 50.0, 'LBXSTR': 100.0,
        'LBXSCH': 180.0, 'LBXSUA': 5.0, 'LBXSAL': 4.0, 'LBXSCR': 1.0,
        'LBXSATSI': 20.0, 'LBXSNASI': 140.0, 'LBXSBU': 14.0, 'LBXSGTSI': 20.0,
        'LBXSASSI': 20.0,
    }])
    result = rename_and_convert(df)
    assert abs(result.iloc[0]['FastingGlucose'] - 5.0) < 0.05


# NHANES numeric gender codes (1/2) should map to 'Male'/'Female' correctly.
def test_unit_conversion_gender_mapping():
    df = pd.DataFrame([
        {'SEQN': 1, 'RIDAGEYR': 40, 'RIAGENDR': 1, 'CYCLE': 'x', 'LBXGLU': 90.0,
         'LBXGH': 5.5, 'LBDHDD': 50.0, 'LBXSTR': 100.0, 'LBXSCH': 180.0, 'LBXSUA': 5.0,
         'LBXSAL': 4.0, 'LBXSCR': 1.0, 'LBXSATSI': 20.0, 'LBXSNASI': 140.0,
         'LBXSBU': 14.0, 'LBXSGTSI': 20.0, 'LBXSASSI': 20.0},
        {'SEQN': 2, 'RIDAGEYR': 40, 'RIAGENDR': 2, 'CYCLE': 'x', 'LBXGLU': 90.0,
         'LBXGH': 5.5, 'LBDHDD': 50.0, 'LBXSTR': 100.0, 'LBXSCH': 180.0, 'LBXSUA': 5.0,
         'LBXSAL': 4.0, 'LBXSCR': 1.0, 'LBXSATSI': 20.0, 'LBXSNASI': 140.0,
         'LBXSBU': 14.0, 'LBXSGTSI': 20.0, 'LBXSASSI': 20.0},
    ])
    result = rename_and_convert(df)
    assert result.iloc[0]['Gender'] == 'Male'
    assert result.iloc[1]['Gender'] == 'Female'


# ---------------------------------------------------------------------------
# patient_upload.py — input validation
# ---------------------------------------------------------------------------

def _valid_patient_dict():
    return {
        'Age': 45, 'Triglycerides': 1.2, 'TotalCholesterol': 4.5, 'UricAcid': 300,
        'Albumin': 42, 'Creatinine': 70, 'Alanine Aminotransferase': 20,
        'HDLCholesterol': 1.3, 'Sodium': 140, 'BUN': 5, 'GGT': 20, 'AST': 20,
        'FastingGlucose': 5.0, 'Glycated Haemoglobin': 38, 'Gender': 'Male',
    }


# A fully valid patient dict should pass validation and coerce types correctly.
def test_validate_patient_accepts_valid_input():
    result = validate_patient(_valid_patient_dict())
    assert result['Gender'] == 'Male'
    assert isinstance(result['Age'], float)


# A missing required field should raise PatientDataError.
def test_validate_patient_rejects_missing_field():
    patient = _valid_patient_dict()
    del patient['Age']
    with pytest.raises(PatientDataError):
        validate_patient(patient)


# An invalid Gender value should raise PatientDataError.
def test_validate_patient_rejects_invalid_gender():
    patient = _valid_patient_dict()
    patient['Gender'] = 'Unknown'
    with pytest.raises(PatientDataError):
        validate_patient(patient)


# A negative value for a field that can't be negative should raise PatientDataError.
def test_validate_patient_rejects_negative_value():
    patient = _valid_patient_dict()
    patient['Age'] = -5
    with pytest.raises(PatientDataError):
        validate_patient(patient)


# A non-numeric value in a numeric field should raise PatientDataError.
def test_validate_patient_rejects_non_numeric_value():
    patient = _valid_patient_dict()
    patient['Age'] = 'forty-five'
    with pytest.raises(PatientDataError):
        validate_patient(patient)


if __name__ == '__main__':
    # Plain-assert fallback if pytest isn't installed — runs every test
    # function defined above and prints a pass/fail summary.
    import traceback
    tests = [(name, fn) for name, fn in list(globals().items())
              if name.startswith('test_') and callable(fn)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            if 'tmp_path' in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                import tempfile
                with tempfile.TemporaryDirectory() as d:
                    from pathlib import Path
                    fn(Path(d))
            else:
                fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
