
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import patch
import pytest

from src.pipeline.inference import BloodTestPredictor
from src.pipeline.clinical_context import build_clinical_context
from src.pipeline.generate_report_layout import build_doctor_report, build_patient_report
from src.main import run_pipeline, approve_and_generate_patient_report


def _sample_patient(**overrides):
    p = {
        'PatientID': 'INTEG001', 'Age': 52, 'Gender': 'Female',
        'Triglycerides': 1.4, 'TotalCholesterol': 4.8, 'UricAcid': 280,
        'Albumin': 43, 'Creatinine': 68, 'Alanine Aminotransferase': 22,
        'HDLCholesterol': 1.3, 'Sodium': 140, 'BUN': 5.2, 'GGT': 24, 'AST': 21,
        'FastingGlucose': 5.6, 'Glycated Haemoglobin': 40,
    }
    p.update(overrides)
    return p


@pytest.fixture(scope='module')
def predictor():
    return BloodTestPredictor()


# ---------------------------------------------------------------------------
# FT-Transformer/SHAP/Autoencoder output -> Clinical Context
# ---------------------------------------------------------------------------

def test_predictor_output_flows_into_clinical_context(predictor):
    patient = _sample_patient()
    model_outputs = predictor.predict(patient)

    assert 'risk_label' in model_outputs
    assert 'shap_importance' in model_outputs
    assert 'anomaly_score' in model_outputs

    context = build_clinical_context(patient, model_outputs)

    assert 'biomarker_table' in context
    assert 'top_shap_drivers' in context
    assert len(context['biomarker_table']) > 0
    table_names = {b['biomarker'] for b in context['biomarker_table']}
    for driver in context['top_shap_drivers']:
        assert driver['biomarker'] == 'Age' or driver['biomarker'] in table_names


# ---------------------------------------------------------------------------
# Clinical Context -> LLM Report Generation -> report layout
# ---------------------------------------------------------------------------

@patch('src.main.generate_clinical_narrative')
def test_clinical_context_flows_into_report_generation(mock_narrative, predictor):
    mock_narrative.return_value = "Your results look mostly fine, but it's worth a follow-up."

    patient = _sample_patient()
    model_outputs = predictor.predict(patient)
    context = build_clinical_context(patient, model_outputs)

    doctor_report = build_doctor_report('INTEG001', '1972-01-01', '2026-08-20',
                                         context, mock_narrative.return_value)
    assert 'INTEG001' in doctor_report
    assert mock_narrative.return_value in doctor_report


# ---------------------------------------------------------------------------
# Full run_pipeline() wiring — critical check -> model -> context -> report
# ---------------------------------------------------------------------------

@patch('src.main.generate_clinical_narrative')
def test_run_pipeline_processes_normal_patient_end_to_end(mock_narrative, predictor):
    mock_narrative.return_value = "Everything looks within a normal range."
    patient = _sample_patient()

    result = run_pipeline(patient, predictor=predictor, patient_id='INTEG001')

    assert result['status'] == 'PROCESSED'
    assert result['critical_alerts'] == []
    assert 'doctor_report' in result
    assert 'context' in result


# The key non-blocking-design test: a critical value must still flag AND
@patch('src.main.generate_clinical_narrative')
def test_run_pipeline_flags_critical_but_still_completes(mock_narrative, predictor):
    mock_narrative.return_value = "Please seek medical attention soon."
    # FastingGlucose above the 27.8 critical threshold in critical_check.py
    patient = _sample_patient(FastingGlucose=28.5)

    result = run_pipeline(patient, predictor=predictor, patient_id='INTEG002')

    # Per the current design: critical values flag but do NOT stop the pipeline
    assert result['status'] == 'PROCESSED'
    assert len(result['critical_alerts']) > 0
    assert any('FastingGlucose' in f for f in result['critical_alerts'])
    # The pipeline must have still gone all the way through to a doctor report
    assert 'doctor_report' in result


# ---------------------------------------------------------------------------
# Doctor approval routing -> patient report generation
# ---------------------------------------------------------------------------

@patch('src.main.generate_clinical_narrative')
def test_approve_and_generate_patient_report_routes_correctly(mock_narrative, predictor):
    mock_narrative.return_value = "Your results look within a normal range."
    patient = _sample_patient()
    result = run_pipeline(patient, predictor=predictor, patient_id='INTEG003')

    patient_report = approve_and_generate_patient_report(
        result, reviewing_doctor='Dr. Test', approval_date='2026-08-20'
    )
    assert 'Dr. Test' in patient_report
    assert mock_narrative.return_value in patient_report


def test_approve_rejects_unprocessed_result():
    
    bad_result = {'status': 'ERROR', 'message': 'bad file'}
    with pytest.raises(ValueError):
        approve_and_generate_patient_report(bad_result, reviewing_doctor='Dr. Test')


if __name__ == '__main__':
    print("Run this file with pytest, not directly:")
    print("  python -m pytest testing/integration_tests.py -v")
