import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import patch

from src.pipeline.inference import BloodTestPredictor
from src.main import run_pipeline, approve_and_generate_patient_report


# ---------------------------------------------------------------------------
# Synthetic scenarios. Values are deliberately picked to sit clearly inside
# each expected region rather than near a boundary, so a pass/fail here
# reflects genuine pipeline behaviour rather than a threshold edge case.
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        'name': 'Normal',
        'expected_critical': False,
        'patient': {
            'PatientID': 'E2E-NORMAL', 'Age': 34, 'Gender': 'Female',
            'Triglycerides': 1.1, 'TotalCholesterol': 4.4, 'UricAcid': 250,
            'Albumin': 44, 'Creatinine': 65, 'Alanine Aminotransferase': 18,
            'HDLCholesterol': 1.5, 'Sodium': 140, 'BUN': 4.5, 'GGT': 18, 'AST': 19,
            'FastingGlucose': 4.8, 'Glycated Haemoglobin': 34,
        },
    },
    {
        'name': 'High-Risk (At-Risk, non-critical)',
        'expected_critical': False,
        'patient': {
            'PatientID': 'E2E-HIGHRISK', 'Age': 61, 'Gender': 'Male',
            'Triglycerides': 2.6, 'TotalCholesterol': 6.1, 'UricAcid': 410,
            'Albumin': 39, 'Creatinine': 95, 'Alanine Aminotransferase': 38,
            'HDLCholesterol': 0.95, 'Sodium': 141, 'BUN': 6.8, 'GGT': 55, 'AST': 34,
            # HbA1c 46 and glucose 6.2 sit in the Prediabetic/At-Risk band,
            # well below any critical threshold.
            'FastingGlucose': 6.2, 'Glycated Haemoglobin': 46,
        },
    },
    {
        'name': 'Critical-Value (dangerously high glucose)',
        'expected_critical': True,
        'patient': {
            'PatientID': 'E2E-CRITICAL', 'Age': 58, 'Gender': 'Male',
            'Triglycerides': 3.0, 'TotalCholesterol': 6.5, 'UricAcid': 420,
            'Albumin': 38, 'Creatinine': 100, 'Alanine Aminotransferase': 40,
            'HDLCholesterol': 0.9, 'Sodium': 140, 'BUN': 7.0, 'GGT': 60, 'AST': 35,
            # 28.5 mmol/L is above the 27.8 critical threshold in critical_check.py
            'FastingGlucose': 28.5, 'Glycated Haemoglobin': 80,
        },
    },
]


def run_scenarios():
    predictor = BloodTestPredictor()
    results = []

    with patch('src.main.generate_clinical_narrative',
               return_value="[mocked narrative — LLM not called during E2E test]"):
        for scenario in SCENARIOS:
            print(f"\n{'=' * 70}\nSCENARIO: {scenario['name']}\n{'=' * 70}")
            result = run_pipeline(scenario['patient'], predictor=predictor,
                                   patient_id=scenario['patient']['PatientID'])

            got_critical = len(result.get('critical_alerts', [])) > 0
            checks = {
                'pipeline completed (status == PROCESSED)': result['status'] == 'PROCESSED',
                'has doctor_report': 'doctor_report' in result,
                'has clinical context': 'context' in result,
                f"critical_alerts present == {scenario['expected_critical']}":
                    got_critical == scenario['expected_critical'],
            }

            # Doctor approval routing must still work regardless of critical flag,
            # since critical values no longer stop the pipeline in this design.
            try:
                patient_report = approve_and_generate_patient_report(
                    result, reviewing_doctor='Dr. Test', approval_date='2026-08-20'
                )
                checks['doctor approval produces a patient report'] = bool(patient_report)
            except Exception as e:
                checks['doctor approval produces a patient report'] = False
                print(f"  (approval raised: {e})")

            all_passed = all(checks.values())
            print(f"Predicted risk label : {result['context']['risk_label']}"
                  if 'context' in result and 'risk_label' in result.get('context', {})
                  else f"Predicted status     : {result['status']}")
            if result.get('critical_alerts'):
                print(f"Critical alerts      : {result['critical_alerts']}")
            print()
            for check_name, passed in checks.items():
                print(f"  {'PASS' if passed else 'FAIL'}  {check_name}")

            results.append((scenario['name'], all_passed))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    n_passed = sum(1 for _, p in results if p)
    print(f"\n{n_passed}/{len(results)} scenarios passed.")


if __name__ == '__main__':
    run_scenarios()
