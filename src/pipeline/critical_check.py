"""Step 2: Critical value rule engine. This is the SINGLE source of truth
for critical thresholds — clinical_context.py imports from here rather than
duplicating its own copy."""
import json
import os
from datetime import datetime

CRITICAL_THRESHOLDS = {
    'FastingGlucose':           {'low': 2.5,  'high': 27.8},
    'Glycated Haemoglobin':     {'low': None, 'high': 130.0},
    'Creatinine':               {'low': None, 'high': 400.0},
    'Albumin':                  {'low': 20,   'high': None},
    'UricAcid':                 {'low': None, 'high': 773},
    'Alanine Aminotransferase': {'low': None, 'high': 400.0},
    'AST':                      {'low': None, 'high': 400.0},
    'Sodium':                   {'low': 120.0, 'high': 160.0},
    # Severe hypertriglyceridemia is a recognised acute pancreatitis risk.
    'Triglycerides':            {'low': None, 'high': 11.3},
    # Severely elevated urea (BUN) indicates severe/acute renal impairment.
    'BUN':                      {'low': None, 'high': 35.7},
    'GGT':                      {'low': None, 'high': 500.0},
    # NOTE: TotalCholesterol and HDLCholesterol are deliberately NOT included
    # here. They are chronic cardiovascular risk markers, not acute/same-day
    # emergency indicators — standard clinical critical/panic-value protocols
    # do not flag them, and inventing a threshold would misrepresent clinical
    # practice. An abnormal (but non-critical) reading is still surfaced via
    # the routine reference-range check in clinical_context.py.
}


def check_critical(patient: dict) -> list:
    flags = []
    for biomarker, limits in CRITICAL_THRESHOLDS.items():
        value = patient.get(biomarker)
        if value is None:
            continue
        if limits['low'] is not None and value < limits['low']:
            flags.append(f"{biomarker} critically LOW: {value}")
        if limits['high'] is not None and value > limits['high']:
            flags.append(f"{biomarker} critically HIGH: {value}")
    return flags


def escalate(patient: dict, flags: list, log_path='outputs/logs/escalations.jsonl') -> dict:
    alert = {
        'timestamp': datetime.now().isoformat(),
        'patient_id': patient.get('PatientID'),
        'flags': flags,
        'status': 'ESCALATED — requires immediate clinical review',
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps(alert) + '\n')
    return alert


def run_critical_check(patient: dict) -> dict:
    flags = check_critical(patient)
    if flags:
        return escalate(patient, flags)
    return {'status': 'PROCEED', 'data': patient}