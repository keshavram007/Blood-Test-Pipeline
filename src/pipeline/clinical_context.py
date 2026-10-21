
from src.pipeline.critical_check import check_critical

REFERENCE_RANGES = {
    'Triglycerides':            (None, 1.7),
    'TotalCholesterol':         (None, 5.0),
    'UricAcid':                 (140.0, 360.0),
    'Albumin':                  (35.0, 50.0),
    'Creatinine':                (45.0, 90.0),
    'Alanine Aminotransferase': (7.0, 35.0),
    'FastingGlucose':           (3.9, 5.5),
    'HDLCholesterol':           (1.2, None),
    'Glycated Haemoglobin':     (20.0, 42.0),
    'Sodium':                   (135.0, 145.0),
    'BUN':                      (2.5, 7.1),
    'GGT':                      (5.0, 36.0),
    'AST':                      (8.0, 40.0),
}

MALE_REFERENCE_RANGES = {
    'UricAcid':                 (200.0, 430.0),
    'Creatinine':                (60.0, 110.0),
    'Alanine Aminotransferase': (7.0, 45.0),
    'HDLCholesterol':           (1.0, None),
    'GGT':                      (8.0, 61.0),
}

UNITS = {
    'Triglycerides': 'mmol/L', 'TotalCholesterol': 'mmol/L', 'UricAcid': 'umol/L',
    'Albumin': 'g/L', 'Creatinine': 'umol/L', 'Alanine Aminotransferase': 'U/L',
    'FastingGlucose': 'mmol/L', 'HDLCholesterol': 'mmol/L', 'Glycated Haemoglobin': 'mmol/mol',
    'Sodium': 'mmol/L', 'BUN': 'mmol/L', 'GGT': 'U/L', 'AST': 'U/L',
}


def _range_for(biomarker, gender):
    if gender == 'Male' and biomarker in MALE_REFERENCE_RANGES:
        return MALE_REFERENCE_RANGES[biomarker]
    return REFERENCE_RANGES[biomarker]


def _format_range(low, high):
    if low is None:
        return f"<{high}"
    if high is None:
        return f">{low}"
    return f"{low}-{high}"


def flag_biomarker(biomarker, value, gender):
    low, high = _range_for(biomarker, gender)
    if low is not None and value < low:
        return 'Low'
    if high is not None and value > high:
        return 'High'
    return 'Normal'


def build_biomarker_table(patient, gender):
    table = []
    for biomarker in REFERENCE_RANGES:
        if biomarker not in patient:
            continue
        value = patient[biomarker]
        low, high = _range_for(biomarker, gender)
        table.append({
            'biomarker': biomarker,
            'value': value,
            'unit': UNITS.get(biomarker, ''),
            'reference_range': _format_range(low, high),
            'flag': flag_biomarker(biomarker, value, gender),
        })
    return table


def _interpret_anomaly(score):
    if score < 0.1208:
        return 'Within expected pattern for a healthy profile'
    if score < 0.1824:
        return 'Mild deviation from typical healthy biomarker pattern'
    return 'Substantial deviation from typical healthy biomarker pattern'


def build_clinical_context(patient, model_outputs):
    gender = patient.get('Gender', 'Female')
    biomarker_table = build_biomarker_table(patient, gender)
    abnormal_biomarkers = [b for b in biomarker_table if b['flag'] != 'Normal']
    critical_alerts = check_critical(patient) 
    shap_importance = model_outputs.get('shap_importance', {})
    ranked_shap = sorted(
        ((name, value) for name, value in shap_importance.items()
         if name != 'Gender_encoded'),
        key=lambda kv: abs(kv[1]), reverse=True
    )
    flag_lookup = {b['biomarker']: b['flag'] for b in biomarker_table}
    top_shap_drivers = [
        {
            'biomarker': name,
            'shap_value': round(value, 4),
            'direction': 'increased risk' if value > 0 else 'decreased risk',
            'flag': flag_lookup.get(name, 'Normal'),
        }
        for name, value in ranked_shap[:5]
    ]
    anomaly_score = model_outputs.get('anomaly_score', 0.0)

    return {
        'patient': {'age': patient.get('Age'), 'gender': gender},
        'escalate': len(critical_alerts) > 0,
        'critical_alerts': critical_alerts,
        'risk_assessment': {
            'label': model_outputs.get('risk_label'),
            'probability': model_outputs.get('risk_probability'),
            'class_probabilities': model_outputs.get('class_probabilities'),
        },
        'anomaly_assessment': {
            'score': anomaly_score,
            'interpretation': _interpret_anomaly(anomaly_score),
        },
        'biomarker_table': biomarker_table,
        'abnormal_biomarkers': abnormal_biomarkers,
        'top_shap_drivers': top_shap_drivers,
    }


def format_context_for_llm(context):
    lines = []
    patient = context['patient']
    lines.append(f"Patient: {patient['age']}-year-old {patient['gender']}")

    if context['escalate']:
        lines.append("\n*** CRITICAL VALUES DETECTED - ESCALATE ***")
        for alert in context['critical_alerts']:
            lines.append(f"- {alert}")

    risk = context['risk_assessment']
    lines.append(f"\nRisk classification: {risk['label']} (confidence {risk['probability']:.1%})")

    anomaly = context['anomaly_assessment']
    lines.append(f"Anomaly score: {anomaly['score']:.3f} - {anomaly['interpretation']}")

    lines.append("\nBiomarker results:")
    for b in context['biomarker_table']:
        marker = f" [{b['flag'].upper()}]" if b['flag'] != 'Normal' else ''
        lines.append(f"- {b['biomarker']}: {b['value']} {b['unit']} (ref: {b['reference_range']}){marker}")

    if context['top_shap_drivers']:
        lines.append("\nBiomarkers most influencing this risk assessment:")
        for d in context['top_shap_drivers']:
            lines.append(f"- {d['biomarker']} (SHAP {d['shap_value']}, {d['direction']}, {d['flag']})")

    return '\n'.join(lines)


if __name__ == '__main__':
    from src.pipeline.inference import BloodTestPredictor

    example_patient = {
        'Age': 52, 'Gender': 'Male', 'Triglycerides': 2.4, 'TotalCholesterol': 6.1,
        'UricAcid': 410.0, 'Albumin': 44.0, 'Creatinine': 95.0,
        'Alanine Aminotransferase': 38.0, 'HDLCholesterol': 0.9,
        'Sodium': 139.0, 'BUN': 5.8, 'GGT': 55.0, 'AST': 32.0,
        'FastingGlucose': 5.8, 'Glycated Haemoglobin': 40.0,
    }

    predictor = BloodTestPredictor()
    model_outputs = predictor.predict(example_patient)
    context = build_clinical_context(example_patient, model_outputs)

    print(format_context_for_llm(context))