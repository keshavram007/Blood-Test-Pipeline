import datetime


def compute_key_findings(biomarker_table, top_n=5):
    deviations = []
    for b in biomarker_table:
        if b['flag'] == 'Normal':
            continue
        low, high = None, None
        range_str = b['reference_range']
        if '-' in range_str:
            parts = range_str.split('-')
            low, high = float(parts[0]) if parts[0] else None, float(parts[1]) if parts[1] else None
        elif range_str.startswith('<'):
            high = float(range_str[1:])
        elif range_str.startswith('>'):
            low = float(range_str[1:])

        value = b['value']
        if b['flag'] == 'High' and high:
            pct_over = ((value - high) / high) * 100
            deviations.append({'biomarker': b['biomarker'], 'pct_deviation': pct_over, 'direction': 'high'})
        elif b['flag'] == 'Low' and low:
            pct_under = ((low - value) / low) * 100
            deviations.append({'biomarker': b['biomarker'], 'pct_deviation': pct_under, 'direction': 'low'})

    deviations.sort(key=lambda d: d['pct_deviation'], reverse=True)
    return deviations[:top_n]


def _build_header(patient_id, dob, test_date, title):
    lines = []
    lines.append("=" * 60)
    lines.append(title)
    lines.append("=" * 60)
    lines.append(f"Patient ref: {patient_id}    DOB: {dob}    Test date: {test_date}")
    lines.append("=" * 60)
    return lines


def _build_results_section(context, section_number):
    lines = []
    lines.append(f"\n{section_number}. RESULTS SUMMARY")
    lines.append("-" * 60)
    lines.append(f"{'Biomarker':<28}{'Value':<14}{'Reference Range':<18}{'Status'}")
    for b in context['biomarker_table']:
        value_str = f"{b['value']} {b['unit']}"
        lines.append(f"{b['biomarker']:<28}{value_str:<14}{b['reference_range']:<18}{b['flag']}")
    return lines


def _build_risk_section(context, section_number):
    risk = context['risk_assessment']
    anomaly = context['anomaly_assessment']
    lines = []
    lines.append(f"\n{section_number}. RISK ASSESSMENT")
    lines.append("-" * 60)
    lines.append(f"Diabetes Risk Screening Result: {risk['label']} (confidence {risk['probability']:.1%})")
    lines.append(f"Overall Biomarker Deviation Score: {anomaly['score']:.3f} - {anomaly['interpretation']}")
    return lines


def _plain_english_risk_label(risk_label, confidence):
    if risk_label == 'AtRisk':
        if confidence > 0.8:
            return "Your results suggest a notably higher than average chance of blood sugar problems."
        else:
            return "Your results suggest a somewhat higher than average chance of blood sugar problems."
    else:
        return "Your results do not currently suggest an increased risk of blood sugar problems."


def _build_patient_risk_section(context, section_number):
    risk = context['risk_assessment']
    lines = []
    lines.append(f"\n{section_number}. WHAT YOUR RESULTS SHOW")
    lines.append("-" * 60)
    lines.append(_plain_english_risk_label(risk['label'], risk['probability']))
    return lines


def _build_key_findings_section(key_findings, section_number):
    lines = []
    lines.append(f"\n{section_number}. KEY CONTRIBUTING FACTORS (by deviation from reference range)")
    lines.append("-" * 60)
    if key_findings:
        for f in key_findings:
            lines.append(f"- {f['biomarker']}: {f['pct_deviation']:.0f}% {f['direction']} of reference range")
    else:
        lines.append("No biomarkers outside reference range.")
    return lines


def _build_patient_key_findings_section(key_findings, section_number):
    lines = []
    lines.append(f"\n{section_number}. RESULTS THAT STAND OUT")
    lines.append("-" * 60)
    if key_findings:
        lines.append("These results were the main ones outside the usual healthy range:")
        for f in key_findings:
            direction_word = "higher" if f['direction'] == 'high' else "lower"
            lines.append(f"- {f['biomarker']}: {direction_word} than the healthy range")
    else:
        lines.append("All your results were within the healthy range.")
    return lines


def _build_transparency_section(shap_drivers, section_number):
    """SHAP-based feature contribution — doctor version only."""
    lines = []
    lines.append(f"\n{section_number}. MODEL TRANSPARENCY NOTE (For Clinician Reference Only)")
    lines.append("-" * 60)
    lines.append("The following factors had the greatest impact on this patient's AI risk")
    lines.append("assessment (positive = pushed toward higher risk, negative = pushed toward")
    lines.append("lower risk):")
    for d in shap_drivers[:5]:
        sign = '+' if d['shap_value'] > 0 else ''
        lines.append(f"- {d['biomarker']}: {sign}{d['shap_value']} ({d['direction']}, status: {d['flag']})")
    lines.append("")
    return lines


def build_doctor_report(patient_id, dob, test_date, context, clinical_narrative):
    key_findings = compute_key_findings(context['biomarker_table'])
    shap_drivers = context.get('top_shap_drivers', [])

    lines = _build_header(patient_id, dob, test_date, "BLOOD TEST REPORT — DOCTOR REVIEW DRAFT")
    lines += _build_results_section(context, 1)
    lines += _build_risk_section(context, 2)
    lines += _build_key_findings_section(key_findings, 3)
    lines += _build_transparency_section(shap_drivers, 4)

    lines.append("\n5. SUMMARY (plain-English text — this exact wording will be sent to the patient if approved)")
    lines.append("-" * 60)
    lines.append(clinical_narrative)

    lines.append("\n6. REVIEW & APPROVAL")
    lines.append("-" * 60)
    lines.append("Reviewing doctor: ______________________    Date: ______________")
    lines.append("")
    lines.append("*This report was drafted with AI assistance and requires clinician review before")
    lines.append("being provided to the patient. It does not constitute a medical diagnosis.*")

    return "\n".join(lines)


def build_patient_report(patient_id, dob, test_date, context, clinical_narrative,
                          reviewing_doctor=None, approval_date=None):
    key_findings = compute_key_findings(context['biomarker_table'])

    lines = _build_header(patient_id, dob, test_date, "YOUR BLOOD TEST RESULTS")
    lines += _build_results_section(context, 1)
    lines += _build_patient_risk_section(context, 2)
    lines += _build_patient_key_findings_section(key_findings, 3)

    lines.append("\n4. WHAT THIS MEANS FOR YOU")
    lines.append("-" * 60)
    lines.append(clinical_narrative)

    lines.append("\n5. DOCTOR REVIEW")
    lines.append("-" * 60)
    lines.append(f"Reviewed and approved by: {reviewing_doctor or '[Doctor name]'}")
    lines.append(f"Approval date: {approval_date or '[Date]'}")
    lines.append("")
    lines.append("This report has been reviewed and approved by your doctor before being sent to you.")
    lines.append("If you have any questions about these results, please contact your GP surgery.")

    return "\n".join(lines)


if __name__ == '__main__':
    from src.pipeline.inference import BloodTestPredictor
    from src.pipeline.clinical_context import build_clinical_context
    from src.pipeline.report_generation import generate_clinical_narrative

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
    clinical_narrative = generate_clinical_narrative(context)

    patient_id, dob = "P-00521", "1973-04-12"
    test_date = datetime.date.today().isoformat()

    doctor_report = build_doctor_report(patient_id, dob, test_date, context, clinical_narrative)
    print(doctor_report)

    print("\n\n" + "#" * 60)
    print("# (After doctor approval, this version is sent to the patient:)")
    print("#" * 60 + "\n")

    patient_report = build_patient_report(
        patient_id, dob, test_date, context, clinical_narrative,
        reviewing_doctor="Dr. A. Sharma",
        approval_date=datetime.date.today().isoformat()
    )
    print(patient_report)