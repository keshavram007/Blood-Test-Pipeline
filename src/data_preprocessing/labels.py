def derive_diabetes_label(df):
    """NICE/NHS thresholds: HbA1c >=48 mmol/mol OR Glucose >=7.0 mmol/L = Diabetic;
    HbA1c >=42 OR Glucose >=5.5 = Prediabetic; else Normal."""
    df = df.copy()

    def classify(row):
        if row['Glycated Haemoglobin'] >= 48 or row['FastingGlucose'] >= 7.0:
            return 'Diabetic'
        elif row['Glycated Haemoglobin'] >= 42 or row['FastingGlucose'] >= 5.5:
            return 'Prediabetic'
        else:
            return 'Normal'

    df['DiabetesStatus'] = df.apply(classify, axis=1)
    return df