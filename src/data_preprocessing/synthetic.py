import pandas as pd
from imblearn.over_sampling import SMOTE

NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'FastingGlucose',
    'HDLCholesterol', 'Glycated Haemoglobin', 'Sodium', 'BUN', 'GGT', 'AST'
]

def generate_synthetic_data(train_path='data/processed/train.csv',
                              output_path='data/processed/train_smote.csv',
                              random_state=42):
    df = pd.read_csv(train_path)
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})

    feature_cols = NUMERIC_FEATURES + ['Gender_encoded']
    X = df[feature_cols]
    y = df['DiabetesStatus']

    print("Before SMOTE:")
    print(y.value_counts())

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("\nAfter SMOTE:")
    print(y_resampled.value_counts())

    result = pd.DataFrame(X_resampled, columns=feature_cols)
    result['DiabetesStatus'] = y_resampled.values
    result['Gender'] = result['Gender_encoded'].round().map({0: 'Male', 1: 'Female'})
    result = result.drop(columns=['Gender_encoded'])

    round_map = {
        'Age': 0, 'Triglycerides': 2, 'TotalCholesterol': 2, 'UricAcid': 1,
        'Albumin': 1, 'Creatinine': 1, 'Alanine Aminotransferase': 1,
        'FastingGlucose': 2, 'HDLCholesterol': 2, 'Glycated Haemoglobin': 1,
        'Sodium': 1, 'BUN': 2, 'GGT': 1, 'AST': 1
    }
    for col, decimals in round_map.items():
        result[col] = result[col].round(decimals)

    result.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}, shape: {result.shape}")
    return result


if __name__ == '__main__':
    generate_synthetic_data()