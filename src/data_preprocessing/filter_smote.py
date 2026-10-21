import pandas as pd

def filter_synthetic_diabetic():
    train_df = pd.read_csv('data/processed/train.csv')
    smote_df = pd.read_csv('data/processed/train_smote.csv')
    real_diabetic_count = (train_df['DiabetesStatus'] == 'Diabetic').sum()
    diabetic_rows = smote_df[smote_df['DiabetesStatus'] == 'Diabetic'].reset_index(drop=True)
    real_diabetic = diabetic_rows.iloc[:real_diabetic_count]
    synthetic_diabetic = diabetic_rows.iloc[real_diabetic_count:]
    valid_synthetic = synthetic_diabetic[(synthetic_diabetic['FastingGlucose'] >= 7.0) | (synthetic_diabetic['Glycated Haemoglobin'] >= 48)]
    print(f"Synthetic rows kept: {len(valid_synthetic)} / {len(synthetic_diabetic)}")
    normal_rows = smote_df[smote_df['DiabetesStatus'] == 'Normal']
    prediabetic_rows = smote_df[smote_df['DiabetesStatus'] == 'Prediabetic']
    filtered = pd.concat([normal_rows, prediabetic_rows, real_diabetic, valid_synthetic],ignore_index=True)
    filtered.to_csv('data/processed/train_smote.csv', index=False)
    print("\nFinal class distribution:")
    print(filtered['DiabetesStatus'].value_counts())
    return filtered

if __name__ == '__main__':
    filter_synthetic_diabetic()