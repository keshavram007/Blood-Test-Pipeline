import pandas as pd
import os

def holdout(n_per_class=4, random_state=42):
    df = pd.read_csv('data/processed/labeled_dataset.csv')

    demo_sample = df.groupby('DiabetesStatus', group_keys=False).apply(
        lambda x: x.sample(min(len(x), n_per_class), random_state=random_state)
    )

    remaining_df = df.drop(demo_sample.index).reset_index(drop=True)
    demo_sample = demo_sample.reset_index(drop=True)

    os.makedirs('data/sample_upload', exist_ok=True)

    upload_columns = [
        'PatientID', 'Age', 'Gender', 'Triglycerides', 'TotalCholesterol',
        'UricAcid', 'Albumin', 'Creatinine', 'Alanine Aminotransferase',
        'HDLCholesterol', 'Sodium', 'BUN', 'GGT', 'AST',
        'FastingGlucose', 'Glycated Haemoglobin'
    ]

    demo_upload = demo_sample[upload_columns]
    demo_upload.to_csv('data/sample_upload/demo_patients.csv', index=False)

    demo_reference = demo_sample[upload_columns + ['DiabetesStatus']]
    demo_reference.to_csv('data/sample_upload/demo_patients_WITH_ANSWERS.csv', index=False)

    remaining_df.to_csv('data/processed/labeled_dataset.csv', index=False)

    print(f"Held out {len(demo_sample)} patients for UI/demo use:")
    print(f"\nRemaining dataset for train/val/test: {len(remaining_df)} rows")
    print("data/processed/labeled_dataset.csv (OVERWRITTEN — now excludes demo patients)")

    return demo_sample, remaining_df


if __name__ == '__main__':
    holdout(n_per_class=4)