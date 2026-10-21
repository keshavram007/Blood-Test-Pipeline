import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from src.model.train import train_model

NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]

def load_split():
    df = pd.read_csv('data/processed/labeled_dataset.csv')
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
    feature_cols = NUMERIC_FEATURES + ['Gender_encoded']
    label_map = {'Normal': 0, 'Prediabetic': 1, 'Diabetic': 2}

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['DiabetesStatus'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['DiabetesStatus'], random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols]).astype(np.float32)
    X_val = scaler.transform(val_df[feature_cols]).astype(np.float32)
    X_test = scaler.transform(test_df[feature_cols]).astype(np.float32)

    y_train = train_df['DiabetesStatus'].map(label_map).values.astype(np.int64)
    y_val = val_df['DiabetesStatus'].map(label_map).values.astype(np.int64)
    y_test = test_df['DiabetesStatus'].map(label_map).values.astype(np.int64)

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test).to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return f1_score(y_test, preds, average='macro'), preds


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_split()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    label_names = ('Normal', 'Prediabetic', 'Diabetic')
    results = {}

    # --- 1. Class weighting on raw imbalanced data ---
    print("=" * 60, "\nEXPERIMENT: Class Weighting\n", "=" * 60)
    weights = torch.tensor(compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train), dtype=torch.float32)
    model = train_model(X_train, y_train, X_val, y_val, weights,
                         d_token=16, n_layers=3, dropout=0.3, lr=1e-3,
                         save_path='experiments/output/models/exp_classweight.pt', label_names=label_names)
    f1, preds = evaluate(model, X_test, y_test, device)
    results['Class Weighting'] = f1
    print(classification_report(y_test, preds, target_names=label_names))

    # --- 2. SMOTE ---
    print("=" * 60, "\nEXPERIMENT: SMOTE\n", "=" * 60)
    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    X_sm, y_sm = X_sm.astype(np.float32), y_sm.astype(np.int64)
    uniform = torch.tensor([1.0, 1.0, 1.0])
    model = train_model(X_sm, y_sm, X_val, y_val, uniform,
                         d_token=16, n_layers=3, dropout=0.3, lr=1e-3,
                         save_path='experiments/output/models/exp_smote.pt', label_names=label_names)
    f1, preds = evaluate(model, X_test, y_test, device)
    results['SMOTE'] = f1
    print(classification_report(y_test, preds, target_names=label_names))

    print("\n\nSUMMARY:")
    for k, v in results.items():
        print(f"{k}: Macro F1 = {v:.4f}")