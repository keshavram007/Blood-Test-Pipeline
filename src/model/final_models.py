import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from src.model.train import train_model

NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]

os.makedirs('outputs/models', exist_ok=True)

def prepare_data():
    df = pd.read_csv('data/processed/labeled_dataset.csv')
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
    feature_cols = NUMERIC_FEATURES + ['Gender_encoded']

    df['Label'] = df['DiabetesStatus'].apply(lambda x: 'Normal' if x == 'Normal' else 'AtRisk')
    label_map = {'Normal': 0, 'AtRisk': 1}
    label_names = ['Normal', 'AtRisk']

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['Label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Label'], random_state=42)

    os.makedirs('data/splits', exist_ok=True)
    train_df[['PatientID']].to_csv('data/splits/final_binary_train_ids.csv', index=False)
    val_df[['PatientID']].to_csv('data/splits/final_binary_val_ids.csv', index=False)
    test_df[['PatientID']].to_csv('data/splits/final_binary_test_ids.csv', index=False)
    print("Saved split PatientIDs to data/splits/final_binary_{train,val,test}_ids.csv")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols]).astype(np.float32)
    X_val = scaler.transform(val_df[feature_cols]).astype(np.float32)
    X_test = scaler.transform(test_df[feature_cols]).astype(np.float32)

    y_train = train_df['Label'].map(label_map).values.astype(np.int64)
    y_val = val_df['Label'].map(label_map).values.astype(np.int64)
    y_test = test_df['Label'].map(label_map).values.astype(np.int64)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    X_train_sm, y_train_sm = X_train_sm.astype(np.float32), y_train_sm.astype(np.int64)

    return X_train_sm, y_train_sm, X_val, y_val, X_test, y_test, scaler, label_names


def train_final_model():
    print("=" * 60)
    print("FINAL MODEL: Binary FT-Transformer (Normal vs At-Risk)")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, label_names = prepare_data()
    weights = torch.tensor(
        compute_class_weight('balanced', classes=np.unique(y_train), y=y_train),
        dtype=torch.float32
    )

    model = train_model(
        X_train, y_train, X_val, y_val, weights,
        d_token=32, n_layers=3, dropout=0.3, lr=1e-3,
        save_path='outputs/models/final_binary_ft.pt',
        label_names=tuple(label_names)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test).to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    f1 = f1_score(y_test, preds, average='macro')
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"\nFINAL MODEL — Test Macro F1: {f1:.4f}, Accuracy: {acc:.4f}, AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=label_names))

    joblib.dump(scaler, 'outputs/models/final_binary_scaler.pkl')
    print("\nSaved: outputs/models/final_binary_ft.pt, final_binary_scaler.pkl")

    return model, scaler, f1, acc, auc


if __name__ == '__main__':
    train_final_model()