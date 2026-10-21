import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.model.train import train_model

NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]


def prepare_full_binary_data():
    df = pd.read_csv('data/processed/labeled_dataset.csv')
    df['BinaryLabel'] = df['DiabetesStatus'].apply(lambda x: 'Normal' if x == 'Normal' else 'AtRisk')
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
    feature_cols = NUMERIC_FEATURES + ['Gender_encoded']
    label_map = {'Normal': 0, 'AtRisk': 1}

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['BinaryLabel'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['BinaryLabel'], random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols]).astype(np.float32)
    X_val = scaler.transform(val_df[feature_cols]).astype(np.float32)
    X_test = scaler.transform(test_df[feature_cols]).astype(np.float32)

    y_train = train_df['BinaryLabel'].map(label_map).values.astype(np.int64)
    y_val = val_df['BinaryLabel'].map(label_map).values.astype(np.int64)
    y_test = test_df['BinaryLabel'].map(label_map).values.astype(np.int64)

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_smote_ratio(ratio):
    """
    ratio=1 -> SMOTE brings AtRisk to 1:1 with Normal (full balance)
    ratio=2 -> SMOTE brings AtRisk to only half of Normal's count (1:2)
    ratio=3 -> AtRisk to 1/3 of Normal's count
    ratio=4 -> AtRisk to 1/4 of Normal's count
    """
    print("\n\n" + "#" * 70)
    print(f"# FULL DATASET — SMOTE TARGET RATIO 1:{ratio} (AtRisk:Normal)")
    print("#" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_full_binary_data()

    normal_count = np.sum(y_train == 0)
    target_atrisk_count = normal_count // ratio  # e.g. ratio=2 -> half of Normal's count

    current_atrisk_count = np.sum(y_train == 1)
    if target_atrisk_count <= current_atrisk_count:
        print(f"Target AtRisk count ({target_atrisk_count}) <= current count ({current_atrisk_count}) "
              f"— skipping SMOTE, using data as-is.")
        X_train_res, y_train_res = X_train, y_train
    else:
        smote = SMOTE(sampling_strategy={1: target_atrisk_count}, random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        X_train_res, y_train_res = X_train_res.astype(np.float32), y_train_res.astype(np.int64)

    print(f"Before SMOTE: Normal={normal_count}, AtRisk={current_atrisk_count}")
    print(f"After SMOTE:  Normal={np.sum(y_train_res==0)}, AtRisk={np.sum(y_train_res==1)}")

    weights = torch.tensor(
        compute_class_weight('balanced', classes=np.unique(y_train_res), y=y_train_res),
        dtype=torch.float32
    )

    ft_model = train_model(
        X_train_res, y_train_res, X_val, y_val, weights,
        d_token=32, n_layers=2, dropout=0.2, lr=1e-3,
        save_path=f'outputs/models/binary_smote_ratio_1to{ratio}_ft.pt',
        label_names=('Normal', 'AtRisk')
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ft_model.eval()
    with torch.no_grad():
        logits = ft_model(torch.tensor(X_test).to(device))
        ft_preds = torch.argmax(logits, dim=1).cpu().numpy()
    ft_f1 = f1_score(y_test, ft_preds, average='macro')
    ft_acc = accuracy_score(y_test, ft_preds)

    xgb_model = XGBClassifier(objective='binary:logistic', random_state=42,
                               learning_rate=0.1, max_depth=7, n_estimators=300,
                               eval_metric='logloss')
    xgb_model.fit(X_train_res, y_train_res)
    xgb_preds = xgb_model.predict(X_test)
    xgb_f1 = f1_score(y_test, xgb_preds, average='macro')
    xgb_acc = accuracy_score(y_test, xgb_preds)

    print(f"\nFT-Transformer: Macro F1={ft_f1:.4f}, Accuracy={ft_acc:.4f}")
    print(classification_report(y_test, ft_preds, target_names=['Normal', 'AtRisk']))
    print(f"\nXGBoost: Macro F1={xgb_f1:.4f}, Accuracy={xgb_acc:.4f}")
    print(classification_report(y_test, xgb_preds, target_names=['Normal', 'AtRisk']))

    return {'ratio': ratio, 'ft_f1': ft_f1, 'ft_acc': ft_acc, 'xgb_f1': xgb_f1, 'xgb_acc': xgb_acc}


if __name__ == '__main__':
    results = [run_smote_ratio(r) for r in [1, 2, 3, 4]]

    print("\n\n" + "=" * 70)
    print("FINAL COMPARISON — SMOTE TARGET RATIOS (full dataset, no age/gender matching)")
    print("=" * 70)
    for r in results:
        print(f"1:{r['ratio']}  |  FT: F1={r['ft_f1']:.4f}, Acc={r['ft_acc']:.4f}  |  "
              f"XGB: F1={r['xgb_f1']:.4f}, Acc={r['xgb_acc']:.4f}")