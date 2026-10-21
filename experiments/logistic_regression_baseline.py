"""Logistic Regression baseline — the standard, simplest possible model, used to
establish a performance floor that more complex models (FT-Transformer, XGBoost,
TabNet) should meaningfully beat."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score

NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]


def prepare_data(binary):
    df = pd.read_csv('data/processed/labeled_dataset.csv')
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
    feature_cols = NUMERIC_FEATURES + ['Gender_encoded']

    if binary:
        df['Label'] = df['DiabetesStatus'].apply(lambda x: 'Normal' if x == 'Normal' else 'AtRisk')
        label_map = {'Normal': 0, 'AtRisk': 1}
        label_names = ['Normal', 'AtRisk']
    else:
        df['Label'] = df['DiabetesStatus']
        label_map = {'Normal': 0, 'Prediabetic': 1, 'Diabetic': 2}
        label_names = ['Normal', 'Prediabetic', 'Diabetic']

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['Label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Label'], random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    y_train = train_df['Label'].map(label_map).values
    y_test = test_df['Label'].map(label_map).values

    return X_train, y_train, X_test, y_test, label_names


def run_logistic_regression(binary):
    task_name = "BINARY (Normal vs At-Risk)" if binary else "3-CLASS"
    print("\n" + "=" * 60)
    print(f"LOGISTIC REGRESSION BASELINE — {task_name}")
    print("=" * 60)

    X_train, y_train, X_test, y_test, label_names = prepare_data(binary)

    # class_weight='balanced' handles imbalance automatically — standard practice
    # for a baseline, avoids needing separate SMOTE step
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    acc = accuracy_score(y_test, preds)

    print(f"\nMacro F1: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=label_names))

    from sklearn.metrics import precision_score, recall_score

    precision = precision_score(y_test, preds, average=None)
    recall = recall_score(y_test, preds, average=None)  # this IS sensitivity

    print(f"\nPer-class Precision: {dict(zip(label_names, precision.round(3)))}")
    print(f"Per-class Recall (Sensitivity): {dict(zip(label_names, recall.round(3)))}")

    if binary:
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        print(f"AUC-ROC: {auc:.4f}")
        return f1, acc, auc

    return f1, acc, None


if __name__ == '__main__':
    f1_3class, acc_3class, _ = run_logistic_regression(binary=False)
    f1_binary, acc_binary, auc_binary = run_logistic_regression(binary=True)

    print("\n\n" + "=" * 60)
    print("LOGISTIC REGRESSION BASELINE SUMMARY")
    print("=" * 60)
    print(f"3-class | Macro F1: {f1_3class:.4f}, Accuracy: {acc_3class:.4f}")
    print(f"Binary  | Macro F1: {f1_binary:.4f}, Accuracy: {acc_binary:.4f}, AUC-ROC: {auc_binary:.4f}")
    print("\n>>> Final models should be compared against this baseline to justify")
    print(">>> the added complexity of FT-Transformer / XGBoost / TabNet.")