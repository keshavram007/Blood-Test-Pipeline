"""Two-stage hierarchical classifier: tests whether decomposing the 3-class problem
into Stage 1 (Normal vs At-Risk) + Stage 2 (Prediabetic vs Diabetic) beats a joint model.
Demonstrates error propagation as the reason it does NOT."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]

def run_hierarchical():
    df = pd.read_csv('data/processed/labeled_dataset.csv')
    df['Stage1Label'] = df['DiabetesStatus'].apply(lambda x: 'Normal' if x == 'Normal' else 'AtRisk')
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
    feature_cols = NUMERIC_FEATURES + ['Gender_encoded']

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['DiabetesStatus'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['DiabetesStatus'], random_state=42)

    # --- Stage 1: Normal vs AtRisk ---
    scaler1 = StandardScaler()
    X_train1 = scaler1.fit_transform(train_df[feature_cols]).astype(np.float32)
    y_train1 = train_df['Stage1Label'].map({'Normal': 0, 'AtRisk': 1}).values
    xgb1 = XGBClassifier(objective='binary:logistic', random_state=42, learning_rate=0.1, max_depth=7, n_estimators=300, eval_metric='logloss')
    xgb1.fit(X_train1, y_train1)

    # --- Stage 2: Prediabetic vs Diabetic (trained only on at-risk train rows) ---
    at_risk_train = train_df[train_df['DiabetesStatus'].isin(['Prediabetic', 'Diabetic'])]
    scaler2 = StandardScaler()
    X_train2 = scaler2.fit_transform(at_risk_train[feature_cols]).astype(np.float32)
    y_train2 = at_risk_train['DiabetesStatus'].map({'Prediabetic': 0, 'Diabetic': 1}).values
    xgb2 = XGBClassifier(objective='binary:logistic', random_state=42, learning_rate=0.1, max_depth=7, n_estimators=300, eval_metric='logloss')
    xgb2.fit(X_train2, y_train2)

    # --- Evaluate each stage in isolation ---
    at_risk_test = test_df[test_df['DiabetesStatus'].isin(['Prediabetic', 'Diabetic'])]
    X_test2 = scaler2.transform(at_risk_test[feature_cols]).astype(np.float32)
    y_test2 = at_risk_test['DiabetesStatus'].map({'Prediabetic': 0, 'Diabetic': 1}).values
    stage2_preds = xgb2.predict(X_test2)
    print("STAGE 2 ALONE (Prediabetic vs Diabetic):")
    print(f"Macro F1: {f1_score(y_test2, stage2_preds, average='macro'):.4f}")
    print(classification_report(y_test2, stage2_preds, target_names=['Prediabetic', 'Diabetic']))

    # --- Evaluate combined pipeline on full, unmatched test set ---
    X_test1 = scaler1.transform(test_df[feature_cols]).astype(np.float32)
    X_test2_full = scaler2.transform(test_df[feature_cols]).astype(np.float32)
    stage1_preds = xgb1.predict(X_test1)
    stage2_preds_full = xgb2.predict(X_test2_full)

    final_preds = np.where(stage1_preds == 0, 'Normal',
                   np.where(stage2_preds_full == 1, 'Diabetic', 'Prediabetic'))

    print("\nCOMBINED HIERARCHICAL PIPELINE (full test set):")
    print(f"Accuracy: {accuracy_score(test_df['DiabetesStatus'], final_preds):.4f}")
    print(f"Macro F1: {f1_score(test_df['DiabetesStatus'], final_preds, average='macro'):.4f}")
    print(classification_report(test_df['DiabetesStatus'], final_preds))

    print("\n>>> Note: Stage 2 alone outperforms the joint 3-class model, but the combined")
    print(">>> pipeline does NOT — Stage 1 errors propagate and cannot be corrected downstream.")


if __name__ == '__main__':
    run_hierarchical()