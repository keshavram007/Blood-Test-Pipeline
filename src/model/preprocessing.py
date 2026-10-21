import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

NUMERIC_FEATURES_CONSERVATIVE = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]

NUMERIC_FEATURES_FULL = NUMERIC_FEATURES_CONSERVATIVE + ['FastingGlucose', 'Glycated Haemoglobin']

LABEL_COL = 'DiabetesStatus'
LABEL_MAP = {'Normal': 0, 'Prediabetic': 1, 'Diabetic': 2}


def prepare_features(train_df, val_df, test_df, feature_set='conservative', scaler_path=None):
    numeric_features = NUMERIC_FEATURES_FULL if feature_set == 'full' else NUMERIC_FEATURES_CONSERVATIVE
    if scaler_path is None:
        scaler_path = f'outputs/models/scaler_{feature_set}.pkl'

    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()
    for df in [train_df, val_df, test_df]:
        df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[numeric_features])
    val_scaled = scaler.transform(val_df[numeric_features])
    test_scaled = scaler.transform(test_df[numeric_features])

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    def build_X_y(df, scaled_numeric):
        X_num = scaled_numeric.astype(np.float32)
        X_cat = df['Gender_encoded'].values.reshape(-1, 1).astype(np.float32)
        X = np.hstack([X_num, X_cat])
        y = df[LABEL_COL].map(LABEL_MAP).values.astype(np.int64)
        return X, y

    X_train, y_train = build_X_y(train_df, train_scaled)
    X_val, y_val = build_X_y(val_df, val_scaled)
    X_test, y_test = build_X_y(test_df, test_scaled)

    return X_train, y_train, X_val, y_val, X_test, y_test