# experiments/compute_anomaly_thresholds.py
import os
import pandas as pd
import numpy as np
import torch
import joblib
from src.model.reconstruction_head import ReconstructionAutoencoder

NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]
FEATURE_COLS = NUMERIC_FEATURES + ['Gender_encoded']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


test_df = pd.read_csv('data/processed/test.csv')
test_df['Gender_encoded'] = test_df['Gender'].map({'Male': 0, 'Female': 1})

scaler = joblib.load('outputs/models/reconstruction_scaler.pkl')
X_test = scaler.transform(test_df[FEATURE_COLS]).astype('float32')

model = ReconstructionAutoencoder(n_features=13, hidden_dim=32, bottleneck_dim=8)
model.load_state_dict(torch.load('outputs/models/reconstruction_autoencoder.pt',
                                  map_location=device))
model.to(device).eval()

with torch.no_grad():
    X_tensor = torch.tensor(X_test).to(device)
    reconstructed = model(X_tensor)
    scores = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

test_df = test_df.copy()
test_df['AnomalyScore'] = scores

print("Anomaly score distribution by class:")
print(test_df.groupby('DiabetesStatus')['AnomalyScore'].describe())

normal_95th = test_df[test_df['DiabetesStatus'] == 'Normal']['AnomalyScore'].quantile(0.95)
prediabetic_50th = test_df[test_df['DiabetesStatus'] == 'Prediabetic']['AnomalyScore'].quantile(0.50)
diabetic_25th = test_df[test_df['DiabetesStatus'] == 'Diabetic']['AnomalyScore'].quantile(0.25)

print(f"\nPercentile-based suggestion (reference only — NOT used below,")
print(f"since Normal's heavy-tailed outliers make its 95th percentile unreliable):")
print(f"Normal 95th percentile:     {normal_95th:.4f}")
print(f"Prediabetic median (50th):  {prediabetic_50th:.4f}")
print(f"Diabetic 25th percentile:   {diabetic_25th:.4f}")

# Median-based cutoffs: medians are robust to the extreme outliers present in
# the Normal class (max anomaly score 77.16 vs. a median of ~0.10), unlike the
# 95th-percentile approach above. Each cutoff is the midpoint between two
# adjacent classes' medians.
normal_median = test_df[test_df['DiabetesStatus'] == 'Normal']['AnomalyScore'].median()
prediabetic_median = test_df[test_df['DiabetesStatus'] == 'Prediabetic']['AnomalyScore'].median()
diabetic_median = test_df[test_df['DiabetesStatus'] == 'Diabetic']['AnomalyScore'].median()

lower_cutoff = (normal_median + prediabetic_median) / 2
upper_cutoff = (prediabetic_median + diabetic_median) / 2

print(f"\nMedian-based cutoffs (used in _interpret_anomaly()):")
print(f"Normal median:      {normal_median:.4f}")
print(f"Prediabetic median: {prediabetic_median:.4f}")
print(f"Diabetic median:    {diabetic_median:.4f}")
print(f"Lower cutoff (mid Normal/Prediabetic):      {lower_cutoff:.4f}")
print(f"Upper cutoff (mid Prediabetic/Diabetic):    {upper_cutoff:.4f}")

os.makedirs('outputs/experiments/logs', exist_ok=True)
test_df.to_csv('outputs/experiments/logs/anomaly_scores_test.csv', index=False)
print("\nSaved: outputs/experiments/logs/anomaly_scores_test.csv")
