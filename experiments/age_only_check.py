"""Quick diagnostic: how much predictive power does Age alone carry for the
binary AtRisk classification? Motivated by attention weights consistently
ranking Age as the top contributing feature — checking whether the model
is meaningfully using the other 11 biomarkers, or largely relying on Age."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

df = pd.read_csv('data/processed/labeled_dataset.csv')
df['Label'] = df['DiabetesStatus'].apply(lambda x: 'Normal' if x == 'Normal' else 'AtRisk')
label_map = {'Normal': 0, 'AtRisk': 1}

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['Label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Label'], random_state=42)

y_train = train_df['Label'].map(label_map).values
y_test = test_df['Label'].map(label_map).values

# --- Test 1: Age ALONE ---
X_train_age = train_df[['Age']].values
X_test_age = test_df[['Age']].values

model_age = LogisticRegression(class_weight='balanced')
model_age.fit(X_train_age, y_train)
preds_age = model_age.predict(X_test_age)

print("=" * 60)
print("AGE ALONE")
print("=" * 60)
print(f"Macro F1: {f1_score(y_test, preds_age, average='macro'):.4f}")
print(classification_report(y_test, preds_age, target_names=['Normal', 'AtRisk']))

# --- Test 2: All 12 biomarkers + Gender (your actual model's features) ---
NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]
for d in [train_df, test_df]:
    d['Gender_encoded'] = d['Gender'].map({'Male': 0, 'Female': 1})
feature_cols = NUMERIC_FEATURES + ['Gender_encoded']

X_train_full = train_df[feature_cols].values
X_test_full = test_df[feature_cols].values

model_full = LogisticRegression(class_weight='balanced', max_iter=1000)
model_full.fit(X_train_full, y_train)
preds_full = model_full.predict(X_test_full)

print("\n" + "=" * 60)
print("ALL 12 BIOMARKERS + GENDER")
print("=" * 60)
print(f"Macro F1: {f1_score(y_test, preds_full, average='macro'):.4f}")
print(classification_report(y_test, preds_full, target_names=['Normal', 'AtRisk']))

# --- Test 3: All biomarkers WITHOUT Age (isolate the other 11's contribution) ---
features_no_age = [f for f in NUMERIC_FEATURES if f != 'Age'] + ['Gender_encoded']
X_train_noage = train_df[features_no_age].values
X_test_noage = test_df[features_no_age].values

model_noage = LogisticRegression(class_weight='balanced', max_iter=1000)
model_noage.fit(X_train_noage, y_train)
preds_noage = model_noage.predict(X_test_noage)

print("\n" + "=" * 60)
print("ALL BIOMARKERS EXCEPT AGE")
print("=" * 60)
print(f"Macro F1: {f1_score(y_test, preds_noage, average='macro'):.4f}")
print(classification_report(y_test, preds_noage, target_names=['Normal', 'AtRisk']))

print("\n\nSUMMARY:")
print(f"Age alone:              {f1_score(y_test, preds_age, average='macro'):.4f}")
print(f"All 12 + Gender:        {f1_score(y_test, preds_full, average='macro'):.4f}")
print(f"All biomarkers, no Age: {f1_score(y_test, preds_noage, average='macro'):.4f}")