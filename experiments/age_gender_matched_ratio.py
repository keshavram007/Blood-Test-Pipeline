"""Age/gender-matched ratio experiment.

Motivation: attention weights consistently rank Age as the top contributing
feature (see experiments/age_only_check.py). That check compares Age-alone
vs. full-biomarker-panel performance on the ordinary train/test split, but
in that split Diabetic patients are also, on average, older than Normal
patients — so some of Age's apparent power could just be reflecting the
real age skew between classes rather than the biomarkers being weak.

This script controls for that by matching: for every Diabetic patient, it
draws Normal patients of the SAME Age and SAME Gender only, at four
different class-imbalance ratios (1 Diabetic : 4 / 3 / 2 / 1 Normal). Age
is therefore identical within every matched group, so it carries zero
information for telling Diabetic apart from Normal in that group. Training
a classifier on the biomarker panel alone (Age excluded) on these matched
sets shows how much separating power the biomarkers themselves have once
age is no longer a usable shortcut; a control run that re-adds Age is
included to confirm it does not help once matching has neutralised it.

Usage: python -m experiments.age_gender_matched_ratio
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]
BIOMARKERS_NO_AGE = [f for f in NUMERIC_FEATURES if f != 'Age']

RATIOS = [4, 3, 2, 1]  # Normal patients per Diabetic patient
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv('data/processed/labeled_dataset.csv')
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
    return df


def build_matched_set(df, ratio, rng):
    """For every Diabetic patient, sample up to `ratio` Normal patients with
    the same exact Age and Gender, without reusing a Normal patient more
    than once within this matched set. Diabetic patients for whom no exact
    age/gender match exists are dropped and counted."""
    diabetic = df[df['DiabetesStatus'] == 'Diabetic'].copy()
    normal_pool = df[df['DiabetesStatus'] == 'Normal'].copy()

    used_normal_ids = set()
    matched_rows = []
    unmatched_diabetic = 0

    diabetic = diabetic.sample(frac=1.0, random_state=rng.integers(1e9))  # shuffle order

    for _, d_row in diabetic.iterrows():
        candidates = normal_pool[
            (normal_pool['Age'] == d_row['Age']) &
            (normal_pool['Gender'] == d_row['Gender']) &
            (~normal_pool['PatientID'].isin(used_normal_ids))
        ]
        if len(candidates) == 0:
            unmatched_diabetic += 1
            continue

        take = candidates.sample(n=min(ratio, len(candidates)), random_state=rng.integers(1e9))
        used_normal_ids.update(take['PatientID'].tolist())

        matched_rows.append(d_row)
        matched_rows.extend([row for _, row in take.iterrows()])

    matched_df = pd.DataFrame(matched_rows)
    return matched_df, unmatched_diabetic


def evaluate(matched_df, feature_cols, run_label):
    label_map = {'Normal': 0, 'Diabetic': 1}
    y = matched_df['DiabetesStatus'].map(label_map).values
    X = matched_df[feature_cols].values

    # Some matched sets are small, especially at ratio 1:4 where exact
    # age/gender matches are scarcer — stratified split still applies since
    # every ratio keeps at least a handful of Diabetic patients.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    f1 = f1_score(y_test, preds, average='macro')
    acc = accuracy_score(y_test, preds)

    print(f"\n--- {run_label} ---")
    print(f"Macro F1: {f1:.4f} | Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=['Normal', 'Diabetic']))
    return f1, acc


def main():
    df = load_data()
    rng = np.random.default_rng(RANDOM_STATE)

    summary = []

    for ratio in RATIOS:
        print("\n\n" + "#" * 70)
        print(f"# AGE/GENDER-MATCHED SET — 1 DIABETIC : {ratio} NORMAL")
        print("#" * 70)

        matched_df, unmatched = build_matched_set(df, ratio, rng)
        n_diabetic = (matched_df['DiabetesStatus'] == 'Diabetic').sum()
        n_normal = (matched_df['DiabetesStatus'] == 'Normal').sum()
        print(f"Matched Diabetic patients: {n_diabetic} "
              f"(dropped {unmatched} with no exact Age/Gender match)")
        print(f"Matched Normal patients:   {n_normal}")

        f1_full, acc_full = evaluate(
            matched_df, BIOMARKERS_NO_AGE + ['Gender_encoded'],
            "Biomarkers only (Age excluded)"
        )
        f1_age, acc_age = evaluate(
            matched_df, NUMERIC_FEATURES + ['Gender_encoded'],
            "Biomarkers + Age (control — Age should add ~nothing once matched)"
        )

        summary.append({
            'ratio': ratio,
            'n_diabetic': n_diabetic,
            'n_normal': n_normal,
            'unmatched_diabetic': unmatched,
            'f1_no_age': f1_full,
            'acc_no_age': acc_full,
            'f1_with_age': f1_age,
            'acc_with_age': acc_age,
        })

    print("\n\n" + "=" * 70)
    print("SUMMARY — AGE/GENDER-MATCHED RATIO EXPERIMENT")
    print("=" * 70)
    for s in summary:
        print(
            f"1:{s['ratio']}  |  n_diabetic={s['n_diabetic']} (dropped {s['unmatched_diabetic']})  "
            f"n_normal={s['n_normal']}  |  "
            f"No-Age: F1={s['f1_no_age']:.4f}, Acc={s['acc_no_age']:.4f}  |  "
            f"+Age:   F1={s['f1_with_age']:.4f}, Acc={s['acc_with_age']:.4f}"
        )


if __name__ == '__main__':
    main()
