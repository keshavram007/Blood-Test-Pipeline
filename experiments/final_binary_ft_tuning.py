import itertools
import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.model.train import train_model

NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]

def prepare_binary_data():
    df = pd.read_csv('data/processed/labeled_dataset.csv')
    df['BinaryLabel'] = df['DiabetesStatus'].apply(lambda x: 'Normal' if x == 'Normal' else 'AtRisk')
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})

    feature_cols = NUMERIC_FEATURES + ['Gender_encoded']
    label_map = {'Normal': 0, 'AtRisk': 1}

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['BinaryLabel'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['BinaryLabel'], random_state=42)
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols]).astype(np.float32)
    X_val = scaler.transform(val_df[feature_cols]).astype(np.float32)
    X_test = scaler.transform(test_df[feature_cols]).astype(np.float32)

    y_train = train_df['BinaryLabel'].map(label_map).values.astype(np.int64)
    y_val = val_df['BinaryLabel'].map(label_map).values.astype(np.int64)
    y_test = test_df['BinaryLabel'].map(label_map).values.astype(np.int64)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    X_train_sm = X_train_sm.astype(np.float32)
    y_train_sm = y_train_sm.astype(np.int64)
    print(f"After SMOTE: {np.bincount(y_train_sm)}")

    return X_train_sm, y_train_sm, X_val, y_val, X_test, y_test, scaler


def run_grid_search():
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_binary_data()

    class_weights = torch.tensor(
        compute_class_weight('balanced', classes=np.unique(y_train), y=y_train),
        dtype=torch.float32
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rates = [1e-3, 5e-4]
    d_tokens = [16, 32]
    dropouts = [0.2, 0.3]
    n_layers_options = [2, 3]

    best_score = 0
    best_params = None
    best_model = None
    all_results = []

    for lr, d_token, dropout, n_layers in itertools.product(
        learning_rates, d_tokens, dropouts, n_layers_options
    ):
        print(f"\nCONFIG: lr={lr}, d_token={d_token}, dropout={dropout}, n_layers={n_layers}")

        model = train_model(
            X_train, y_train, X_val, y_val, class_weights,
            d_token=d_token, n_layers=n_layers, dropout=dropout, lr=lr,
            save_path=f'experiments/output/models/binary_grid_lr{lr}_dt{d_token}_do{dropout}_nl{n_layers}.pt',
            label_names=('Normal', 'AtRisk')
        )

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(torch.tensor(X_val).to(device)), dim=1).cpu().numpy()

        score = f1_score(y_val, preds, average='macro')
        all_results.append({'lr': lr, 'd_token': d_token, 'dropout': dropout,
                             'n_layers': n_layers, 'macro_f1': score})

        if score > best_score:
            best_score = score
            best_params = {'lr': lr, 'd_token': d_token, 'dropout': dropout, 'n_layers': n_layers}
            best_model = model

    print("\nALL RESULTS (sorted by macro F1):")
    for r in sorted(all_results, key=lambda x: x['macro_f1'], reverse=True):
        print(r)
    print(f"\nBest params: {best_params} | Best val macro F1: {best_score:.4f}")

    best_model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(best_model(torch.tensor(X_test).to(device)), dim=1).cpu().numpy()

    test_f1 = f1_score(y_test, test_preds, average='macro')
    print(f"\nTest Macro F1: {test_f1:.4f}")
    print(classification_report(y_test, test_preds, target_names=['Normal', 'AtRisk']))

    torch.save(best_model.state_dict(), 'experiments/output/models/final_binary_ft_model.pt')
    joblib.dump(scaler, 'experiments/output/models/final_binary_scaler.pkl')
    print("\nSaved final model and scaler to experiments/output/models/.")

    return best_model, best_params, test_f1


if __name__ == '__main__':
    run_grid_search()