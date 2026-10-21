import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from src.model.reconstruction_head import ReconstructionAutoencoder

NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]
FEATURE_COLS = NUMERIC_FEATURES + ['Gender_encoded']

MODEL_PATH = 'outputs/models/reconstruction_autoencoder.pt'
SCALER_PATH = 'outputs/models/reconstruction_scaler.pkl'


def prepare_data():
    df = pd.read_csv('data/processed/labeled_dataset.csv')
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['DiabetesStatus'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['DiabetesStatus'], random_state=42)

    # 1. Filter BOTH train and val to Normal patients for autoencoder objective
    train_normal = train_df[train_df['DiabetesStatus'] == 'Normal']
    val_normal = val_df[val_df['DiabetesStatus'] == 'Normal']

    scaler = StandardScaler()
    X_train_normal = scaler.fit_transform(train_normal[FEATURE_COLS]).astype(np.float32)
    X_val_normal = scaler.transform(val_normal[FEATURE_COLS]).astype(np.float32)
    X_test = scaler.transform(test_df[FEATURE_COLS]).astype(np.float32)

    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved: {SCALER_PATH}")

    return X_train_normal, X_val_normal, X_test, test_df


def train_autoencoder(X_train, X_val, n_epochs=100, lr=1e-3, batch_size=64,
                      hidden_dim=32, bottleneck_dim=8, patience=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = TensorDataset(torch.tensor(X_train))
    val_ds = TensorDataset(torch.tensor(X_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = ReconstructionAutoencoder(
        n_features=X_train.shape[1],
        hidden_dim=hidden_dim, 
        bottleneck_dim=bottleneck_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            reconstructed = model(xb)
            loss = loss_fn(reconstructed, xb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                reconstructed = model(xb)
                loss = loss_fn(reconstructed, xb)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"\nBest val MSE: {best_val_loss:.4f}")
    print(f"Saved: {MODEL_PATH}")
    return model


def compute_anomaly_scores(model, X, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X).to(device)
        reconstructed = model(X_tensor)
        mse_per_row = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
    return mse_per_row.cpu().numpy()


if __name__ == '__main__':
    print("=" * 60)
    print("RECONSTRUCTION HEAD — Training on Normal patients only")
    print("=" * 60)

    X_train_normal, X_val_normal, X_test, test_df = prepare_data()
    print(f"\nTraining on {len(X_train_normal)} Normal patients (Val: {len(X_val_normal)} Normal)")

    model = train_autoencoder(X_train_normal, X_val_normal)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_scores = compute_anomaly_scores(model, X_test, device)
    test_df = test_df.copy()
    test_df['AnomalyScore'] = test_scores

    print("\n" + "=" * 60)
    print("VERIFICATION: Anomaly score summary statistics")
    print("=" * 60)
    print(test_df.groupby('DiabetesStatus')['AnomalyScore'].describe())

    # Evaluation using robust central tendency (Median)
    medians = test_df.groupby('DiabetesStatus')['AnomalyScore'].median()
    print("\nMedian Anomaly Scores:")
    print(medians)

    if medians['Normal'] < medians['Prediabetic'] < medians['Diabetic']:
        print("\n✅ PASS: Normal < Prediabetic < Diabetic (by median) — reconstruction head working correctly.")
    else:
        print("\n❌ WARNING: Expected monotonic pattern not found — investigate before using in production.")