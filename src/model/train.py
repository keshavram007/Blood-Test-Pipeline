import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report

from src.model.ft_transformer import FTTransformer


def train_model(X_train, y_train, X_val, y_val, class_weights,
                 n_epochs=50, lr=1e-3, batch_size=64,
                 d_token=16, n_layers=2, dropout=0.3, patience=10,
                 save_path='outputs/models/ft_transformer_best.pt',
                 label_names=('Normal', 'Prediabetic', 'Diabetic')):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FTTransformer(n_features=X_train.shape[1], n_classes=len(label_names),
                           d_token=d_token, n_layers=n_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{n_epochs} | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(save_path))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.cpu().numpy())

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print("Final validation classification report (BEST checkpoint):")
    print(classification_report(all_true, all_preds, target_names=list(label_names)))

    return model