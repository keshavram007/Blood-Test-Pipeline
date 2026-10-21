import pandas as pd
import numpy as np
import torch
import joblib
import shap

from src.model.ft_transformer import FTTransformer
from src.model.reconstruction_head import ReconstructionAutoencoder

NUMERIC_FEATURES = [
    'Age', 'Triglycerides', 'TotalCholesterol', 'UricAcid', 'Albumin',
    'Creatinine', 'Alanine Aminotransferase', 'HDLCholesterol',
    'Sodium', 'BUN', 'GGT', 'AST'
]
FEATURE_COLS = NUMERIC_FEATURES + ['Gender_encoded']
LABEL_NAMES = ('Normal', 'AtRisk')

CLASSIFIER_PATH = 'outputs/models/final_binary_ft.pt'
CLASSIFIER_SCALER_PATH = 'outputs/models/final_binary_scaler.pkl'
AUTOENCODER_PATH = 'outputs/models/reconstruction_autoencoder.pt'
AUTOENCODER_SCALER_PATH = 'outputs/models/reconstruction_scaler.pkl'

SHAP_BACKGROUND_SIZE = 30
SHAP_KMEANS_CLUSTERS = 15
SHAP_NSAMPLES = 50


class BloodTestPredictor:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classifier = FTTransformer(
            n_features=len(FEATURE_COLS), n_classes=len(LABEL_NAMES),
            d_token=32, n_layers=3, dropout=0.3)
        self.classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=self.device))
        self.classifier.to(self.device).eval()
        self.classifier_scaler = joblib.load(CLASSIFIER_SCALER_PATH)

        self.autoencoder = ReconstructionAutoencoder(
            n_features=len(FEATURE_COLS), hidden_dim=32, bottleneck_dim=8)
        self.autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH, map_location=self.device))
        self.autoencoder.to(self.device).eval()
        self.autoencoder_scaler = joblib.load(AUTOENCODER_SCALER_PATH)

        self._shap_explainer = None

    def _predict_fn_for_shap(self, X_numpy):
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_numpy), 64):
                xb = torch.tensor(X_numpy[i:i+64], dtype=torch.float32).to(self.device)
                logits = self.classifier(xb)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds.append(probs)
        return np.concatenate(preds)

    def _build_shap_explainer(self):
        df = pd.read_csv('data/processed/train.csv')
        df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
        background_raw = df[FEATURE_COLS].sample(n=SHAP_BACKGROUND_SIZE, random_state=42)
        X_background = self.classifier_scaler.transform(background_raw).astype(np.float32)
        background_summary = shap.kmeans(X_background, SHAP_KMEANS_CLUSTERS)

        self._shap_explainer = shap.KernelExplainer(self._predict_fn_for_shap, background_summary)

    def get_shap_values(self, X_scaled_row):
        if self._shap_explainer is None:
            self._build_shap_explainer()

        shap_vals = self._shap_explainer.shap_values(X_scaled_row, nsamples=SHAP_NSAMPLES)
        shap_vals = np.array(shap_vals).flatten()
        return dict(zip(FEATURE_COLS, shap_vals.tolist()))

    def predict(self, patient):
        row = pd.DataFrame([{
            **patient,
            'Gender_encoded': 0 if patient['Gender'] == 'Male' else 1,
        }])
        x_row = row[FEATURE_COLS]

        X_cls = self.classifier_scaler.transform(x_row).astype(np.float32)
        with torch.no_grad():
            xb = torch.tensor(X_cls).to(self.device)
            logits = self.classifier(xb)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        risk_idx = int(np.argmax(probs))
        shap_importance = self.get_shap_values(X_cls)

        X_ae = self.autoencoder_scaler.transform(x_row).astype(np.float32)
        with torch.no_grad():
            xb_ae = torch.tensor(X_ae).to(self.device)
            reconstructed = self.autoencoder(xb_ae)
            anomaly_score = float(torch.mean((xb_ae - reconstructed) ** 2).item())

        return {
            'risk_label': LABEL_NAMES[risk_idx],
            'risk_probability': float(probs[risk_idx]),
            'class_probabilities': dict(zip(LABEL_NAMES, probs.tolist())),
            'anomaly_score': anomaly_score,
            'shap_importance': shap_importance
        }