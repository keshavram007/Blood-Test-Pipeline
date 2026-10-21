import torch.nn as nn

class ReconstructionAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=32, bottleneck_dim=8, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))