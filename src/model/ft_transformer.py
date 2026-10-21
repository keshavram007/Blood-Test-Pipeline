import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.randn(n_features, d_token) * 0.02)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class AttentionCapturingEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.last_attn_weights = None

    def forward(self, src):
                                                   #query,key,value
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True, average_attn_weights=True)
        self.last_attn_weights = attn_weights.detach() # dont use for gradient calculations
        src = self.norm1(src + self.dropout1(attn_output))
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src


class FTTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_token=32, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        self.layers = nn.ModuleList([
            AttentionCapturingEncoderLayer(d_token, n_heads, d_token * 4, dropout)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, n_classes)
        )

    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for layer in self.layers:
            tokens = layer(tokens)
        cls_output = tokens[:, 0, :]
        return self.classifier(cls_output)

    def get_attention_weights(self):
        return self.layers[-1].last_attn_weights