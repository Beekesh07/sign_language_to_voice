import torch
import torch.nn as nn


class SignGRU(nn.Module):
    """
    Small bidirectional GRU. With only ~15 videos per word, a compact
    recurrent model generalises better than a bigger transformer.
    """

    def __init__(self, input_dim, num_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 192),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(192, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden * 4, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(self.input(x))
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.head(pooled)
