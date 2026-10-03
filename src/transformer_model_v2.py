import torch
import torch.nn as nn


class SignTransformerV2(nn.Module):

    def __init__(
        self,
        input_dim=225,
        d_model=128,
        num_heads=4,
        num_layers=3,
        num_classes=30,
        dropout=0.3
    ):

        super().__init__()

        # 225 input features -> 128 transformer features
        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        # Learnable positional embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, 30, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                d_model,
                128
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        # x:
        # (batch, 30, 225)

        x = self.input_projection(x)

        x = x + self.position_embedding

        x = self.transformer(x)

        x = self.norm(x)

        # Average across 30 frames
        x = x.mean(dim=1)

        x = self.classifier(x)

        return x