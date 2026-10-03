import torch
import torch.nn as nn


class SignTransformer(nn.Module):

    def __init__(
        self,
        input_dim=126,
        d_model=128,
        num_heads=4,
        num_layers=3,
        num_classes=30,
        dropout=0.2
    ):

        super().__init__()

        # 126 landmark values -> 128 transformer features
        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        # Learnable positional information
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

        self.classifier = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, x):

        # x shape:
        # batch, 30 frames, 126 features

        x = self.input_projection(x)

        x = x + self.position_embedding

        x = self.transformer(x)

        # Average all 30 frames
        x = x.mean(dim=1)

        x = self.classifier(x)

        return x