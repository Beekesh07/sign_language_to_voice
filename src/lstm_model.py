import torch
import torch.nn as nn


class SignLSTM(nn.Module):

    def __init__(
        self,
        input_dim=225,
        hidden_dim=128,
        num_layers=2,
        num_classes=30,
        dropout=0.3
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim,
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
        # batch, 30 frames, 225 features

        output, (hidden, cell) = self.lstm(x)

        # last LSTM layer's final hidden state
        x = hidden[-1]

        x = self.classifier(x)

        return x