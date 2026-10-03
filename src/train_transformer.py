import os
import json
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from transformer_model import SignTransformer


# ------------------------------------------------
# PATHS
# ------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAINING_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "training"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 0.001


# ------------------------------------------------
# DEVICE
# ------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

X_train = np.load(
    os.path.join(
        TRAINING_DIR,
        "X_train.npy"
    )
)

y_train = np.load(
    os.path.join(
        TRAINING_DIR,
        "y_train.npy"
    )
)

X_val = np.load(
    os.path.join(
        TRAINING_DIR,
        "X_val.npy"
    )
)

y_val = np.load(
    os.path.join(
        TRAINING_DIR,
        "y_val.npy"
    )
)


# ------------------------------------------------
# LOAD LABELS
# ------------------------------------------------

with open(
    os.path.join(
        TRAINING_DIR,
        "labels.json"
    ),
    "r"
) as file:

    label_to_id = json.load(file)


NUM_CLASSES = len(label_to_id)

print(
    "Number of classes:",
    NUM_CLASSES
)

print(
    "Train shape:",
    X_train.shape
)

print(
    "Validation shape:",
    X_val.shape
)


# ------------------------------------------------
# NUMPY -> TORCH
# ------------------------------------------------

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.long
)


# ------------------------------------------------
# DATASETS
# ------------------------------------------------

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)


# ------------------------------------------------
# DATA LOADERS
# ------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ------------------------------------------------
# MODEL
# ------------------------------------------------

model = SignTransformer(
    input_dim=126,
    d_model=128,
    num_heads=4,
    num_layers=3,
    num_classes=NUM_CLASSES,
    dropout=0.2
)

model = model.to(device)


# ------------------------------------------------
# LOSS
# ------------------------------------------------

criterion = nn.CrossEntropyLoss()


# ------------------------------------------------
# OPTIMIZER
# ------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ------------------------------------------------
# BEST MODEL TRACKING
# ------------------------------------------------

best_val_accuracy = 0.0

best_model_path = os.path.join(
    MODEL_DIR,
    "best_transformer.pth"
)


# ------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------

for epoch in range(EPOCHS):

    # =========================
    # TRAINING
    # =========================

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)


        # Clear old gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(
            X_batch
        )


        # Calculate loss
        loss = criterion(
            outputs,
            y_batch
        )


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        train_loss += (
            loss.item()
            *
            X_batch.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        train_correct += (
            predictions
            ==
            y_batch
        ).sum().item()


        train_total += (
            y_batch.size(0)
        )


    train_loss = (
        train_loss
        /
        train_total
    )

    train_accuracy = (
        train_correct
        /
        train_total
        *
        100
    )


    # =========================
    # VALIDATION
    # =========================

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)

            y_batch = y_batch.to(device)


            outputs = model(
                X_batch
            )


            loss = criterion(
                outputs,
                y_batch
            )


            val_loss += (
                loss.item()
                *
                X_batch.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_correct += (
                predictions
                ==
                y_batch
            ).sum().item()


            val_total += (
                y_batch.size(0)
            )


    val_loss = (
        val_loss
        /
        val_total
    )

    val_accuracy = (
        val_correct
        /
        val_total
        *
        100
    )


    # =========================
    # PRINT RESULTS
    # =========================

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.2f}% "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.2f}%"
    )


    # =========================
    # SAVE BEST MODEL
    # =========================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            f"Best model saved "
            f"({best_val_accuracy:.2f}%)"
        )


# ------------------------------------------------
# FINISHED
# ------------------------------------------------

print("\nTraining complete.")

print(
    "Best validation accuracy:",
    f"{best_val_accuracy:.2f}%"
)

print(
    "Model saved at:",
    best_model_path
)