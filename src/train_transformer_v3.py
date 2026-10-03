import os
import json
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from transformer_model_v2 import SignTransformerV2


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAINING_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "training_v2"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 1e-4

PATIENCE = 12


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using device:",
    device
)


# ============================================================
# LOAD DATA
# ============================================================

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


# ============================================================
# LOAD LABELS
# ============================================================

with open(
    os.path.join(
        TRAINING_DIR,
        "labels.json"
    ),
    "r"
) as file:

    label_to_id = json.load(file)


NUM_CLASSES = len(
    label_to_id
)


print(
    "Number of classes:",
    NUM_CLASSES
)

print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)


# ============================================================
# AUGMENTATION
# ============================================================

def augment_sequence(sequence):

    sequence = sequence.copy()

    # --------------------------------------------------------
    # 1. Small Gaussian noise
    # --------------------------------------------------------

    if np.random.rand() < 0.5:

        noise = np.random.normal(
            loc=0.0,
            scale=0.01,
            size=sequence.shape
        ).astype(np.float32)

        sequence += noise


    # --------------------------------------------------------
    # 2. Small scaling
    # --------------------------------------------------------

    if np.random.rand() < 0.5:

        scale = np.random.uniform(
            0.95,
            1.05
        )

        sequence *= scale


    # --------------------------------------------------------
    # 3. Temporal shift
    # --------------------------------------------------------

    if np.random.rand() < 0.5:

        shift = np.random.randint(
            -2,
            3
        )

        sequence = np.roll(
            sequence,
            shift,
            axis=0
        )


    # --------------------------------------------------------
    # 4. Random frame dropout
    # --------------------------------------------------------

    if np.random.rand() < 0.3:

        frame_index = np.random.randint(
            0,
            sequence.shape[0]
        )

        if frame_index > 0:

            sequence[frame_index] = sequence[
                frame_index - 1
            ]


    return sequence.astype(
        np.float32
    )


# ============================================================
# CUSTOM DATASET
# ============================================================

class SignDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        augment=False
    ):

        self.X = X
        self.y = y
        self.augment = augment


    def __len__(self):

        return len(
            self.X
        )


    def __getitem__(
        self,
        index
    ):

        sample = self.X[
            index
        ]

        label = self.y[
            index
        ]

        if self.augment:

            sample = augment_sequence(
                sample
            )

        sample = torch.tensor(
            sample,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return sample, label


# ============================================================
# DATASETS
# ============================================================

train_dataset = SignDataset(
    X_train,
    y_train,
    augment=True
)

val_dataset = SignDataset(
    X_val,
    y_val,
    augment=False
)


# ============================================================
# DATALOADERS
# ============================================================

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


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)

print(
    "\nClass counts:"
)

print(
    class_counts
)


class_weights = (
    len(y_train)
    /
    (
        NUM_CLASSES
        *
        class_counts
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


print(
    "\nClass weights:"
)

print(
    class_weights
)


# ============================================================
# MODEL
# ============================================================

model = SignTransformerV2(
    input_dim=225,
    d_model=128,
    num_heads=4,
    num_layers=3,
    num_classes=NUM_CLASSES,
    dropout=0.3
)

model = model.to(
    device
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)


# ============================================================
# BEST MODEL + EARLY STOPPING
# ============================================================

best_val_accuracy = 0.0

epochs_without_improvement = 0

best_model_path = os.path.join(
    MODEL_DIR,
    "best_transformer_v3.pth"
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    EPOCHS
):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )


        optimizer.zero_grad()


        outputs = model(
            X_batch
        )


        loss = criterion(
            outputs,
            y_batch
        )


        loss.backward()


        # Prevent extreme gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


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


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )


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


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(
        val_accuracy
    )


    current_lr = optimizer.param_groups[
        0
    ]["lr"]


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.2f}% "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.2f}% "
        f"| LR: {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        epochs_without_improvement = 0


        torch.save(
            model.state_dict(),
            best_model_path
        )


        print(
            f"Best V3 model saved: "
            f"{best_val_accuracy:.2f}%"
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# FINISHED
# ============================================================

print(
    "\nV3 training complete."
)

print(
    "Best validation accuracy:",
    f"{best_val_accuracy:.2f}%"
)

print(
    "Saved model:",
    best_model_path
)