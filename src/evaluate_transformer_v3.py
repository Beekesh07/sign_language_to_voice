import os
import json
import numpy as np
import torch

from transformer_model_v2 import SignTransformerV2


# -------------------------
# PATHS
# -------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAINING_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "training_v2"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_transformer_v3.pth"
)


# -------------------------
# DEVICE
# -------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# -------------------------
# LOAD TEST DATA
# -------------------------

X_test = np.load(
    os.path.join(
        TRAINING_DIR,
        "X_test.npy"
    )
)

y_test = np.load(
    os.path.join(
        TRAINING_DIR,
        "y_test.npy"
    )
)


# -------------------------
# LOAD LABELS
# -------------------------

with open(
    os.path.join(
        TRAINING_DIR,
        "labels.json"
    ),
    "r"
) as file:

    label_to_id = json.load(file)


NUM_CLASSES = len(label_to_id)


# -------------------------
# CREATE MODEL
# -------------------------

model = SignTransformerV2(
    input_dim=225,
    d_model=128,
    num_heads=4,
    num_layers=3,
    num_classes=NUM_CLASSES,
    dropout=0.3
)


# -------------------------
# LOAD V3 MODEL
# -------------------------

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# -------------------------
# TENSOR
# -------------------------

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
).to(device)


# -------------------------
# PREDICTION
# -------------------------

with torch.no_grad():

    outputs = model(
        X_test
    )

    predictions = torch.argmax(
        outputs,
        dim=1
    )


# -------------------------
# ACCURACY
# -------------------------

correct = (
    predictions == y_test
).sum().item()

total = y_test.size(0)

accuracy = (
    correct / total
) * 100


print(
    "\nTest samples:",
    total
)

print(
    "Correct predictions:",
    correct
)

print(
    "V3 Test accuracy:",
    f"{accuracy:.2f}%"
)