import os
import json
import numpy as np
import torch

from sklearn.metrics import confusion_matrix, classification_report

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
    "best_transformer_v2.pth"
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
# LOAD DATA
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


id_to_label = {
    value: key
    for key, value in label_to_id.items()
}

NUM_CLASSES = len(label_to_id)


# -------------------------
# MODEL
# -------------------------

model = SignTransformerV2(
    input_dim=225,
    d_model=128,
    num_heads=4,
    num_layers=3,
    num_classes=NUM_CLASSES,
    dropout=0.3
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# -------------------------
# PREDICT
# -------------------------

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    outputs = model(
        X_test_tensor
    )

    predictions = torch.argmax(
        outputs,
        dim=1
    ).cpu().numpy()


# -------------------------
# ACCURACY
# -------------------------

correct = np.sum(
    predictions == y_test
)

accuracy = (
    correct / len(y_test)
) * 100


print(
    "\nOverall Test Accuracy:",
    f"{accuracy:.2f}%"
)


# -------------------------
# CLASSIFICATION REPORT
# -------------------------

target_names = [
    id_to_label[i]
    for i in range(NUM_CLASSES)
]

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        predictions,
        target_names=target_names,
        zero_division=0
    )
)


# -------------------------
# CONFUSION MATRIX
# -------------------------

cm = confusion_matrix(
    y_test,
    predictions
)


print(
    "\nConfusion Matrix:\n"
)

print(cm)


# -------------------------
# WRONG PREDICTIONS
# -------------------------

print(
    "\nWrong Predictions:\n"
)

wrong_indices = np.where(
    predictions != y_test
)[0]


for index in wrong_indices:

    actual = id_to_label[
        int(y_test[index])
    ]

    predicted = id_to_label[
        int(predictions[index])
    ]

    print(
        f"Sample {index}: "
        f"Actual = {actual} | "
        f"Predicted = {predicted}"
    )