import os
import json
import numpy as np
from sklearn.model_selection import train_test_split


# -------------------------
# PATHS
# -------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "training"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# -------------------------
# CONFIG
# -------------------------

MIN_SAMPLES = 5


# -------------------------
# FIND VALID LABELS
# -------------------------

labels = []

for folder in sorted(os.listdir(PROCESSED_DIR)):

    folder_path = os.path.join(
        PROCESSED_DIR,
        folder
    )

    if not os.path.isdir(folder_path):
        continue

    if folder.startswith("."):
        continue

    npy_files = [
        file
        for file in os.listdir(folder_path)
        if file.endswith(".npy")
    ]

    if len(npy_files) >= MIN_SAMPLES:

        labels.append(folder)

    else:

        print(
            f"Skipping '{folder}' "
            f"because it has only "
            f"{len(npy_files)} samples"
        )


# -------------------------
# CREATE LABEL MAPPING
# -------------------------

label_to_id = {
    label: index
    for index, label in enumerate(labels)
}

print("\nTotal usable labels:", len(labels))

print("\nLabel mapping:")

for label, label_id in label_to_id.items():

    print(
        label_id,
        "->",
        label
    )


# -------------------------
# LOAD DATA
# -------------------------

X = []
y = []


for label in labels:

    folder_path = os.path.join(
        PROCESSED_DIR,
        label
    )

    files = [
        file
        for file in os.listdir(folder_path)
        if file.endswith(".npy")
    ]

    print(
        f"\nLoading {label}: "
        f"{len(files)} samples"
    )

    for file in files:

        file_path = os.path.join(
            folder_path,
            file
        )

        sample = np.load(
            file_path
        )

        # Safety check
        if sample.shape != (30, 126):

            print(
                f"Skipping invalid shape: "
                f"{file_path} "
                f"{sample.shape}"
            )

            continue

        X.append(sample)

        y.append(
            label_to_id[label]
        )


# -------------------------
# CONVERT TO NUMPY
# -------------------------

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int64
)


print("\nFull dataset:")

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# -------------------------
# FIRST SPLIT
#
# 80% train+validation
# 20% test
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -------------------------
# SECOND SPLIT
#
# From remaining 80%:
# 80% training
# 20% validation
#
# Final approximately:
#
# 64% training
# 16% validation
# 20% testing
# -------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)


# -------------------------
# SAVE DATA
# -------------------------

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_train.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_train.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_val.npy"
    ),
    X_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_val.npy"
    ),
    y_val
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_test.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_test.npy"
    ),
    y_test
)


# -------------------------
# SAVE LABELS
# -------------------------

with open(
    os.path.join(
        OUTPUT_DIR,
        "labels.json"
    ),
    "w"
) as file:

    json.dump(
        label_to_id,
        file,
        indent=4
    )


# -------------------------
# FINAL RESULTS
# -------------------------

print("\nDataset prepared successfully.")

print(
    "\nTrain:",
    X_train.shape,
    y_train.shape
)

print(
    "Validation:",
    X_val.shape,
    y_val.shape
)

print(
    "Test:",
    X_test.shape,
    y_test.shape
)

print(
    "\nSaved inside:"
)

print(
    OUTPUT_DIR
)