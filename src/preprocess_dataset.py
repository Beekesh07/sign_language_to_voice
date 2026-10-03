import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# PROJECT PATHS
# -----------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "isl-isolated-40words"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hand_landmarker.task"
)

SEQUENCE_LENGTH = 30


# -----------------------------
# DOWNLOAD MEDIAPIPE MODEL
# -----------------------------

if not os.path.exists(MODEL_PATH):

    print("Downloading hand landmark model...")

    url = (
        "https://storage.googleapis.com/"
        "mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/"
        "hand_landmarker.task"
    )

    urllib.request.urlretrieve(
        url,
        MODEL_PATH
    )

    print("Model downloaded successfully.")


# -----------------------------
# CREATE MEDIAPIPE DETECTOR
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# -----------------------------
# EXTRACT LANDMARKS
# -----------------------------

def extract_landmarks(frame):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    left_hand = np.zeros(
        63,
        dtype=np.float32
    )

    right_hand = np.zeros(
        63,
        dtype=np.float32
    )

    for i, hand_landmarks in enumerate(
        result.hand_landmarks
    ):

        coords = []

        for landmark in hand_landmarks:

            coords.extend(
                [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]
            )

        coords = np.array(
            coords,
            dtype=np.float32
        )

        if i == 0:
            left_hand = coords

        elif i == 1:
            right_hand = coords

    return np.concatenate(
        [
            left_hand,
            right_hand
        ]
    )


# -----------------------------
# PROCESS ONE VIDEO
# -----------------------------

def process_video(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        landmarks = extract_landmarks(
            frame
        )

        frames.append(
            landmarks
        )

    cap.release()

    if len(frames) == 0:
        return None

    frames = np.array(
        frames,
        dtype=np.float32
    )

    indices = np.linspace(
        0,
        len(frames) - 1,
        SEQUENCE_LENGTH
    ).astype(int)

    return frames[indices]


# -----------------------------
# PROCESS WHOLE DATASET
# -----------------------------

def preprocess_dataset():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    labels = sorted(
        [
            folder
            for folder in os.listdir(DATASET_DIR)
            if os.path.isdir(
                os.path.join(
                    DATASET_DIR,
                    folder
                )
            )
        ]
    )

    print(
        f"\nTotal labels found: {len(labels)}"
    )

    for label in labels:

        label_path = os.path.join(
            DATASET_DIR,
            label
        )

        output_label_path = os.path.join(
            OUTPUT_DIR,
            label
        )

        os.makedirs(
            output_label_path,
            exist_ok=True
        )

        video_files = [
            file
            for file in os.listdir(label_path)
            if file.lower().endswith(".mp4")
        ]

        print(
            f"\nProcessing {label}: "
            f"{len(video_files)} videos"
        )

        for index, video_file in enumerate(
            video_files
        ):

            video_path = os.path.join(
                label_path,
                video_file
            )

            sequence = process_video(
                video_path
            )

            if sequence is None:

                print(
                    f"Skipped: {video_file}"
                )

                continue

            save_path = os.path.join(
                output_label_path,
                f"{index:04d}.npy"
            )

            np.save(
                save_path,
                sequence
            )

            print(
                f"{index + 1}/"
                f"{len(video_files)} saved"
            )


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":

    preprocess_dataset()

    detector.close()

    print(
        "\nDataset preprocessing complete."
    )
    