import os
import json
import cv2
import numpy as np
import torch
import mediapipe as mp

from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from transformer_model_v2 import SignTransformerV2


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

TRAINING_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "training_v2"
)

HAND_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hand_landmarker.task"
)

POSE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "pose_landmarker_lite.task"
)

TRANSFORMER_PATH = os.path.join(
    MODEL_DIR,
    "best_transformer_v2.pth"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30

CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


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


id_to_label = {
    value: key
    for key, value in label_to_id.items()
}


NUM_CLASSES = len(label_to_id)


# ============================================================
# LOAD TRANSFORMER
# ============================================================

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
        TRANSFORMER_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ============================================================
# MEDIAPIPE HAND DETECTOR
# ============================================================

hand_base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    num_hands=2
)

hand_detector = vision.HandLandmarker.create_from_options(
    hand_options
)


# ============================================================
# MEDIAPIPE POSE DETECTOR
# ============================================================

pose_base_options = python.BaseOptions(
    model_asset_path=POSE_MODEL_PATH
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    num_poses=1
)

pose_detector = vision.PoseLandmarker.create_from_options(
    pose_options
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_points(points, center, scale):

    points = points.copy()

    points[:, 0] -= center[0]
    points[:, 1] -= center[1]
    points[:, 2] -= center[2]

    if scale > 1e-6:
        points /= scale

    return points


# ============================================================
# EXTRACT 225 FEATURES
# MUST MATCH PREPROCESSING V2
# ============================================================

def extract_features(frame):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # POSE
    # --------------------------------------------------------

    pose_result = pose_detector.detect(
        mp_image
    )

    pose_points = np.zeros(
        (33, 3),
        dtype=np.float32
    )

    body_center = np.array(
        [0.5, 0.5, 0.0],
        dtype=np.float32
    )

    body_scale = 1.0


    if pose_result.pose_landmarks:

        landmarks = pose_result.pose_landmarks[0]

        pose_points = np.array(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]
                for landmark in landmarks
            ],
            dtype=np.float32
        )

        left_shoulder = pose_points[11]
        right_shoulder = pose_points[12]

        body_center = (
            left_shoulder
            +
            right_shoulder
        ) / 2.0

        body_scale = np.linalg.norm(
            left_shoulder[:2]
            -
            right_shoulder[:2]
        )

        if body_scale < 1e-6:
            body_scale = 1.0


    # --------------------------------------------------------
    # HANDS
    # --------------------------------------------------------

    hand_result = hand_detector.detect(
        mp_image
    )

    left_hand = np.zeros(
        (21, 3),
        dtype=np.float32
    )

    right_hand = np.zeros(
        (21, 3),
        dtype=np.float32
    )


    if (
        hand_result.hand_landmarks
        and hand_result.handedness
    ):

        for hand_landmarks, handedness in zip(
            hand_result.hand_landmarks,
            hand_result.handedness
        ):

            hand_points = np.array(
                [
                    [
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ]
                    for landmark in hand_landmarks
                ],
                dtype=np.float32
            )

            hand_name = (
                handedness[0]
                .category_name
            )

            if hand_name == "Left":

                left_hand = hand_points

            elif hand_name == "Right":

                right_hand = hand_points


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    pose_points = normalize_points(
        pose_points,
        body_center,
        body_scale
    )


    if np.any(left_hand):

        left_hand = normalize_points(
            left_hand,
            body_center,
            body_scale
        )


    if np.any(right_hand):

        right_hand = normalize_points(
            right_hand,
            body_center,
            body_scale
        )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    features = np.concatenate(
        [
            left_hand.flatten(),
            right_hand.flatten(),
            pose_points.flatten()
        ]
    )

    return features.astype(
        np.float32
    )


# ============================================================
# FRAME BUFFER
# ============================================================

sequence_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera could not be opened.")

    exit()


print("\nCamera started.")

print("Perform a sign for about 1 second.")

print("Press Q to quit.")


# ============================================================
# REAL-TIME LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # --------------------------------------------------------
    # Extract landmarks
    # --------------------------------------------------------

    features = extract_features(
        frame
    )

    sequence_buffer.append(
        features
    )


    predicted_label = "Collecting..."

    confidence = 0.0


    # --------------------------------------------------------
    # Predict when we have 30 frames
    # --------------------------------------------------------

    if len(sequence_buffer) == SEQUENCE_LENGTH:

        sequence = np.array(
            sequence_buffer,
            dtype=np.float32
        )

        sequence = np.expand_dims(
            sequence,
            axis=0
        )


        tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        ).to(device)


        with torch.no_grad():

            outputs = model(
                tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence_tensor, predicted = torch.max(
                probabilities,
                dim=1
            )


        predicted_id = predicted.item()

        confidence = confidence_tensor.item()


        if confidence >= CONFIDENCE_THRESHOLD:

            predicted_label = id_to_label[
                predicted_id
            ]

        else:

            predicted_label = "Uncertain"


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Sign: {predicted_label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Frames: {len(sequence_buffer)}/30",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Sign Language Recognition",
        frame
    )


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

hand_detector.close()

pose_detector.close()

cv2.destroyAllWindows()