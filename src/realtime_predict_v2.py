import os
import json
import time
import cv2
import numpy as np
import torch
import mediapipe as mp

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

CONFIDENCE_THRESHOLD = 0.35

COUNTDOWN_SECONDS = 3


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
# LOAD TRANSFORMER MODEL
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
# FEATURE EXTRACTION
# MUST MATCH preprocess_dataset_v2.py
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
# PREDICTION FUNCTION
# ============================================================

def predict_sign(sequence):

    sequence = np.array(
        sequence,
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

    predicted_label = id_to_label[
        predicted_id
    ]


    return predicted_label, confidence


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera could not be opened.")

    hand_detector.close()
    pose_detector.close()

    exit()


# ============================================================
# STATE
# ============================================================

status_text = "Press SPACE to record sign"

result_text = "No prediction yet"

result_confidence = 0.0


print("\nCamera started.")

print("Press SPACE once to record one sign.")

print("After countdown, perform the complete sign.")

print("Press Q to quit.")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    display_frame = frame.copy()


    # --------------------------------------------------------
    # DISPLAY CURRENT RESULT
    # --------------------------------------------------------

    cv2.putText(
        display_frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display_frame,
        f"Prediction: {result_text}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        display_frame,
        f"Confidence: {result_confidence:.2f}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    cv2.putText(
        display_frame,
        "SPACE = record | Q = quit",
        (20, display_frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Sign Language Recognition V2",
        display_frame
    )


    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # QUIT
    # ========================================================

    if key == ord("q"):

        break


    # ========================================================
    # SPACE -> COUNTDOWN -> RECORD
    # ========================================================

    if key == 32:

        # ----------------------------------------------------
        # COUNTDOWN
        # ----------------------------------------------------

        for countdown in range(
            COUNTDOWN_SECONDS,
            0,
            -1
        ):

            countdown_start = time.time()

            while (
                time.time()
                -
                countdown_start
                <
                1.0
            ):

                ret, countdown_frame = cap.read()

                if not ret:
                    break


                cv2.putText(
                    countdown_frame,
                    f"GET READY: {countdown}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    countdown_frame,
                    "Start signing when RECORDING appears",
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "Sign Language Recognition V2",
                    countdown_frame
                )

                # Allow Q during countdown
                countdown_key = cv2.waitKey(1) & 0xFF

                if countdown_key == ord("q"):

                    cap.release()
                    hand_detector.close()
                    pose_detector.close()
                    cv2.destroyAllWindows()

                    exit()


        # ----------------------------------------------------
        # RECORD 30 FRAMES
        # ----------------------------------------------------

        print("\nRecording sign...")

        sequence = []


        while len(sequence) < SEQUENCE_LENGTH:

            ret, record_frame = cap.read()

            if not ret:
                break


            features = extract_features(
                record_frame
            )

            sequence.append(
                features
            )


            progress = len(sequence)


            cv2.putText(
                record_frame,
                "RECORDING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )


            cv2.putText(
                record_frame,
                f"Frames: {progress}/{SEQUENCE_LENGTH}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


            cv2.putText(
                record_frame,
                "Perform ONE complete sign",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.imshow(
                "Sign Language Recognition V2",
                record_frame
            )


            record_key = cv2.waitKey(1) & 0xFF

            if record_key == ord("q"):

                cap.release()
                hand_detector.close()
                pose_detector.close()
                cv2.destroyAllWindows()

                exit()


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        if len(sequence) == SEQUENCE_LENGTH:

            predicted_label, confidence = predict_sign(
                sequence
            )


            result_confidence = confidence


            if confidence >= CONFIDENCE_THRESHOLD:

                result_text = predicted_label

                status_text = (
                    "Prediction complete - "
                    "press SPACE for another sign"
                )

            else:

                result_text = (
                    f"Uncertain ({predicted_label})"
                )

                status_text = (
                    "Low confidence - "
                    "press SPACE and try again"
                )


            print(
                f"Prediction: {predicted_label}"
            )

            print(
                f"Confidence: {confidence:.2f}"
            )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

hand_detector.close()

pose_detector.close()

cv2.destroyAllWindows()