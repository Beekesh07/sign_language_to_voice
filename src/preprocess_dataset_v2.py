import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

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
    "processed_v2"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30

HAND_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hand_landmarker.task"
)

POSE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "pose_landmarker_lite.task"
)


# ============================================================
# DOWNLOAD MODELS IF MISSING
# ============================================================

if not os.path.exists(HAND_MODEL_PATH):

    print("Downloading hand landmarker model...")

    HAND_URL = (
        "https://storage.googleapis.com/"
        "mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/"
        "hand_landmarker.task"
    )

    urllib.request.urlretrieve(
        HAND_URL,
        HAND_MODEL_PATH
    )

    print("Hand model downloaded.")


if not os.path.exists(POSE_MODEL_PATH):

    print("Downloading pose landmarker model...")

    POSE_URL = (
        "https://storage.googleapis.com/"
        "mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/"
        "pose_landmarker_lite.task"
    )

    urllib.request.urlretrieve(
        POSE_URL,
        POSE_MODEL_PATH
    )

    print("Pose model downloaded.")


# ============================================================
# CREATE MEDIAPIPE DETECTORS
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
    """
    Normalize x, y, z coordinates relative to body center.

    points shape:
    (N, 3)
    """

    points = points.copy()

    points[:, 0] -= center[0]
    points[:, 1] -= center[1]
    points[:, 2] -= center[2]

    if scale > 1e-6:
        points /= scale

    return points


# ============================================================
# EXTRACT FEATURES FROM ONE FRAME
# ============================================================

def extract_features(frame):

    # OpenCV BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # --------------------------------------------------------
    # POSE DETECTION
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

        # MediaPipe Pose:
        # 11 = left shoulder
        # 12 = right shoulder

        left_shoulder = pose_points[11]
        right_shoulder = pose_points[12]

        body_center = (
            left_shoulder +
            right_shoulder
        ) / 2.0

        # Shoulder distance used as body scale
        body_scale = np.linalg.norm(
            left_shoulder[:2]
            -
            right_shoulder[:2]
        )

        if body_scale < 1e-6:
            body_scale = 1.0


    # --------------------------------------------------------
    # HAND DETECTION
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


    # Only normalize hands when detected.
    # Otherwise keep zero vectors as zeros.

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
    # FLATTEN
    # --------------------------------------------------------

    left_features = left_hand.flatten()

    right_features = right_hand.flatten()

    pose_features = pose_points.flatten()


    # 63 + 63 + 99 = 225

    features = np.concatenate(
        [
            left_features,
            right_features,
            pose_features
        ]
    )

    return features.astype(
        np.float32
    )


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        features = extract_features(
            frame
        )

        frames.append(
            features
        )


    cap.release()


    if len(frames) == 0:
        return None


    frames = np.array(
        frames,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Convert variable-length video -> exactly 30 frames
    # --------------------------------------------------------

    indices = np.linspace(
        0,
        len(frames) - 1,
        SEQUENCE_LENGTH
    ).astype(int)


    sequence = frames[
        indices
    ]


    return sequence


# ============================================================
# PROCESS COMPLETE DATASET
# ============================================================

def preprocess_dataset():

    labels = sorted(
        [
            folder
            for folder in os.listdir(
                DATASET_DIR
            )
            if os.path.isdir(
                os.path.join(
                    DATASET_DIR,
                    folder
                )
            )
            and not folder.startswith(".")
        ]
    )


    print(
        "\nTotal labels:",
        len(labels)
    )


    total_processed = 0


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


        video_files = sorted(
            [
                file
                for file in os.listdir(
                    label_path
                )
                if file.lower().endswith(
                    ".mp4"
                )
            ]
        )


        print(
            f"\nProcessing '{label}' "
            f"({len(video_files)} videos)"
        )


        for index, video_file in enumerate(
            video_files
        ):

            video_path = os.path.join(
                label_path,
                video_file
            )


            try:

                sequence = process_video(
                    video_path
                )


                if sequence is None:

                    print(
                        f"Skipped: "
                        f"{video_file}"
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


                total_processed += 1


                print(
                    f"{index + 1}/"
                    f"{len(video_files)} saved"
                )


            except Exception as error:

                print(
                    f"Error processing "
                    f"{video_file}: "
                    f"{error}"
                )


    print(
        "\nTotal videos processed:",
        total_processed
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    preprocess_dataset()

    hand_detector.close()

    pose_detector.close()

    print(
        "\nV2 preprocessing complete."
    )

    print(
        "Output folder:"
    )

    print(
        OUTPUT_DIR
    )