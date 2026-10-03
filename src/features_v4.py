"""
Shared feature code for V4.

Preprocessing, training and real-time prediction ALL import from this file,
so the features the model sees during training are exactly the features it
sees from the webcam.
"""

import os
import numpy as np
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")

SEQUENCE_LENGTH = 32

# Lower than MediaPipe's default 0.5 -> far fewer missed hands on
# small / blurry / fast-moving hands.
DETECTION_CONFIDENCE = 0.3

# Upper-body pose landmarks only (nose ... hands). Legs/hips are often
# out of frame on a webcam and just add noise.
POSE_UPPER = list(range(0, 23))

# Left/right landmark pairs in MediaPipe Pose (used for mirroring).
POSE_LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12),
                 (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)]


# ============================================================
# DETECTORS  (VIDEO mode = uses tracking between frames)
# ============================================================

class LandmarkExtractor:

    def __init__(self):

        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=DETECTION_CONFIDENCE,
            min_hand_presence_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=DETECTION_CONFIDENCE,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.hands = vision.HandLandmarker.create_from_options(hand_options)
        self.pose = vision.PoseLandmarker.create_from_options(pose_options)

        # VIDEO mode needs strictly increasing timestamps, even across videos
        self._last_ts = -1

    def close(self):
        self.hands.close()
        self.pose.close()

    def extract(self, frame_bgr, timestamp_ms):
        """
        Returns a (225,) vector: left hand (63) + right hand (63) + pose (99),
        normalised to the shoulder centre and shoulder width.
        Missing hands are all zeros.
        """

        timestamp_ms = int(max(timestamp_ms, self._last_ts + 1))
        self._last_ts = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = self.pose.detect_for_video(image, timestamp_ms)
        hand_result = self.hands.detect_for_video(image, timestamp_ms)

        pose = np.zeros((33, 3), np.float32)
        center = np.array([0.5, 0.5, 0.0], np.float32)
        scale = 1.0

        if pose_result.pose_landmarks:
            pose = np.array([[p.x, p.y, p.z] for p in pose_result.pose_landmarks[0]], np.float32)
            center = (pose[11] + pose[12]) / 2.0
            scale = float(np.linalg.norm(pose[11, :2] - pose[12, :2]))
            if scale < 1e-6:
                scale = 1.0

        left = np.zeros((21, 3), np.float32)
        right = np.zeros((21, 3), np.float32)

        for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            pts = np.array([[p.x, p.y, p.z] for p in landmarks], np.float32)
            if handedness[0].category_name == "Left":
                left = pts
            else:
                right = pts

        def norm(p):
            return (p - center) / scale

        pose = norm(pose)
        if left.any():
            left = norm(left)
        if right.any():
            right = norm(right)

        return np.concatenate([left.ravel(), right.ravel(), pose.ravel()]).astype(np.float32)


# ============================================================
# SEQUENCE PROCESSING
# ============================================================

def hands_present(raw_seq):
    """Boolean per frame: is at least one hand detected?"""
    return np.abs(raw_seq[:, :126]).sum(axis=1) > 0


def resample(seq, length=SEQUENCE_LENGTH):
    """Linear-interpolate a (N, D) sequence to (length, D)."""
    if len(seq) == 1:
        return np.repeat(seq, length, axis=0).astype(np.float32)
    idx = np.linspace(0, len(seq) - 1, length)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(seq) - 1)
    w = (idx - i0)[:, None]
    return (seq[i0] * (1 - w) + seq[i1] * w).astype(np.float32)


def forward_fill_hands(seq):
    """If a hand drops out for a few frames mid-sign, reuse its last position."""
    seq = seq.copy()
    for sl in (slice(0, 63), slice(63, 126)):
        last = None
        for t in range(len(seq)):
            if np.abs(seq[t, sl]).sum() > 0:
                last = seq[t, sl].copy()
            elif last is not None:
                seq[t, sl] = last
    return seq


def trim_to_sign(raw_seq, pad=2):
    """
    Keep only the part of the clip where hands are visible (the actual sign),
    dropping idle frames at the start/end. This is the single most important
    fix: in most dataset clips the sign covers only 20-45% of the video.
    """
    present = hands_present(raw_seq)
    if present.sum() < 2:
        return raw_seq
    first, last = np.where(present)[0][[0, -1]]
    first = max(0, first - pad)
    last = min(len(raw_seq) - 1, last + pad)
    return raw_seq[first:last + 1]


def prepare_sequence(raw_seq):
    """raw (N, 225) -> trimmed, gap-filled, resampled (SEQUENCE_LENGTH, 225)."""
    return resample(forward_fill_hands(trim_to_sign(raw_seq)))


def mirror(seq):
    """Flip left/right (for left-handed vs right-handed signers)."""
    s = seq.reshape(len(seq), 75, 3).copy()
    s[:, :, 0] *= -1
    s[:, :42] = np.concatenate([s[:, 21:42], s[:, 0:21]], axis=1)
    pose = s[:, 42:].copy()
    for a, b in POSE_LR_PAIRS:
        pose[:, [a, b]] = pose[:, [b, a]]
    s[:, 42:] = pose
    return s.reshape(len(seq), 225)


def to_model_features(seq):
    """
    (T, 225) -> (T, F) model input:
      per hand: position relative to body (x,y), hand shape relative to wrist
                and scaled by hand size (x,y), present flag
      upper-body pose (x,y)
      + frame-to-frame velocity of everything
    """
    s = seq.reshape(len(seq), 75, 3)
    parts = []
    for hand in (s[:, 0:21], s[:, 21:42]):
        present = (np.abs(hand).sum(axis=(1, 2)) > 0).astype(np.float32)[:, None]
        wrist = hand[:, 0:1]
        size = np.linalg.norm(hand[:, 9, :2] - hand[:, 0, :2], axis=1)[:, None, None] + 1e-6
        shape = ((hand - wrist)[:, :, :2] / size).reshape(len(s), -1) * present
        parts += [hand[:, :, :2].reshape(len(s), -1), shape, present]
    parts.append(s[:, 42:][:, POSE_UPPER, :2].reshape(len(s), -1))
    f = np.concatenate(parts, axis=1)
    velocity = np.diff(f, axis=0, prepend=f[:1])
    return np.concatenate([f, velocity], axis=1).astype(np.float32)


FEATURE_DIM = to_model_features(np.zeros((SEQUENCE_LENGTH, 225), np.float32)).shape[1]
