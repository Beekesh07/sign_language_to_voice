"""
V4 real-time prediction.

No countdown needed: just raise your hands into the frame, do one sign,
then lower your hands out of the frame. The sign is detected automatically
(same "hands visible" trimming that training uses), classified, shown and
spoken aloud.

Keys:  SPACE = finish the current sign now   |   Q = quit
Optional voice: pip install pyttsx3
"""

import os
import time
import threading
import cv2
import numpy as np
import torch

from features_v4 import (
    LandmarkExtractor, PROJECT_ROOT, FEATURE_DIM,
    prepare_sequence, to_model_features,
)
from model_v4 import SignGRU


MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_v4.pth")

CONFIDENCE_THRESHOLD = 0.45
END_AFTER_MISSING_FRAMES = 8     # hands gone this many frames -> sign finished
MIN_SIGN_FRAMES = 8              # ignore accidental flickers
MAX_SIGN_FRAMES = 150            # safety limit (~5 s)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(MODEL_PATH, map_location=device)
labels = ckpt["labels"]
model = SignGRU(FEATURE_DIM, len(labels)).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()


# ------------------------------------------------------------
# Optional text-to-speech
# ------------------------------------------------------------
try:
    import pyttsx3

    def speak(text):
        def run():
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=run, daemon=True).start()
except ImportError:
    def speak(text):
        pass
    print("(pyttsx3 not installed - predictions will not be spoken)")


def predict(raw_frames):
    seq = prepare_sequence(np.array(raw_frames, np.float32))
    x = torch.tensor(to_model_features(seq)[None]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    top = probs.argsort()[::-1][:3]
    return [(labels[i], float(probs[i])) for i in top]


extractor = LandmarkExtractor()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera could not be opened.")

recording = []
missing = 0
result_text = "Raise your hands and sign"
top3_text = ""
sentence = []

print("Ready. Raise hands into view, sign, then lower them. Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    raw = extractor.extract(frame, time.time() * 1000)
    hand_visible = np.abs(raw[:126]).sum() > 0

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    force_end = key == 32

    if hand_visible:
        recording.append(raw)
        missing = 0
    elif recording:
        recording.append(raw)
        missing += 1

    finished = recording and (
        missing >= END_AFTER_MISSING_FRAMES
        or len(recording) >= MAX_SIGN_FRAMES
        or force_end
    )

    if finished:
        if len(recording) - missing >= MIN_SIGN_FRAMES:
            top3 = predict(recording)
            word, conf = top3[0]
            top3_text = "  ".join(f"{w} {c:.2f}" for w, c in top3)
            print("Top 3:", top3_text)
            if conf >= CONFIDENCE_THRESHOLD:
                result_text = f"{word} ({conf:.2f})"
                sentence = (sentence + [word])[-6:]
                speak(word.replace("_", " "))
            else:
                result_text = f"Not sure ({word}?) - try again"
        recording, missing = [], 0

    # ---------------- display ----------------
    display = frame.copy()
    if recording:
        cv2.circle(display, (30, 30), 12, (0, 0, 255), -1)
        cv2.putText(display, f"SIGNING... {len(recording)}", (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(display, result_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(display, top3_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(display, " ".join(sentence).replace("_", " "), (20, display.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, "SPACE = end sign | Q = quit", (20, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.imshow("Sign Language to Voice V4", display)

cap.release()
extractor.close()
cv2.destroyAllWindows()
