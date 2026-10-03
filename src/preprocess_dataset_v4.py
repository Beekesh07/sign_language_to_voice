"""
V4 preprocessing.

Differences from V2:
  * MediaPipe in VIDEO mode with confidence 0.3 (finds many more hands)
  * Saves the FULL per-frame landmark sequence for each video.
    Trimming/resampling happens later (train_v4.py), so you can change
    it without re-running MediaPipe.

Output: data/processed_v4/<label>/<video_name>.npy   shape (num_frames, 225)
"""

import os
import cv2
import numpy as np

from features_v4 import LandmarkExtractor, PROJECT_ROOT, hands_present


DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "isl-isolated-40words")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_v4")


def process_video(extractor, path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(extractor.extract(frame, extractor._last_ts + 1000.0 / fps))
    cap.release()
    return np.array(frames, np.float32) if frames else None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    extractor = LandmarkExtractor()

    labels = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith(".")
    )

    total = 0
    coverage = []

    for label in labels:
        in_dir = os.path.join(DATASET_DIR, label)
        out_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)

        videos = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".mp4"))
        print(f"\n{label}: {len(videos)} videos")

        for name in videos:
            out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".npy")
            if os.path.exists(out_path):
                continue  # resume support
            try:
                seq = process_video(extractor, os.path.join(in_dir, name))
            except Exception as e:
                print("  error:", name, e)
                continue
            if seq is None:
                print("  skipped (no frames):", name)
                continue
            np.save(out_path, seq)
            cov = hands_present(seq).mean()
            coverage.append(cov)
            total += 1
            print(f"  {name}  frames={len(seq)}  hands visible in {cov:.0%} of frames")

    extractor.close()
    print("\nDone. Videos processed:", total)
    if coverage:
        print(f"Average hand coverage: {np.mean(coverage):.0%}")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
