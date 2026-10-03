# Sign Language to Voice (ISL)

Real-time Indian Sign Language (ISL) word recognition from a webcam, with spoken output.

MediaPipe extracts hand and upper-body landmarks from each frame, a small bidirectional GRU classifies the sign, and the predicted word is spoken aloud with `pyttsx3`.

- **30 words**: drink, eat, father, food, friend, go, he, hello, help, hospital, market, mother, no, okay, please, school, she, sister, sit, sorry, student, tea, teacher, thank you, today, water, what, where, yes, you
- **Dataset**: [ISL Isolated Word Dataset (40 words)](https://huggingface.co/datasets/vidit031/isl-isolated-40words). Words with fewer than 5 videos are skipped.
- **Test accuracy**: about 78–80% on held-out dataset videos (up from about 60% in earlier versions)

## How it works

1. **Landmarks**: MediaPipe Hand + Pose (video mode, confidence 0.3), normalised to the shoulder centre and shoulder width.
2. **Sign trimming**: only the frames where the hands are visible are kept, then resampled to 32 frames. Idle frames at the start and end are removed.
3. **Features**: hand position, hand shape relative to the wrist, upper-body pose and frame-to-frame velocity.
4. **Model**: 2-layer bidirectional GRU, trained with augmentation (mirroring, speed changes, rotation, scale and shift).
5. **Real time**: the sign is detected automatically. Raise your hands, sign, then lower your hands.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## Web app (Streamlit)

```bash
streamlit run app.py
```

- **Upload a video**: upload a short clip of one sign, or try one of the samples, to see the predicted word, the top 5 guesses and landmark keyframes. The word is spoken aloud.
- **Live webcam**: the camera runs in the browser. Raise your hands, sign, then lower your hands. Each recognised word is spoken and added to a sentence.

## Desktop app (OpenCV window)

```bash
pip install pyttsx3        # offline voice for the desktop app
python src/realtime_predict_v4.py
```

Sit so that your shoulders and hands are in view. Raise your hands, do one sign, then lower your hands out of the frame. Press SPACE to end a sign manually and Q to quit.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app** and choose this repository.
3. Set **Main file path** to `app.py`.
4. Under **Advanced settings**, choose **Python 3.13**, then click **Deploy**.

`requirements.txt` (Python packages, with CPU-only PyTorch) and `packages.txt` (system libraries for OpenCV) are picked up automatically.

**Live webcam on the cloud:** WebRTC sometimes can't connect through strict networks. If the camera doesn't connect, add a free TURN server (for example from [metered.ca](https://www.metered.ca/tools/openrelay/)) under the app's **Settings → Secrets**:

```toml
[turn]
urls = ["turn:YOUR_TURN_SERVER:80", "turn:YOUR_TURN_SERVER:443"]
username = "YOUR_USERNAME"
credential = "YOUR_PASSWORD"
```

The **Upload a video** tab works without this.

## Train from scratch

```bash
python download_dataset.py              # downloads videos into data/
python src/preprocess_dataset_v4.py     # extracts landmarks -> data/processed_v4
python src/train_v4.py                  # trains -> models/best_v4.pth
```

## Project structure

```
app.py                      Streamlit web app
requirements.txt            Python packages
packages.txt                system packages for Streamlit Cloud
samples/                    demo clips (INCLUDE dataset, CC-BY-4.0)
src/
  features_v4.py            shared landmark extraction and feature code
  model_v4.py               GRU model
  preprocess_dataset_v4.py  videos -> landmark sequences
  train_v4.py               training and evaluation
  realtime_predict_v4.py    webcam prediction and speech
models/
  best_v4.pth               trained model
  hand_landmarker.task      MediaPipe hand model
  pose_landmarker_lite.task MediaPipe pose model
```

Older v1–v3 scripts are kept in `src/` for reference.

## Dataset credits

The dataset aggregates INCLUDE (AI4Bharat, CC-BY-4.0), CISLR (Exploration-Lab), ISL500/ISL-DATA and the ISLRTC dictionary. Please cite the upstream sources. See the dataset card for licence details.
