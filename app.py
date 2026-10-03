"""
Sign Language to Voice - Streamlit app

Run locally:   streamlit run app.py
"""

import os
import sys
import io
import time
import queue
import tempfile
import threading

import av
import cv2
import numpy as np
import streamlit as st
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features_v4 import (  # noqa: E402
    LandmarkExtractor, FEATURE_DIM, hands_present, trim_to_sign,
    prepare_sequence, to_model_features,
)
from model_v4 import SignGRU  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "best_v4.pth")
SAMPLES_DIR = os.path.join(ROOT, "samples")

# Live mode: sign ends when hands are gone for this many frames
END_AFTER_MISSING_FRAMES = 8
MIN_SIGN_FRAMES = 8
MAX_SIGN_FRAMES = 150

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


st.set_page_config(
    page_title="Sign Language to Voice",
    page_icon="🤟",
    layout="wide",
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = SignGRU(FEATURE_DIM, len(ckpt["labels"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["labels"]


MODEL, LABELS = load_model()


def pretty(word):
    return word.replace("_", " ")


def predict(raw_frames, k=5):
    seq = prepare_sequence(np.asarray(raw_frames, np.float32))
    x = torch.tensor(to_model_features(seq)[None])
    with torch.no_grad():
        probs = torch.softmax(MODEL(x), dim=1)[0].numpy()
    top = probs.argsort()[::-1][:k]
    return [(LABELS[i], float(probs[i])) for i in top]


# ============================================================
# SPEECH (Google TTS -> audio played in the browser)
# ============================================================

@st.cache_data(show_spinner=False)
def tts_mp3(text):
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


_speech_counter = 0


def speak(text, container=st, autoplay=True):
    global _speech_counter
    audio = tts_mp3(text)
    if audio:
        # Streamlit builds the audio element's ID from its content, so the same
        # word spoken twice would clash. A unique alt text keeps every ID different.
        _speech_counter += 1
        container.audio(audio, format="audio/mp3", autoplay=autoplay,
                        alt=f"Spoken word: {text} ({_speech_counter})")
    else:
        container.caption("🔇 Speech unavailable (no internet connection for text-to-speech).")


# ============================================================
# DRAWING
# ============================================================

def draw_landmarks(img, extractor):
    h, w = img.shape[:2]
    if extractor.last_pose is not None:
        p = extractor.last_pose
        for a, b in POSE_CONNECTIONS:
            cv2.line(img, (int(p[a, 0] * w), int(p[a, 1] * h)),
                     (int(p[b, 0] * w), int(p[b, 1] * h)), (255, 200, 0), 2)
    for hand in extractor.last_hands:
        pts = [(int(x * w), int(y * h)) for x, y in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (0, 255, 120), 2)
        for pt in pts:
            cv2.circle(img, pt, 3, (255, 255, 255), -1)
    return img


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
      .word-card {padding: 1.4rem 1.6rem; border-radius: 14px;
                  background: linear-gradient(135deg, #1f6f5c 0%, #2a9d8f 100%);
                  color: #fff; margin-bottom: 0.8rem;}
      .word-card .label {font-size: 0.85rem; opacity: 0.85; letter-spacing: .06em; text-transform: uppercase;}
      .word-card .word {font-size: 2.6rem; font-weight: 700; line-height: 1.15;}
      .word-card .conf {font-size: 0.95rem; opacity: 0.9;}
      .word-card.unsure {background: linear-gradient(135deg, #8a5a00 0%, #c98a1b 100%);}
      .sentence {font-size: 1.5rem; font-weight: 600; padding: .6rem 1rem;
                 border-radius: 10px; border: 1px dashed rgba(128,128,128,.5); min-height: 3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def word_card(word, conf, threshold, container=st):
    sure = conf >= threshold
    container.markdown(
        f"""<div class="word-card {'' if sure else 'unsure'}">
              <div class="label">{'Predicted sign' if sure else 'Not sure — best guess'}</div>
              <div class="word">{pretty(word).title()}</div>
              <div class="conf">Confidence {conf:.0%}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def top_k_bars(top, container=st):
    for word, p in top:
        c1, c2 = container.columns([1, 3])
        c1.write(pretty(word))
        c2.progress(min(max(p, 0.0), 1.0), text=f"{p:.0%}")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Confidence threshold", 0.1, 0.9, 0.45, 0.05,
        help="Below this, the app shows 'Not sure' and doesn't speak.",
    )
    voice_on = st.toggle("Speak predictions", value=True)

    st.divider()
    st.subheader(f"Supported words ({len(LABELS)})")
    st.write(", ".join(pretty(w) for w in LABELS))

    st.divider()
    st.caption(
        "Indian Sign Language (ISL) isolated-word recognition. "
        "MediaPipe landmarks + bidirectional GRU."
    )


# ============================================================
# HEADER
# ============================================================

st.title("🤟 Sign Language to Voice")
st.write(
    "Recognises **Indian Sign Language** words from video and speaks them aloud. "
    "Upload a short clip of one sign, or use your webcam live."
)

tab_upload, tab_live, tab_about = st.tabs(["📁 Upload a video", "🎥 Live webcam", "ℹ️ How it works"])


# ============================================================
# TAB 1 - UPLOAD
# ============================================================

def process_video_file(path, progress):
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    raw, frames, drawn = [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # keep processing fast for large uploads
        h, w = frame.shape[:2]
        if max(h, w) > 960:
            s = 960 / max(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        raw.append(extractor.extract(frame, i * 1000.0 / fps))
        drawn.append(draw_landmarks(frame.copy(), extractor))
        i += 1
        progress.progress(min(i / total, 1.0), text=f"Reading landmarks… frame {i}/{total}")
    cap.release()
    extractor.close()
    progress.empty()
    return np.array(raw, np.float32), drawn


with tab_upload:
    left, right = st.columns([1, 1], gap="large")

    with left:
        source = st.radio("Video source", ["Upload my own", "Try a sample"], horizontal=True)

        video_bytes, video_name = None, None
        if source == "Upload my own":
            up = st.file_uploader(
                "Video of ONE sign (mp4, mov, avi, webm, ≤ 50 MB)",
                type=["mp4", "mov", "avi", "webm", "mkv"],
            )
            st.caption("Tip: keep your shoulders and both hands in view, like a news-reader framing.")
            if up is not None:
                video_bytes, video_name = up.getvalue(), up.name
        else:
            samples = sorted(f for f in os.listdir(SAMPLES_DIR) if f.endswith(".mp4")) if os.path.isdir(SAMPLES_DIR) else []
            if samples:
                choice = st.selectbox("Sample clip", samples, format_func=lambda f: pretty(os.path.splitext(f)[0]).title())
                with open(os.path.join(SAMPLES_DIR, choice), "rb") as f:
                    video_bytes, video_name = f.read(), choice
                st.caption("Samples are from the INCLUDE dataset (CC-BY-4.0).")
            else:
                st.info("No sample videos found in the samples/ folder.")

        if video_bytes:
            st.video(video_bytes)

        run = st.button("🔍 Recognise sign", type="primary", disabled=video_bytes is None, width="stretch")

    with right:
        if run and video_bytes:
            suffix = os.path.splitext(video_name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name
            try:
                raw, drawn = process_video_file(tmp_path, st.progress(0.0, text="Starting…"))
            finally:
                os.unlink(tmp_path)

            if len(raw) == 0:
                st.error("Couldn't read any frames from this video. Try converting it to MP4.")
            elif hands_present(raw).sum() < 2:
                st.warning("No hands were detected in this video. Make sure both hands are clearly visible.")
            else:
                top = predict(raw)
                st.session_state["upload_result"] = {
                    "top": top, "raw_len": len(raw),
                    "coverage": float(hands_present(raw).mean()),
                    "sign_len": len(trim_to_sign(raw)),
                    "keyframes": [drawn[j] for j in np.linspace(
                        int(np.argmax(hands_present(raw))),
                        len(raw) - 1 - int(np.argmax(hands_present(raw)[::-1])), 4).astype(int)],
                    "spoken": False,
                }

        res = st.session_state.get("upload_result")
        if res:
            word, conf = res["top"][0]
            word_card(word, conf, threshold)
            if voice_on and conf >= threshold:
                speak(pretty(word), autoplay=not res["spoken"])
                res["spoken"] = True

            st.subheader("Top predictions")
            top_k_bars(res["top"])

            st.subheader("What the model saw")
            m1, m2, m3 = st.columns(3)
            m1.metric("Frames", res["raw_len"])
            m2.metric("Hands visible", f"{res['coverage']:.0%}")
            m3.metric("Sign frames", res["sign_len"])
            cols = st.columns(4)
            for c, img in zip(cols, res["keyframes"]):
                c.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")
        elif not run:
            st.info("Choose a video on the left and press **Recognise sign**.")


# ============================================================
# TAB 3 - ABOUT
# ============================================================

with tab_about:
    st.markdown(
        f"""
### Pipeline

1. **Landmarks**: MediaPipe Hand + Pose find 21 points per hand and the upper body in every frame.
2. **Sign trimming**: only the frames where hands are visible are kept, so idle time before and after the sign is ignored. The result is resampled to 32 frames.
3. **Features**: hand position relative to the shoulders, hand shape relative to the wrist, upper-body pose and frame-to-frame motion.
4. **Model**: a 2-layer bidirectional GRU classifies the sign into one of **{len(LABELS)} words**.
5. **Voice**: the predicted word is spoken using Google text-to-speech.

### Accuracy

About **78–80%** on held-out videos from the dataset. Accuracy on new people and cameras is usually lower.
Signing clearly, facing the camera with good lighting, helps a lot.

### Dataset

[ISL Isolated Word Dataset (40 words)](https://huggingface.co/datasets/vidit031/isl-isolated-40words), aggregated from
INCLUDE (AI4Bharat, CC-BY-4.0), CISLR, ISL500/ISL-DATA and the ISLRTC dictionary.
Words with fewer than 5 videos were left out.
"""
    )


# ============================================================
# TAB 2 - LIVE WEBCAM
# ============================================================

def rtc_configuration():
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:
        if "turn" in st.secrets:
            t = st.secrets["turn"]
            servers.append({
                "urls": list(t["urls"]),
                "username": t["username"],
                "credential": t["credential"],
            })
    except Exception:
        pass  # no secrets file
    return {"iceServers": servers}


class SignProcessor:
    """Runs on the WebRTC worker thread for each browser session."""

    def __init__(self):
        self.extractor = LandmarkExtractor()
        self.recording = []
        self.missing = 0
        self.results = queue.Queue()
        self.lock = threading.Lock()
        self.force_end = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        raw = self.extractor.extract(img, time.time() * 1000)
        visible = np.abs(raw[:126]).sum() > 0

        with self.lock:
            if visible:
                self.recording.append(raw)
                self.missing = 0
            elif self.recording:
                self.recording.append(raw)
                self.missing += 1

            done = self.recording and (
                self.missing >= END_AFTER_MISSING_FRAMES
                or len(self.recording) >= MAX_SIGN_FRAMES
                or self.force_end
            )
            if done:
                if len(self.recording) - self.missing >= MIN_SIGN_FRAMES:
                    self.results.put(predict(self.recording))
                self.recording, self.missing, self.force_end = [], 0, False
            n = len(self.recording)

        out = draw_landmarks(img, self.extractor)
        if n:
            cv2.circle(out, (24, 24), 10, (0, 0, 255), -1)
            cv2.putText(out, f"SIGNING {n}", (42, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(out, "Raise your hands to sign", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(out, format="bgr24")


with tab_live:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode

    st.markdown(
        "1. Press **START** and allow camera access.  \n"
        "2. Sit back so your **shoulders and both hands** are in view.  \n"
        "3. Raise your hands, do **one sign**, then **lower your hands out of the frame** — "
        "that's how the app knows the sign is finished."
    )

    cam_col, out_col = st.columns([3, 2], gap="large")

    with cam_col:
        ctx = webrtc_streamer(
            key="sign-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration(),
            video_processor_factory=SignProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )
        st.caption(
            "Camera not connecting? Some networks block WebRTC. "
            "Use the **Upload a video** tab instead — it always works."
        )

    with out_col:
        b1, b2 = st.columns(2)
        if b1.button("⏹ End sign now", width="stretch") and ctx.video_processor:
            with ctx.video_processor.lock:
                ctx.video_processor.force_end = True
        if b2.button("🧹 Clear sentence", width="stretch"):
            st.session_state["sentence"] = []

        st.session_state.setdefault("sentence", [])
        result_box = st.empty()
        bars_box = st.container()
        st.markdown("**Sentence**")
        sentence_box = st.empty()
        audio_box = st.empty()

        def show_sentence():
            text = " ".join(pretty(w) for w in st.session_state["sentence"]) or "&nbsp;"
            sentence_box.markdown(f'<div class="sentence">{text}</div>', unsafe_allow_html=True)

        show_sentence()
        if not ctx.state.playing:
            result_box.info("Press **START** to begin.")
        else:
            result_box.info("Waiting for a sign… raise your hands, sign, then lower them.")

        # Poll the processor for finished signs while the camera runs
        while ctx.state.playing and ctx.video_processor:
            try:
                top = ctx.video_processor.results.get(timeout=0.2)
            except queue.Empty:
                continue
            word, conf = top[0]
            with result_box.container():
                word_card(word, conf, threshold)
                top_k_bars(top[:3])
            if conf >= threshold:
                st.session_state["sentence"] = (st.session_state["sentence"] + [word])[-8:]
                show_sentence()
                if voice_on:
                    audio_box.empty()
                    speak(pretty(word), audio_box)
