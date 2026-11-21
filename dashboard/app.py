"""Streamlit dashboard.

    streamlit run dashboard/app.py

Tabs: Live webcam (needs streamlit-webrtc) | Upload video | Gloss -> sentence | Model report
"""

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from core.config import CFG, CHECKPOINT_PATH, REPORTS_DIR
from llm_layer.grammar_correction import _pick_provider, gloss_to_sentence
from speech.text_to_speech import synthesize_mp3

st.set_page_config(page_title="AI Sign Language Translator", page_icon="🤟", layout="wide")


@st.cache_resource
def load_predictor():
    from core.model import Predictor
    return Predictor(CHECKPOINT_PATH)


def new_extractor():
    from core.landmarks import LandmarkExtractor
    return LandmarkExtractor()


def show_sentence(sentence, backend, speak=True):
    st.markdown(f"### 🗣️ {sentence}")
    st.caption(f"grammar backend: {backend}")
    if speak and sentence:
        audio = synthesize_mp3(sentence)
        if audio:
            st.audio(audio, format="audio/mp3")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🤟 Sign Translator")
predictor = None
try:
    predictor = load_predictor()
    st.sidebar.success(f"Model loaded: {len(predictor.classes)} signs")
    if predictor.metrics.get("val_acc") is not None:
        st.sidebar.metric("Validation accuracy", f"{predictor.metrics['val_acc']:.1%}")
    with st.sidebar.expander("Known signs"):
        st.write(", ".join(predictor.classes))
except Exception as e:
    st.sidebar.error(f"No model: {e}")

st.sidebar.caption(f"LLM provider: **{_pick_provider()}** (set SIGN_LLM_PROVIDER / API key)")
CFG.inference.confidence_threshold = st.sidebar.slider(
    "Confidence threshold", 0.1, 0.95, CFG.inference.confidence_threshold, 0.05)
speak_out = st.sidebar.toggle("Speak result", value=True)

live_tab, upload_tab, gloss_tab, report_tab = st.tabs(
    ["📷 Live webcam", "🎞️ Upload video", "✍️ Gloss → sentence", "📊 Model report"])

# --------------------------------------------------------------------- live
with live_tab:
    try:
        import av
        from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
        has_webrtc = True
    except ImportError:
        has_webrtc = False

    if predictor is None:
        st.info("Train a model first: `python -m training.train_transformer`")
    elif not has_webrtc:
        st.warning("Install `streamlit-webrtc` and `av` for live webcam in the browser, "
                   "or run `python -m inference.sentence_detection` locally.")
    else:
        from core.landmarks import draw_landmarks
        from core.recognizer import StreamingRecognizer

        class Processor(VideoProcessorBase):
            def __init__(self):
                self.ex = new_extractor()
                self.rec = StreamingRecognizer(predictor.predict, CFG.inference)
                self.lock = threading.Lock()
                self.finished: list[list[str]] = []

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                vec, dbg = self.ex.process(img)
                with self.lock:
                    for ev in self.rec.push(vec):
                        if ev.kind == "sentence_end" and ev.words:
                            self.finished.append(ev.words)
                out = draw_landmarks(img, dbg)[:, ::-1].copy()
                return av.VideoFrame.from_ndarray(out, format="bgr24")

        c1, c2 = st.columns([3, 2])
        with c1:
            ctx = webrtc_streamer(key="sign", mode=WebRtcMode.SENDRECV,
                                  video_processor_factory=Processor,
                                  media_stream_constraints={"video": True, "audio": False},
                                  rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        with c2:
            st.markdown("**How to use:** sign a word, drop hands briefly, sign the next. "
                        "Keep hands down ~1.5 s to finish the sentence.")
            status, gloss_box, sent_box = st.empty(), st.empty(), st.empty()
            if "live_history" not in st.session_state:
                st.session_state.live_history = []
            while ctx.state.playing and ctx.video_processor:
                vp = ctx.video_processor
                with vp.lock:
                    words, top, signing = list(vp.rec.words), vp.rec.last_top_k, vp.rec.in_sign
                    done, vp.finished = vp.finished, []
                status.markdown(f"**State:** {'✋ signing' if signing else 'waiting'}"
                                + (f" — last guess **{top[0][0]}** {top[0][1]:.0%}" if top else ""))
                gloss_box.markdown("**Gloss:** " + " ".join(w.upper() for w in words))
                for ws in done:
                    s, b = gloss_to_sentence(ws)
                    st.session_state.live_history.append(s)
                    sent_box.markdown(f"### 🗣️ {s}")
                time.sleep(0.2)
            if st.session_state.live_history:
                st.write("**Conversation**")
                for s in st.session_state.live_history:
                    st.write("•", s)

# ------------------------------------------------------------------- upload
with upload_tab:
    st.write("Upload a clip of one sign or several signs in a row.")
    mode = st.radio("Mode", ["auto", "word", "sentence"], horizontal=True)
    up = st.file_uploader("Video", type=["mp4", "avi", "mov", "webm", "mkv"])
    if up and predictor is not None:
        st.video(up)
        with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
            f.write(up.getbuffer())
        with st.spinner("Extracting landmarks and translating..."):
            from core.pipeline import translate_video
            with new_extractor() as ex:
                res = translate_video(f.name, predictor, ex, mode)
        Path(f.name).unlink(missing_ok=True)
        if not res["words"]:
            st.warning("No confident sign found. Check lighting / that hands are visible.")
        else:
            st.markdown("**Gloss:** " + " ".join(w.upper() for w in res["words"]))
            show_sentence(res["sentence"], res["llm_backend"], speak_out)
        with st.expander("Details"):
            st.json(res)

# -------------------------------------------------------------------- gloss
with gloss_tab:
    txt = st.text_input("Type a gloss sequence", "ME HAPPY SEE YOU")
    if st.button("Translate"):
        s, b = gloss_to_sentence(txt.split())
        show_sentence(s, b, speak_out)

# ------------------------------------------------------------------- report
with report_tab:
    mpath = REPORTS_DIR / "metrics.json"
    if not mpath.exists():
        st.info("Run `python -m training.evaluate` to generate the report.")
    else:
        m = json.loads(mpath.read_text())
        c = st.columns(4)
        c[0].metric("Test top-1", f"{m['top1_accuracy']:.1%}")
        c[1].metric("Test top-5", f"{m['top5_accuracy']:.1%}")
        c[2].metric("Macro F1", f"{m['macro_f1']:.3f}")
        c[3].metric("Classes", m["num_classes"])
        for img in ["training_curves.png", "confusion_matrix.png"]:
            if (REPORTS_DIR / img).exists():
                st.image(str(REPORTS_DIR / img))
        for txtf in ["worst_confusions.txt", "classification_report.txt"]:
            if (REPORTS_DIR / txtf).exists():
                with st.expander(txtf):
                    st.code((REPORTS_DIR / txtf).read_text())
