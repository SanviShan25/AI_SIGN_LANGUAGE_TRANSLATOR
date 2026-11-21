# 🤟 AI Sign Language Translator

Real-time **sign language → English text → speech**, from a normal webcam.

```
Webcam / video
   │  MediaPipe (hands + upper body)            core/landmarks.py
   ▼
Landmarks (T × 147)
   │  body-relative normalisation, hand shape,  core/preprocessing.py
   │  time resampling, augmentation
   ▼
Transformer encoder classifier                  core/model.py
   │
   ▼
Streaming segmenter: one sign = one word        core/recognizer.py
   │  "ME  HAPPY  SEE  YOU"
   ▼
LLM grammar layer (Claude / OpenAI / Groq /     llm_layer/grammar_correction.py
Ollama, offline rule-based fallback)
   │  "I'm happy to see you."
   ▼
Text-to-speech                                  speech/text_to_speech.py
   │
   ▼
OpenCV app · Streamlit dashboard · REST API     inference/ · dashboard/ · api/
```

## Project structure

| Folder | What it does |
|---|---|
| `core/` | config, landmark extraction, features, model, streaming recogniser, pipeline (shared by everything) |
| `data_collection/` | `extract_landmarks.py` (videos → .npy), `record_signs.py` (webcam → .npy), `collect_data.py` (webcam → video) |
| `training/` | `train_transformer.py`, `evaluate.py` (held-out test set, confusion matrix, curves) |
| `inference/` | `realtime_detection.py` (words), `sentence_detection.py` (full pipeline with speech) |
| `llm_layer/` | gloss → natural sentence |
| `speech/` | offline + online TTS |
| `dashboard/` | Streamlit web app (live webcam, upload, metrics) |
| `api/` | FastAPI server for apps / websites |
| `scripts/` | model download, synthetic data |
| `tests/` | pytest unit tests |

## Quick start

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.download_models    # needed for mediapipe >= 0.10.30 (Tasks API)
python test_mediapipe.py             # check webcam + hand tracking
pytest -q                            # unit tests
make smoke                           # whole train → eval pipeline on synthetic data (no videos needed)
```

### 1. Get data (pick one or both)

**A. WLASL** — put videos as `datasets/WLASL/dataset/SL/<word>/<id>.mp4`, then:
```bash
python -m data_collection.extract_landmarks --max-classes 50 --workers 4
```

**B. Your own signs** (fastest, best for a live demo):
```bash
python -m data_collection.record_signs --signs hello thanks yes no please sorry help --per-sign 30
```
Record with different lighting, distance, clothes and both hands. 30+ clips per sign.

### 2. Train and evaluate
```bash
python -m training.train_transformer
python -m training.evaluate          # → reports/metrics.json, confusion_matrix.png, ...
```

### 3. Run it
```bash
python -m inference.realtime_detection      # word level
python -m inference.sentence_detection      # words → sentence → speech
streamlit run dashboard/app.py              # web dashboard
uvicorn api.server:app --port 8000          # REST API → http://localhost:8000/docs
```

**How to sign for the live app:** sign a word, drop your hands out of view for a moment, sign the next word. Keep hands down ~1.5 s (or press ENTER) to finish the sentence.

### LLM layer (optional)
Works offline with a rule-based fallback. For proper grammar set one of:
```bash
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY / GROQ_API_KEY
export SIGN_LLM_PROVIDER=ollama     # free + local, needs `ollama run llama3.1`
```

### Docker
```bash
docker build -t sign-translator .
docker run -p 8000:8000 -v $PWD/models:/app/models sign-translator
```

## Design decisions

- **Landmarks, not pixels.** 147 numbers per frame instead of a full image: trains on a laptop, robust to background and skin tone, runs in real time on CPU.
- **Hands put in fixed left/right slots** by MediaPipe handedness, plus upper-body pose, because where a sign happens relative to the body changes its meaning.
- **Body-relative normalisation** (centred on shoulders, scaled by shoulder width), so distance from camera and position in frame don't matter. Covered by a unit test.
- **Augmentation:** speed change, temporal crop, rotation, scale, jitter, left/right mirroring, simulated tracking loss.
- **Hand-presence segmentation** instead of a blind sliding window, so one sign gives one prediction and the confidence threshold rejects garbage.
- **Held-out test split** saved in `reports/split.json`; the reported accuracy is never on training data.

## Results

Fill this in after `python -m training.evaluate`:

| Dataset | Classes | Test top-1 | Test top-5 | Macro F1 |
|---|---|---|---|---|
| own recordings | – | – | – | – |
| WLASL-100 | 100 | – | – | – |

## Limitations / roadmap
- Isolated-sign classifier + segmentation, not true continuous sign language recognition (CSLR, e.g. CTC on How2Sign / PHOENIX).
- No facial expressions yet (important for grammar in ASL); next step is adding face landmarks.
- Indian Sign Language (ISL) support via the INCLUDE dataset.
- Export to ONNX / TFLite for mobile; run MediaPipe in the browser and call `/translate/landmarks`.
