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

**A. WLASL** — put videos as `datasets/WLASL/dataset/SL/<word>/<id>.mp4`
(from the Kaggle "WLASL processed" download this is done with symlinks, no copy), then:
```bash
python -m scripts.prepare_wlasl --src ~/.cache/kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5
python -m data_collection.extract_landmarks --max-classes 50 --workers 4   # → datasets/landmarks/
```

**B. Your own signs** (fastest, best for a live demo):
```bash
python -m data_collection.record_signs --signs hello thanks yes no please sorry help --per-sign 30
```
Clips go to `datasets/own_landmarks/<sign>/` (separate from WLASL, so the benchmark stays clean).
Record with different lighting, distance, clothes and both hands. 30+ clips per sign.

### 2. Train and evaluate
```bash
python -m training.train_transformer   # WLASL only → models/sign_transformer.pt
python -m training.evaluate          # → reports/metrics.json, confusion_matrix.png, ...

# combined model (WLASL + your own recordings) for the live demo
python -m training.train_transformer --data datasets/landmarks datasets/own_landmarks \
    --out models/combined.pt --reports reports/combined
python -m training.evaluate --model models/combined.pt --reports reports/combined
```

### 3. Run it
```bash
python -m inference.realtime_detection      # word level
python -m inference.sentence_detection      # words → sentence → speech
streamlit run dashboard/app.py              # web dashboard
uvicorn api.server:app --port 8000          # REST API → http://localhost:8000/docs
```
All of these load `models/sign_transformer.pt` by default. To use another checkpoint:
```bash
python -m inference.realtime_detection --model models/combined.pt
python -m inference.sentence_detection --model models/combined.pt
SIGN_MODEL_PATH=models/combined.pt streamlit run dashboard/app.py
SIGN_MODEL_PATH=models/combined.pt uvicorn api.server:app --port 8000
```
(`SIGN_MODEL_PATH` may be absolute or relative to the project root.)

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

## Datasets & licenses

| Dataset | Used for | License / terms |
|---|---|---|
| [WLASL](https://dxli94.github.io/WLASL/) (Kaggle "WLASL processed") | benchmark + combined model | Computational Use of Data Agreement (C-UDA): academic, non-commercial use |
| [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (Microsoft Research) | combined model (webcam signers, official signer-independent splits) | Microsoft Research License: **non-commercial research only**, no redistribution |

- **No videos are redistributed.** Videos, landmarks and checkpoints trained on them stay local (`datasets/` and `models/*.pt` are git-ignored). Only code and aggregate results are published.
- ASL Citizen is fetched with `python -m scripts.prepare_asl_citizen`, which downloads only the videos of the classes you use (not the 46 GB archive). Commercial use requires contacting Microsoft (ASL_Citizen@microsoft.com).
- If you publish results, cite *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition* (Desai et al., 2023) and *Word-level Deep Sign Language Recognition from Video* (Li et al., 2020, WLASL).

## Design decisions

- **Landmarks, not pixels.** 147 numbers per frame instead of a full image: trains on a laptop, robust to background and skin tone, runs in real time on CPU.
- **Hands put in fixed left/right slots** by MediaPipe handedness, plus upper-body pose, because where a sign happens relative to the body changes its meaning.
- **Body-relative normalisation** (centred on shoulders, scaled by shoulder width), so distance from camera and position in frame don't matter. Covered by a unit test.
- **Augmentation:** speed change, temporal crop, rotation, scale, jitter, left/right mirroring, simulated tracking loss.
- **Hand-presence segmentation** instead of a blind sliding window, so one sign gives one prediction and the confidence threshold rejects garbage.
- **Held-out test split** saved in `reports/split.json`; the reported accuracy is never on training data.

## Results

Numbers from `python -m training.evaluate` on the held-out test split (never trained on).

**WLASL only** (benchmark: no own recordings mixed in)

| Dataset | Classes | Clips (train/val/test) | Test top-1 | Test top-5 | Macro F1 |
|---|---|---|---|---|---|
| WLASL, 50 classes with most videos | 50 | 614 (429/92/93) | 60.2 % | 87.1 % | 0.547 |

Default config (d_model 192, 4 layers, 1.3 M params), best val acc 81.5 % (epoch 100).
Only ~2 test clips per class, so expect a few points of noise.

**Combined model: WLASL + ASL Citizen** (`models/combined.pt`, for the live webcam demo)

Same 50 classes. ASL Citizen adds 2,152 webcam clips from 52 signers for 48 of them (`short`
and `trade` have no ASL Citizen match; `delay` = `POSTPONE`). ASL Citizen uses its official
signer-independent splits, so its **test signers are never seen in training**. The WLASL
train/val/test split is identical to the benchmark above, so the WLASL rows are directly comparable.

| Model | Test set | Clips | Test top-1 | Test top-5 | Macro F1 |
|---|---|---|---|---|---|
| WLASL-only (benchmark) | WLASL | 93 | 60.2 % | 87.1 % | 0.547 |
| WLASL-only (benchmark) | ASL Citizen, unseen signers | 855 | 65.7 % | 83.0 % | 0.638 |
| WLASL-only (benchmark) | overall | 948 | 65.2 % | 83.4 % | 0.638 |
| **Combined** | WLASL | 93 | 72.0 % | 90.3 % | 0.677 |
| **Combined** | ASL Citizen, unseen signers | 855 | **94.4 %** | 98.9 % | 0.924 |
| **Combined** | overall | 948 | 92.2 % | 98.1 % | 0.892 |

```bash
python -m scripts.prepare_asl_citizen --map delay=POSTPONE what=WHAT1+WHAT2   # ~1.1 GB, only these classes
python -m data_collection.extract_landmarks --src datasets/ASL_CITIZEN --out datasets/asl_citizen_landmarks --workers 4
python -m training.train_transformer --data datasets/landmarks datasets/asl_citizen_landmarks \
    --out models/combined.pt --reports reports/combined
python -m training.evaluate --model models/combined.pt --reports reports/combined
# WLASL-only model on the same combined test set:
python -m training.evaluate --split reports/combined/split.json --reports reports/wlasl_model_on_combined_test
```
Folders with the same class name are merged into one class; any folder can be added the same
way (e.g. `datasets/own_landmarks`).

## Limitations / roadmap
- Isolated-sign classifier + segmentation, not true continuous sign language recognition (CSLR, e.g. CTC on How2Sign / PHOENIX).
- No facial expressions yet (important for grammar in ASL); next step is adding face landmarks.
- Indian Sign Language (ISL) support via the INCLUDE dataset.
- Export to ONNX / TFLite for mobile; run MediaPipe in the browser and call `/translate/landmarks`.
