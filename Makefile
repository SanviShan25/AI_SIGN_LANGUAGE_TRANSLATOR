.PHONY: setup models extract train eval live sentence dashboard api test smoke

setup:      ; pip install -r requirements.txt
models:     ; python -m scripts.download_models
extract:    ; python -m data_collection.extract_landmarks --max-classes 50 --workers 4
train:      ; python -m training.train_transformer
eval:       ; python -m training.evaluate
live:       ; python -m inference.realtime_detection
sentence:   ; python -m inference.sentence_detection
dashboard:  ; streamlit run dashboard/app.py
api:        ; uvicorn api.server:app --reload --port 8000
test:       ; pytest -q
# end-to-end check without any videos
smoke:
	python -m scripts.make_synthetic_data --out datasets/synthetic
	python -m training.train_transformer --data datasets/synthetic --epochs 15
	python -m training.evaluate
