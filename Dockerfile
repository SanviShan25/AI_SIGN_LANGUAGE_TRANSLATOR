FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m scripts.download_models
EXPOSE 8000 8501
# API by default; for the dashboard:  docker run -p 8501:8501 IMAGE streamlit run dashboard/app.py --server.address 0.0.0.0
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
