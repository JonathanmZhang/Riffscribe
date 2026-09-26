FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Guitar separation (Demucs), in separate layers so the large TensorFlow layer
# above stays cached. CPU-only PyTorch from its own index; pinned as a
# matching pair that works with numpy 1.x and Demucs 4.0.1.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 torchaudio==2.5.1
COPY requirements-separation.txt .
RUN pip install --no-cache-dir -r requirements-separation.txt

# Bake the default model's weights into the image so they aren't downloaded
# on every container start or code rebuild (this layer only re-runs when the
# dependency layers above change).
ENV TORCH_HOME=/app/models
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_6s')"

COPY tasks/ ./tasks/
COPY scripts/ ./scripts/
CMD ["celery", "-A", "tasks.celery_app", "worker", "--loglevel=info"]
