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

# Evaluation tooling only (scripts/make_chord_testset.py): FluidSynth and the
# FluidR3_GM General MIDI soundfont (MIT license), from Debian packages at
# build time rather than committed. Its own layer after the ML layers so
# those stay cached. Not used by the pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends fluidsynth fluid-soundfont-gm \
    && rm -rf /var/lib/apt/lists/*

# Time-stretch experiment only (scripts/chord_stages.py detect speed): the
# Rubber Band CLI and its Python wrapper. Not used by the pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends rubberband-cli \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir pyrubberband==0.4.0

COPY tasks/ ./tasks/
COPY scripts/ ./scripts/
CMD ["celery", "-A", "tasks.celery_app", "worker", "--loglevel=info"]
