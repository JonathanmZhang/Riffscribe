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

# Evaluation tooling only (scripts/egset12_benchmark.py): Spotify's pedalboard,
# for the processed-distortion versions of the EGSet12 benchmark.
RUN pip install --no-cache-dir pedalboard==0.9.16

# Chord names (tasks/chords.py): BTC (MIT, code and weights), cloned at a
# pinned commit; only the large-vocabulary checkpoint is kept. It runs on
# the packages above (torch, librosa, mir_eval, PyYAML). Its numpy aliases
# (np.float/np.int/np.bool) were removed in numpy 1.24; the builtins are
# exact replacements.
ARG BTC_SHA=2682317be668032e6e4b269ded36adaa2ad57df0
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && git clone https://github.com/jayg996/BTC-ISMIR19 /opt/btc && git -C /opt/btc checkout $BTC_SHA \
    && sed -i -E 's/\bnp\.(float|int|bool)\b/\1/g' /opt/btc/utils/*.py /opt/btc/*.py \
    && rm -rf /opt/btc/.git /opt/btc/png /opt/btc/test/example.mp3 /opt/btc/test/btc_model.pt \
    && apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY tasks/ ./tasks/
COPY scripts/ ./scripts/
CMD ["celery", "-A", "tasks.celery_app", "worker", "--loglevel=info"]
