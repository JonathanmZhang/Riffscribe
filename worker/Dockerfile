# High-quality separation weights (MVSep Mega 53 Stems, tasks/mega53.py),
# downloaded only when HQ_SEPARATION=1 (docker-compose.gpu.yml). Its own
# stage, so the 1.37 GB release checkpoint is cached apart from the code
# and never lands in the final image: the worker stage below mounts it and
# keeps a two-head checkpoint. The files are release assets of
# ZFTurbo/Music-Source-Separation-Training v1.0.21, pinned by URL and
# SHA-256. They are not covered by this repo's MIT license (README).
FROM python:3.11-slim AS mega53-weights
ARG HQ_SEPARATION=0
ARG MEGA53_CKPT_URL=https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.21/mvsep_mega_model_bs_roformer_53_stems_v1.ckpt
ARG MEGA53_CKPT_SHA256=c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f
ARG MEGA53_CONFIG_URL=https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.21/mvsep_mega_model_bs_roformer_53_stems.yaml
ARG MEGA53_CONFIG_SHA256=7e198062a251587088adb91215a4f44ab59e67bd62fcc805cf54d6e7dfc51103
RUN mkdir /mega53 && if [ "$HQ_SEPARATION" = "1" ]; then \
        python -c "import sys, urllib.request as u; u.urlretrieve(sys.argv[1], '/mega53/full.ckpt'); u.urlretrieve(sys.argv[2], '/mega53/config.yaml')" \
            "$MEGA53_CKPT_URL" "$MEGA53_CONFIG_URL" \
        && echo "$MEGA53_CKPT_SHA256  /mega53/full.ckpt" | sha256sum -c - \
        && echo "$MEGA53_CONFIG_SHA256  /mega53/config.yaml" | sha256sum -c - ; \
    fi

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

# Beat tracking (tasks/beats.py): beat_this (MIT, code and weights). It
# adds only beat-this and rotary-embedding-torch on top of the torch above;
# protobuf stays < 5 for TensorFlow 2.15. Its "final0" checkpoint is baked
# in (TORCH_HOME=/app/models) instead of downloading on first use.
COPY requirements-beats.txt .
RUN pip install --no-cache-dir -r requirements-beats.txt \
    && python -c "from beat_this.inference import File2Beats; File2Beats(checkpoint_path='final0', device='cpu', dbn=False)"

# High-quality separation (tasks/mega53.py), only with HQ_SEPARATION=1: the
# worker-separation image of docker-compose.gpu.yml. With the default 0
# these layers do nothing, and the image is the CPU one above. They come
# after the shared layers so both images reuse those.
# 1. PyTorch's CUDA 12.1 wheels in place of the CPU ones, same version
#    (uninstall first: pip treats the CPU wheel as satisfying torch==2.5.1).
#    The wheels bundle the CUDA runtime; the host needs only an NVIDIA
#    driver and Docker's GPU support.
ARG HQ_SEPARATION=0
RUN if [ "$HQ_SEPARATION" = "1" ]; then \
        pip uninstall -y torch torchaudio \
        && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
            torch==2.5.1 torchaudio==2.5.1 ; \
    fi
# 2. MSST (MIT), the model code and chunked inference, at a pinned commit.
ARG MSST_SHA=84b1eac0887756b4f1a9d7a1ff49105939749ed2
COPY requirements-separation-hq.txt .
RUN if [ "$HQ_SEPARATION" = "1" ]; then \
        pip install --no-cache-dir -r requirements-separation-hq.txt \
        && apt-get update && apt-get install -y --no-install-recommends git \
        && git clone https://github.com/ZFTurbo/Music-Source-Separation-Training /opt/msst \
        && git -C /opt/msst checkout $MSST_SHA && rm -rf /opt/msst/.git \
        && apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* ; \
    fi
# 3. The two-head checkpoint, cut from the downloaded one (mounted from the
#    stage above, not copied).
COPY tasks/mega53.py /tmp/mega53.py
RUN --mount=type=bind,from=mega53-weights,source=/mega53,target=/mnt/mega53 \
    if [ "$HQ_SEPARATION" = "1" ]; then \
        python /tmp/mega53.py strip /mnt/mega53/full.ckpt /mnt/mega53/config.yaml /app/models/mega53 ; \
    fi

COPY tasks/ ./tasks/
COPY scripts/ ./scripts/

# Build provenance for the measurement scripts' stale-image guard
# (scripts/build_info.py): the commit this image was built from, read from
# the repo's .git (the "gitmeta" build context in docker-compose.yml), and a
# hash of the code above. Last, so a new commit only re-runs this layer.
COPY --from=gitmeta HEAD packed-ref[s] /tmp/gitmeta/
COPY --from=gitmeta refs /tmp/gitmeta/refs
RUN python -m scripts.build_info record && rm -rf /tmp/gitmeta
CMD ["celery", "-A", "tasks.celery_app", "worker", "--loglevel=info"]
