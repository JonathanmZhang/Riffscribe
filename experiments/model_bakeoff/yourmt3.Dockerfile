# YourMT3+ on top of the unchanged worker image: its extra deps only add
# packages (pip dry-run: torch 2.5.1 / numpy 1.26.4 / TF 2.15 untouched,
# `pip check` clean). Build context: experiments/model_bakeoff/.
#   docker build -f yourmt3.Dockerfile -t bakeoff-yourmt3 .
FROM stratotab-worker
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# The Space's requirements.txt minus the web app / YouTube bits;
# torch/torchaudio/numpy/librosa come from the worker image. wandb is imported
# unconditionally by model/ymt3.py (logging is disabled at runtime); unpinned
# it pulls protobuf 7, which breaks TensorFlow 2.15 (needs protobuf < 5), so
# protobuf is pinned and pip settles on wandb 0.28.0.
RUN pip install --no-cache-dir "lightning>=2.2.1" transformers==4.45.1 einops mido deprecated wandb "protobuf<5" \
    "mir_eval @ git+https://github.com/craffel/mir_eval.git"
# Code + the default checkpoint (YPTF.MoE+Multi noPS, 562 MB) from the
# Hugging Face Space at a pinned commit, without the other checkpoints.
ARG SPACE_SHA=5e66c1ea173a8186e0d20432b841d3180cc015b5
ARG CKPT=amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt
RUN pip install --no-cache-dir "huggingface_hub[hf_xet]" && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('mimbres/YourMT3', repo_type='space', revision='$SPACE_SHA', local_dir='/opt/yourmt3', \
    allow_patterns=['amt/src/**', 'model_helper.py', 'README.md', '$CKPT'])"
WORKDIR /opt/yourmt3
