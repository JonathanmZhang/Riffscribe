# Beat trackers for the rhythm measurement, on top of the unchanged worker
# image (experiment only).
#   docker build -t rhythm-bench experiments/rhythm
# beat_this (CPJKU, ISMIR 2024; MIT code and weights): pip dry-run against the
# worker adds only beat-this and rotary-embedding-torch (torch 2.5.1, numpy,
# TF untouched). Its "final0" checkpoint is baked in (TORCH_HOME=/app/models
# from the worker image) instead of downloading on every run.
FROM stratotab-worker
RUN pip install --no-cache-dir beat-this==1.1.0 "protobuf<5"
RUN python -c "from beat_this.inference import File2Beats; File2Beats(checkpoint_path='final0', device='cpu', dbn=False)"
