# MT3 (Magenta, JAX/T5X) in its own container: it can't go in the worker
# image - its unpinned git deps (flax/t5x HEAD) need jax >= 0.11, which needs
# Python >= 3.12 (the worker is 3.11), plus its own TensorFlow.
# The official Colab installs unpinned HEAD (`pip install jax[cuda12] -e .`);
# mt3-requirements.lock.txt is the `pip freeze` of that install with CPU jax
# (2026-09-29) plus tensorboard, which t5x imports but doesn't declare. It is
# installed --no-deps: HEAD pulls mixed sets (e.g. seqio and seqio-nightly)
# that pip's resolver rejects when asked to re-resolve them.
#   docker build -f mt3.Dockerfile -t bakeoff-mt3 .
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libasound2-dev libjack-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*
ARG MT3_SHA=88f4b3d63d7d1ce35a6e380ec190debbbd9b4b4d
RUN git clone https://github.com/magenta/mt3 /opt/mt3 && git -C /opt/mt3 checkout $MT3_SHA
COPY mt3-requirements.lock.txt /opt/
RUN pip install --no-cache-dir --no-deps -r /opt/mt3-requirements.lock.txt && pip install --no-cache-dir --no-deps -e /opt/mt3
# Official multi-instrument checkpoint (gs://mt3/checkpoints/mt3, public bucket).
RUN python - <<'EOF'
import json, os, urllib.request
base = "https://storage.googleapis.com/storage/v1/b/mt3/o"
items, token = [], None
while True:
    url = base + "?prefix=checkpoints/mt3/&fields=nextPageToken,items(name)" + (f"&pageToken={token}" if token else "")
    page = json.load(urllib.request.urlopen(url))
    items += [i["name"] for i in page.get("items", []) if not i["name"].endswith("/")]
    token = page.get("nextPageToken")
    if not token:
        break
for name in items:
    dest = os.path.join("/opt", name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve("https://storage.googleapis.com/mt3/" + urllib.request.quote(name), dest)
print(len(items), "checkpoint files")
EOF
WORKDIR /opt/mt3
