# TabCNN trained with GuitarProFX (Pedroza et al., DAFx 2024) on top of the
# unchanged worker image. The checkpoint is a whole pickled amt-tools model,
# so it needs amt-tools (MIT). Unpinned, amt-tools' deps (mirdata ->
# smart_open -> google-cloud-storage) pull protobuf 7, which breaks
# TensorFlow 2.15, so protobuf is pinned below 5.
#   docker build -f tabcnn.Dockerfile -t bakeoff-tabcnn .
FROM stratotab-worker
# amt-tools --no-deps: its pynput dep (live-demo keyboard input) needs evdev
# built against kernel headers; amt-tools already tolerates it missing.
# sounddevice is imported unconditionally, and loads PortAudio at import.
RUN apt-get update && apt-get install -y --no-install-recommends libportaudio2 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --no-deps amt-tools==0.3.2 && \
    pip install --no-cache-dir jams mirdata sacred tensorboardX matplotlib sounddevice "protobuf<5"
# Weights: EGSet12's Zenodo record (CC BY 4.0), "best_TabCNN_tablature_trancription_model".
ADD "https://zenodo.org/records/11406378/files/best_TabCNN_tablature_trancription_model?download=1" /opt/tabcnn/model.pt
