# Chord recognizers on top of the unchanged worker image (experiment only).
#   docker build -t chords-bench experiments/chord_recognition
# pip dry-run against the worker: crema/pumpp/jams/vamp only add packages
# (torch 2.5.1, numpy 1.26.4, TF 2.15 / keras 2.15 untouched); protobuf is
# pinned < 5 because TF 2.15 breaks on protobuf 7.
FROM stratotab-worker
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential vamp-plugin-sdk libboost-dev \
    && rm -rf /var/lib/apt/lists/*

# crema (ISC; the chord model ships inside the package). Its pumpp dep
# subclasses sklearn's LabelEncoder and breaks on scikit-learn >= 1.6
# (__sklearn_tags__), so scikit-learn is downgraded from the worker's 1.9.1.
RUN pip install --no-cache-dir crema==0.2.0 vamp==1.1.0 "protobuf<5" scikit-learn==1.5.2

# BTC, Park et al. ISMIR 2019 (MIT; checkpoints in the repo's test/).
ARG BTC_SHA=2682317be668032e6e4b269ded36adaa2ad57df0
# Its numpy aliases (np.float/np.int/np.bool) were removed in numpy 1.24;
# the builtins are exact replacements.
RUN git clone https://github.com/jayg996/BTC-ISMIR19 /opt/btc && git -C /opt/btc checkout $BTC_SHA \
    && sed -i -E 's/\bnp\.(float|int|bool)\b/\1/g' /opt/btc/utils/*.py /opt/btc/*.py

# Chordino / NNLS-Chroma Vamp plugin (GPL-2.0), built from source.
ARG NNLS_SHA=4c5f214a75cb354f8d8c933e161377c5b4d83713
RUN git clone https://github.com/c4dm/nnls-chroma /opt/nnls-chroma && git -C /opt/nnls-chroma checkout $NNLS_SHA \
    && cd /opt/nnls-chroma \
    && make -f Makefile.linux VAMP_SDK_DIR=/usr/lib/x86_64-linux-gnu BOOST_ROOT=/usr/include \
    && mkdir -p /usr/local/lib/vamp && cp nnls-chroma.so nnls-chroma.cat nnls-chroma.n3 /usr/local/lib/vamp/
ENV VAMP_PATH=/usr/local/lib/vamp
