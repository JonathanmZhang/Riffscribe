# BTC chord recognizer (Park et al., ISMIR 2019; MIT, code and weights) on top
# of the unchanged worker image. Measurement only.
#   docker build -t chord-names experiments/chord_names
# Needs nothing beyond the worker's own packages (torch, librosa, mir_eval,
# PyYAML), so no dependency changes.
FROM stratotab-worker
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# BTC's numpy aliases (np.float/np.int/np.bool) were removed in numpy 1.24;
# the builtins are exact replacements.
ARG BTC_SHA=2682317be668032e6e4b269ded36adaa2ad57df0
RUN git clone https://github.com/jayg996/BTC-ISMIR19 /opt/btc && git -C /opt/btc checkout $BTC_SHA \
    && sed -i -E 's/\bnp\.(float|int|bool)\b/\1/g' /opt/btc/utils/*.py /opt/btc/*.py
