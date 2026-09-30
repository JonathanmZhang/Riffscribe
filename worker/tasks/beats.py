"""Beat and downbeat times for the tab: beat_this (Foscarin, Schlüter,
Widmer, ISMIR 2024, "Beat this! Accurate beat tracking without DBN
postprocessing"; MIT code and weights), "final0" checkpoint, without DBN
postprocessing (that needs madmom, whose models are non-commercial). The
checkpoint is baked into the image (worker/Dockerfile, TORCH_HOME).

Pure: no Redis or Celery, so scripts can call track_beats directly.
Measured on EGSet12 against beats from its Guitar Pro files
(experiments/rhythm/README.md): beat F 65-70% at +-70ms and tempo right for
8/12 performances in every tone, where librosa's beat_track (what the app
used before) got 55-62% and 6-8/12. Its misses are half tempo on fast
songs (140-176 bpm). Downbeats are ~54% right, so bar lines from it alone
are unreliable.
"""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

BEAT_THIS_CHECKPOINT = "final0"

_model: dict = {}


def _load():
    """Loads beat_this once per process."""
    if not _model:
        from beat_this.inference import File2Beats

        _model["f2b"] = File2Beats(checkpoint_path=BEAT_THIS_CHECKPOINT, device="cpu", dbn=False)
    return _model["f2b"]


def track_beats(audio_path: str) -> tuple[list[float], list[float]]:
    """(beat times, downbeat times) in seconds for audio_path, rounded to
    the millisecond. Downbeats are a subset of the beats (bar starts)."""
    beats, downbeats = _load()(audio_path)
    return [round(float(b), 3) for b in beats], [round(float(d), 3) for d in downbeats]


def tempo_from_beats(beats: list[float]) -> int:
    """Tempo as 60 / median inter-beat interval, rounded to a whole BPM; 0
    with fewer than two beats. The median keeps a few missed or extra beats
    from moving it."""
    if len(beats) < 2:
        return 0
    return int(round(60.0 / float(np.median(np.diff(beats)))))


def librosa_tempo_bpm(audio_path: str) -> int:
    """Fallback when beat_this fails: librosa's global tempo estimate (what
    the app showed before beat_this), rounded to a whole BPM; 0 when no beat
    is found. Less reliable: on solo guitar it often comes out at double
    tempo or locks onto the off-beats."""
    audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    tempo, _beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
    # librosa >= 0.10 returns tempo as a 1-element array, older versions a scalar.
    return int(round(float(np.atleast_1d(tempo)[0])))


def estimate_rhythm(audio_path: str) -> dict:
    """{"tempo_bpm", "beats", "downbeats"} for audio_path. beat_this
    first; if it raises, logs the error and falls back to librosa's tempo
    with empty beat lists, so a beat-tracking failure never fails a job."""
    try:
        beats, downbeats = track_beats(audio_path)
        return {"tempo_bpm": tempo_from_beats(beats), "beats": beats, "downbeats": downbeats}
    except Exception:
        logger.exception("beat_this failed on %s; falling back to librosa's tempo without beat times", audio_path)
    return {"tempo_bpm": librosa_tempo_bpm(audio_path), "beats": [], "downbeats": []}
