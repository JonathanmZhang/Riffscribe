import bisect
import logging
import os

import librosa
import numpy as np
import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from basic_pitch.note_creation import model_frames_to_time

from tasks.celery_app import app
from tasks.fretboard import CHORD_ONSET_TOLERANCE_SECONDS
from tasks.storage import get_job, update_job

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = float(os.environ.get("BASIC_PITCH_CONFIDENCE_THRESHOLD", "0.5"))
# Lower confidence bar for notes that start with a confident note (chord
# tones); see select_notes. Values >= CONFIDENCE_THRESHOLD disable it.
CHORD_TONE_CONFIDENCE_FLOOR = float(os.environ.get("CHORD_TONE_CONFIDENCE_FLOOR", "0.45"))
# Re-trigger merging (see merge_retriggers). An empty gap disables it.
_merge_gap = os.environ.get("RETRIGGER_MERGE_GAP_SECONDS", "0.04")
RETRIGGER_MERGE_GAP_SECONDS = float(_merge_gap) if _merge_gap else None
RETRIGGER_MAX_ONSET_ACTIVATION = float(os.environ.get("RETRIGGER_MAX_ONSET_ACTIVATION", "0.6"))

BASIC_PITCH_MIDI_OFFSET = 21  # Basic Pitch's pitch axis starts at A0
ONSET_LOOKUP_SECONDS = 0.03


def detect_note_events(audio_path: str) -> tuple[list[dict], dict]:
    """Runs Basic Pitch on audio_path and returns (every note event, before
    any confidence filter, and the raw model output). Each event has pitch,
    midi, start_time, end_time, amplitude and onset_activation (Basic Pitch's
    peak onset activation for that pitch within 30ms of the start: how
    strongly the model heard an attack). Pure: used by extract_notes and by
    the chord inspection/evaluation scripts, which need the events the filter
    drops and the model's per-frame activations.
    """
    model_output, _, note_events = predict(audio_path, ICASSP_2022_MODEL_PATH)
    onsets = np.asarray(model_output["onset"])
    frame_times = np.asarray(model_frames_to_time(onsets.shape[0]))

    def onset_activation(start: float, midi: int) -> float:
        frames = np.abs(frame_times - start) <= ONSET_LOOKUP_SECONDS
        index = midi - BASIC_PITCH_MIDI_OFFSET
        return float(onsets[frames, index].max()) if frames.any() and 0 <= index < onsets.shape[1] else 0.0

    events = [
        {
            "pitch": pretty_midi.note_number_to_name(pitch_midi),
            "midi": int(pitch_midi),
            "start_time": start_time,
            "end_time": end_time,
            "amplitude": float(amplitude),
            "onset_activation": onset_activation(start_time, int(pitch_midi)),
        }
        for start_time, end_time, pitch_midi, amplitude, _pitch_bends in note_events
    ]
    return events, model_output


def merge_retriggers(
    events: list[dict],
    max_gap: float | None = RETRIGGER_MERGE_GAP_SECONDS,
    max_onset_activation: float = RETRIGGER_MAX_ONSET_ACTIVATION,
) -> list[dict]:
    """Merges re-triggers: when an event of the same pitch starts while the
    previous one is still sounding, or within max_gap seconds after it ends,
    they become one note (earlier start, later end, higher amplitude). With
    distortion Basic Pitch often splits one sustained note this way.

    The gap alone can't tell a re-trigger from a real repeated note - Basic
    Pitch ends a note exactly where the next same-pitch note starts in ~90%
    of both cases - so merging can be gated on the second event's onset
    activation: only events whose attack Basic Pitch heard weakly (below
    max_onset_activation) are merged. 1.0 = no gate. max_gap None disables.
    """
    if max_gap is None:
        return list(events)
    merged: list[dict] = []
    last_by_pitch: dict[int, dict] = {}
    for event in sorted(events, key=lambda e: e["start_time"]):
        previous = last_by_pitch.get(event["midi"])
        if (
            previous is not None
            and event["start_time"] <= previous["end_time"] + max_gap
            and event.get("onset_activation", 0.0) < max_onset_activation
        ):
            previous["end_time"] = max(previous["end_time"], event["end_time"])
            previous["amplitude"] = max(previous["amplitude"], event["amplitude"])
            continue
        copy = dict(event)
        merged.append(copy)
        last_by_pitch[event["midi"]] = copy
    return merged


def select_notes(
    events: list[dict],
    threshold: float = CONFIDENCE_THRESHOLD,
    chord_floor: float = CHORD_TONE_CONFIDENCE_FLOOR,
    merge_gap: float | None = RETRIGGER_MERGE_GAP_SECONDS,
    merge_max_onset: float = RETRIGGER_MAX_ONSET_ACTIVATION,
) -> list[dict]:
    """Post-detection note selection: returns copies of events, each with a
    "kept" flag. Shared by extract_notes (the pipeline) and the measurement
    scripts, so both apply exactly the same rules.

    Re-triggers are merged first (merge_retriggers), so a merged note's
    confidence is the higher of its parts.

    A note at or above `threshold` is kept. A weaker note is also kept when
    it's at least `chord_floor` AND a note at or above `threshold` starts
    within the chord onset window (CHORD_ONSET_TOLERANCE_SECONDS) of it:
    Basic Pitch tends to give the other tones of a strummed chord 0.35-0.5,
    while they arrive together with at least one confident note. Isolated
    weak notes stay filtered. chord_floor >= threshold disables the rule.
    """
    events = merge_retriggers(events, merge_gap, merge_max_onset)
    confident_onsets = sorted(e["start_time"] for e in events if e["amplitude"] >= threshold)

    def near_confident(start: float) -> bool:
        i = bisect.bisect_left(confident_onsets, start - CHORD_ONSET_TOLERANCE_SECONDS)
        return i < len(confident_onsets) and confident_onsets[i] <= start + CHORD_ONSET_TOLERANCE_SECONDS

    return [
        {
            **event,
            "kept": event["amplitude"] >= threshold
            or (chord_floor <= event["amplitude"] < threshold and near_confident(event["start_time"])),
        }
        for event in events
    ]


def extract_notes(audio_path: str) -> tuple[int, list[dict]]:
    """Runs Basic Pitch on audio_path and applies the confidence filter.
    Returns (raw note count, kept notes in raw_note_events shape). Pure: no
    Redis/Celery, so scripts/ab_separation.py runs exactly the same
    extraction as the pipeline.
    """
    events, _ = detect_note_events(audio_path)
    kept = [
        {key: event[key] for key in ("pitch", "start_time", "end_time", "amplitude")}
        for event in select_notes(events)
        if event["kept"]
    ]
    return len(events), kept


def _estimate_tempo_bpm(audio_path: str) -> int:
    """Global tempo estimate via librosa's onset-based beat tracker, rounded
    to the nearest integer BPM. This is an ESTIMATE, not a measured tempo:
    beat_track keys on percussive onsets, so it is noticeably less reliable
    on solo instrument recordings with no drums or clear pulse (legato
    playing, rubato, sparse picking), where it can lock onto half/double the
    felt tempo or onto note density instead of the beat. Returns 0 when no
    beat is found (e.g. silent audio).
    """
    audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    tempo, _beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
    # librosa >= 0.10 returns tempo as a 1-element array, older versions a scalar.
    return int(round(float(np.atleast_1d(tempo)[0])))


@app.task(name="transcribe", soft_time_limit=120)
def transcribe(job_id: str) -> str:
    try:
        update_job(job_id, status="processing", stage="transcribing")

        job = get_job(job_id)
        # separate_guitar sets transcription_audio_path to the normalized
        # guitar stem; without separation, transcribe the normalized full mix.
        audio_path = job.get("transcription_audio_path") or job.get("normalized_audio_path")
        if not audio_path:
            raise ValueError(f"job {job_id} has no normalized_audio_path; ingest_audio must run first")

        source = "separated guitar stem" if job.get("transcription_audio_path") else "full mix"
        logger.info("transcribe: job %s running basic-pitch on %s (%s)", job_id, audio_path, source)

        raw_count, raw_note_events = extract_notes(audio_path)

        logger.info(
            "transcribe: job %s kept %d/%d notes above confidence threshold %.2f",
            job_id,
            len(raw_note_events),
            raw_count,
            CONFIDENCE_THRESHOLD,
        )

        # Same file Basic Pitch used, so tempo and notes describe the same audio.
        tempo_bpm = _estimate_tempo_bpm(audio_path)
        logger.info("transcribe: job %s estimated tempo ~%d bpm", job_id, tempo_bpm)

        update_job(job_id, raw_note_events=raw_note_events, tempo_bpm=tempo_bpm)
    except Exception as exc:
        logger.exception("transcribe failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"transcribe failed: {exc}")
        raise
    return job_id
