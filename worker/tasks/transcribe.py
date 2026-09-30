import bisect
import logging
import os

import numpy as np
import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
from basic_pitch.inference import run_inference
from basic_pitch.note_creation import model_output_to_notes

from tasks.beats import estimate_rhythm
from tasks.celery_app import app
from tasks.chords import recognize_chords
from tasks.fretboard import CHORD_ONSET_TOLERANCE_SECONDS
from tasks.storage import get_job, update_job

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = float(os.environ.get("BASIC_PITCH_CONFIDENCE_THRESHOLD", "0.5"))
# Lower confidence bar for notes that start with a confident note (chord
# tones); see select_notes. Values >= CONFIDENCE_THRESHOLD disable it.
CHORD_TONE_CONFIDENCE_FLOOR = float(os.environ.get("CHORD_TONE_CONFIDENCE_FLOOR", "0.45"))
# Basic Pitch's own note-creation settings. predict()'s defaults are 0.5 / 0.3 /
# 127.7ms; 0.4 frame and 80ms were chosen on the EGSet12 real-guitar benchmark
# (scripts/egset12_benchmark.py): the 128ms minimum is longer than a 16th note
# at ~120bpm and was dropping fast notes. Lower onset thresholds cost precision.
BASIC_PITCH_ONSET_THRESHOLD = float(os.environ.get("BASIC_PITCH_ONSET_THRESHOLD", "0.5"))
BASIC_PITCH_FRAME_THRESHOLD = float(os.environ.get("BASIC_PITCH_FRAME_THRESHOLD", "0.4"))
BASIC_PITCH_MIN_NOTE_LENGTH_MS = float(os.environ.get("BASIC_PITCH_MIN_NOTE_LENGTH_MS", "80"))


def notes_from_model_output(
    model_output: dict,
    onset_threshold: float = BASIC_PITCH_ONSET_THRESHOLD,
    frame_threshold: float = BASIC_PITCH_FRAME_THRESHOLD,
    min_note_length_ms: float = BASIC_PITCH_MIN_NOTE_LENGTH_MS,
) -> list[dict]:
    """Basic Pitch's note creation step (what predict() does after running
    the network), with its three settings exposed. Pure and cheap, so the
    measurement scripts can sweep settings on a cached model output. Each
    event has pitch, midi, start_time, end_time and amplitude.
    """
    min_note_frames = int(np.round(min_note_length_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    _, note_events = model_output_to_notes(
        model_output,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_frames,
        # Same as predict()'s defaults for the settings not exposed here.
        min_freq=None,
        max_freq=None,
        multiple_pitch_bends=False,
        melodia_trick=True,
        midi_tempo=120,
    )
    return [
        {
            "pitch": pretty_midi.note_number_to_name(pitch_midi),
            "midi": int(pitch_midi),
            "start_time": start_time,
            "end_time": end_time,
            "amplitude": float(amplitude),
        }
        for start_time, end_time, pitch_midi, amplitude, _pitch_bends in note_events
    ]


def detect_note_events(audio_path: str) -> tuple[list[dict], dict]:
    """Runs Basic Pitch on audio_path and returns (every note event, before
    any confidence filter, and the raw model output). Pure: used by
    extract_notes and by the chord inspection/evaluation scripts, which need
    the events the filter drops and the model's per-frame activations.
    """
    model_output = run_inference(audio_path, ICASSP_2022_MODEL_PATH)
    return notes_from_model_output(model_output), model_output


def select_notes(
    events: list[dict],
    threshold: float = CONFIDENCE_THRESHOLD,
    chord_floor: float = CHORD_TONE_CONFIDENCE_FLOOR,
) -> list[dict]:
    """Post-detection note selection: returns copies of events, each with a
    "kept" flag. Shared by extract_notes (the pipeline) and the measurement
    scripts, so both apply exactly the same rules.

    A note at or above `threshold` is kept. A weaker note is also kept when
    it's at least `chord_floor` AND a note at or above `threshold` starts
    within the chord onset window (CHORD_ONSET_TOLERANCE_SECONDS) of it:
    Basic Pitch tends to give the other tones of a strummed chord 0.35-0.5,
    while they arrive together with at least one confident note. Isolated
    weak notes stay filtered. chord_floor >= threshold disables the rule.
    """
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

        # Same file Basic Pitch used, so beats and notes describe the same
        # audio. Falls back to librosa's tempo without beats on failure.
        rhythm = estimate_rhythm(audio_path)
        logger.info("transcribe: job %s estimated tempo ~%d bpm, %d beat(s), %d downbeat(s)", job_id,
                    rhythm["tempo_bpm"], len(rhythm["beats"]), len(rhythm["downbeats"]))

        # Chord names for the tab, from the same audio. They're display-only,
        # so a failure here is logged and the job continues without them
        # rather than losing the transcription.
        try:
            chord_segments = recognize_chords(audio_path)
            logger.info("transcribe: job %s recognized %d chord segment(s)", job_id, len(chord_segments))
        except Exception:
            logger.exception("transcribe: chord recognition failed for job %s; continuing without chord names",
                             job_id)
            chord_segments = []

        update_job(job_id, raw_note_events=raw_note_events, tempo_bpm=rhythm["tempo_bpm"], beats=rhythm["beats"],
                   downbeats=rhythm["downbeats"], chord_segments=chord_segments)
    except Exception as exc:
        logger.exception("transcribe failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"transcribe failed: {exc}")
        raise
    return job_id
