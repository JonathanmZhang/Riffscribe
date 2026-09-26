"""Audio file helpers shared by the pipeline tasks and standalone scripts.

Deliberately free of Celery and Redis imports so any module (tasks,
scripts/) can use it without import cycles or a running broker.
"""

import os
import subprocess

import librosa
import soundfile as sf

DATA_DIR = "/app/data"
# Format every transcription input is normalized to (mono, 22.05kHz PCM_16).
TARGET_SAMPLE_RATE = 22050


def job_dir(job_id: str) -> str:
    path = os.path.join(DATA_DIR, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def probe_duration_seconds(input_path: str) -> float:
    """Reads the duration from the container header via ffprobe, without
    decoding the audio, so over-long inputs are rejected before the
    expensive librosa load/resample. Works for every format ingest accepts
    (mp3/wav/m4a uploads and whatever yt-dlp downloads, e.g. webm), unlike
    librosa.get_duration(path=), which depends on soundfile's format support.
    """
    output = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
        ],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.strip()
    return float(output)


def normalize_to_wav(input_path: str, output_path: str) -> str:
    """Decodes any supported input and writes it as mono 22.05kHz PCM_16 WAV,
    the format Basic Pitch is fed."""
    audio, _ = librosa.load(input_path, sr=TARGET_SAMPLE_RATE, mono=True)
    sf.write(output_path, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return output_path
