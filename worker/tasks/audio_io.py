"""Audio file helpers shared by the pipeline tasks and standalone scripts.

Deliberately free of Celery and Redis imports so any module (tasks,
scripts/) can use it without import cycles or a running broker.
"""

import json
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
    """Reads the audio duration from the file header via ffprobe, without
    decoding, so over-long inputs are rejected before any expensive work.
    Works for every format ingest accepts, including video containers
    (mp4/mov/webm), unlike librosa.get_duration(path=), which depends on
    soundfile's format support.

    Uses the first audio stream's duration (a video track can be longer than
    the audio), falling back to the container duration when the stream
    doesn't report one (Matroska/webm doesn't). Raises ValueError if the file
    has no audio track at all.
    """
    output = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration:format=duration",
            "-of", "json",
            input_path,
        ],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout
    info = json.loads(output)
    if not info.get("streams"):
        raise ValueError("this file has no audio track")
    stream_duration = info["streams"][0].get("duration")
    return float(stream_duration if stream_duration not in (None, "N/A") else info["format"]["duration"])


def ensure_decodable_audio(input_path: str, output_dir: str) -> str:
    """Returns a path soundfile can read directly: input_path itself when
    it's already natively readable (wav/flac/ogg/mp3), otherwise the first
    audio track extracted with ffmpeg to <output_dir>/source_audio.wav at its
    original sample rate and channel count (any video track is ignored).

    librosa can load m4a/mp4/mov/webm itself, but only through its audioread
    fallback, which is deprecated since librosa 0.10 and removed in 1.0 -
    and which just shells out to ffmpeg anyway. Doing that step explicitly
    keeps ingest and separation off the deprecated path.
    """
    try:
        sf.info(input_path)
        return input_path
    except sf.LibsndfileError:
        pass

    output_path = os.path.join(output_dir, "source_audio.wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", input_path,
                "-map", "0:a:0", "-vn",
                "-c:a", "pcm_s16le",
                output_path,
            ],
            capture_output=True, text=True, check=True, timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        raise ValueError(f"could not extract audio from this file ({detail[-1] if detail else 'ffmpeg failed'})") from exc
    return output_path


def normalize_to_wav(input_path: str, output_path: str) -> str:
    """Decodes any supported input and writes it as mono 22.05kHz PCM_16 WAV,
    the format Basic Pitch is fed."""
    audio, _ = librosa.load(input_path, sr=TARGET_SAMPLE_RATE, mono=True)
    sf.write(output_path, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return output_path
