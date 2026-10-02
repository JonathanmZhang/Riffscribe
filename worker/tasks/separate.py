"""Optional guitar source separation, run between ingest_audio and transcribe
when a job has isolate_guitar set. Two separators, chosen per job by its
separation_quality: "standard" = Demucs htdemucs_6s on the CPU (below),
"high" = Mega 53 on an NVIDIA GPU (tasks/mega53.py). If Mega 53 can't run or
fails, the job falls back to Demucs and says so in separation_note.

Uses Demucs' Python API directly rather than the demucs CLI, and does file
I/O with librosa/soundfile rather than torchaudio: Demucs 4.0.1 reads and
writes through torchaudio, whose I/O layer has changed in recent releases.
"""

import functools
import logging
import os
import time

import librosa
import numpy as np
import soundfile as sf
import torch
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import celeryd_after_setup
from demucs.apply import apply_model
from demucs.pretrained import get_model

from tasks import mega53
from tasks.audio_io import job_dir, normalize_to_wav
from tasks.celery_app import app
from tasks.separation_limits import separation_duration_error
from tasks.storage import get_job, get_separation_capabilities, set_separation_capabilities, update_job

logger = logging.getLogger(__name__)

# htdemucs_6s is the only standard Demucs model with a guitar stem. Its
# weights are downloaded into the image at build time (see worker/Dockerfile);
# any other model is downloaded on first use and not persisted.
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs_6s")
DEMUCS_SHIFTS = int(os.environ.get("DEMUCS_SHIFTS", "1"))
# Separation is far slower than the rest of the pipeline on CPU, so it has
# its own, tighter duration cap (tasks/separation_limits.py, also checked
# early by ingest_audio) and a longer time limit than the 120s tasks.
SEPARATION_SOFT_TIME_LIMIT_SECONDS = int(os.environ.get("SEPARATION_SOFT_TIME_LIMIT_SECONDS", "900"))
# Longest a Mega 53 run may take before the job falls back to Demucs. Under
# the soft time limit above, so Demucs still has time to run after it.
HQ_SEPARATION_TIMEOUT_SECONDS = float(os.environ.get("HQ_SEPARATION_TIMEOUT_SECONDS", "420"))

GUITAR_SOURCE = "guitar"
SEPARATION_QUEUE = "separation"

# Intra-op threads for Demucs inference, so separation doesn't compete with
# the main worker's TensorFlow for every core. Default: half the CPUs
# available to the container (on this 8-CPU setup that equals torch's own
# default of one thread per physical core).
SEPARATION_TORCH_THREADS = int(
    os.environ.get("SEPARATION_TORCH_THREADS") or max(1, (os.cpu_count() or 2) // 2)
)


@functools.lru_cache(maxsize=None)
def _load_model(name: str):
    """Loaded once per worker process; repeat jobs reuse it."""
    model = get_model(name)
    model.eval()
    return model


def separate_guitar_stem(input_path: str) -> tuple[np.ndarray, int]:
    """Separates the guitar from input_path. Returns (stem, sample_rate) with
    stem shaped (channels, samples), clipped to [-1, 1]. Pure: no Redis or
    Celery, so scripts can call it directly.
    """
    # Set per call rather than at import: the main worker imports this module
    # too, and the setting only matters in the process that runs inference.
    torch.set_num_threads(SEPARATION_TORCH_THREADS)
    model = _load_model(DEMUCS_MODEL)
    if GUITAR_SOURCE not in model.sources:
        raise ValueError(f"Demucs model {DEMUCS_MODEL!r} has no guitar stem (sources: {model.sources})")

    # Load at the model's native rate (44.1kHz for htdemucs_6s) and keep
    # channels, then match the model's channel count the way Demucs' own
    # loader does: duplicate mono, keep the first N of a wider layout.
    audio, _ = librosa.load(input_path, sr=model.samplerate, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.shape[0] == 1:
        audio = np.repeat(audio, model.audio_channels, axis=0)
    elif audio.shape[0] > model.audio_channels:
        audio = audio[: model.audio_channels]

    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))

    # Same normalization demucs.separate applies around apply_model.
    ref = wav.mean(0)
    mean = ref.mean()
    std = ref.std()
    if std == 0:  # silent input: nothing to scale, avoid dividing by zero
        std = torch.tensor(1.0)
    wav = (wav - mean) / std

    with torch.no_grad():
        sources = apply_model(
            model, wav[None], shifts=DEMUCS_SHIFTS, split=True, overlap=0.25, progress=False
        )[0]
    sources = sources * std + mean

    stem = sources[model.sources.index(GUITAR_SOURCE)].cpu().numpy()
    return np.clip(stem, -1.0, 1.0), model.samplerate


@celeryd_after_setup.connect
def _report_capabilities(sender, instance, **kwargs):
    """At worker start: records whether this worker can run high-quality
    separation (a GPU and the Mega 53 weights), for GET /capabilities. Only
    the worker serving the separation queue reports. Never stops the start."""
    try:
        if SEPARATION_QUEUE not in instance.app.amqp.queues.consume_from:
            return
        capabilities = mega53.probe_in_subprocess()
        set_separation_capabilities(capabilities)
        logger.info(
            "separation worker %s: high-quality separation %s",
            sender,
            f"available on {capabilities['gpu']} ({capabilities['gpu_memory_mib']} MiB)"
            if capabilities["available"]
            else f"unavailable: {capabilities['reason']}",
        )
    except Exception:
        logger.exception("could not report separation capabilities")


def _separate_high_quality(job_id: str, source_path: str, stem_path: str) -> str | None:
    """Writes the Mega 53 stem to stem_path. Returns None on success, or the
    reason it wasn't used (the caller then falls back to Demucs)."""
    capabilities = get_separation_capabilities()
    if not capabilities.get("available"):
        return capabilities.get("reason") or "the separation worker has not reported a GPU"
    try:
        info = mega53.separate_to_file(source_path, stem_path, HQ_SEPARATION_TIMEOUT_SECONDS)
    except mega53.HqSeparationError as exc:
        return str(exc)
    logger.info(
        "separate_guitar: job %s Mega 53 (%s head) separated %.1fs of audio in %.1fs (real-time factor "
        "%.2f, model load %.1fs) on %s, peak GPU memory %d MiB allocated / %d MiB reserved",
        job_id,
        info["head"],
        info["audio_seconds"],
        info["seconds"],
        info["seconds"] / info["audio_seconds"] if info["audio_seconds"] else 0.0,
        info["load_seconds"],
        info["gpu"],
        info["peak_vram_allocated_mib"],
        info["peak_vram_reserved_mib"],
    )
    update_job(job_id, separation_peak_vram_mib=info["peak_vram_reserved_mib"])
    return None


@app.task(name="separate_guitar", soft_time_limit=SEPARATION_SOFT_TIME_LIMIT_SECONDS)
def separate_guitar(job_id: str) -> str:
    try:
        update_job(job_id, status="processing", stage="separating")

        job = get_job(job_id)
        source_path = job.get("source_audio_path")
        if not source_path:
            raise ValueError(f"job {job_id} has no source_audio_path; ingest_audio must run first")

        # Safety net: ingest_audio already rejects over-long audio for
        # isolate_guitar jobs before normalizing.
        duration = job.get("source_duration_seconds")
        too_long = separation_duration_error(duration) if duration is not None else None
        if too_long:
            raise ValueError(too_long)

        directory = job_dir(job_id)
        stem_path = os.path.join(directory, "guitar_stem.wav")
        # Both separators run on the original source (e.g. 44.1/48kHz
        # stereo), not normalized.wav: they are trained on 44.1kHz stereo.
        start = time.perf_counter()

        separator, note = "demucs", None
        if job.get("separation_quality") == "high":
            try:
                reason = _separate_high_quality(job_id, source_path, stem_path)
            except SoftTimeLimitExceeded:
                raise
            except Exception as exc:  # the optional separator never fails the job
                logger.exception("separate_guitar: job %s Mega 53 failed unexpectedly", job_id)
                reason = f"an unexpected error ({exc})"
            if reason is None:
                separator = "mega53"
            else:
                note = (
                    f"High-quality separation was not used: {reason}. "
                    "Standard separation (Demucs) was used instead."
                )
                logger.warning("separate_guitar: job %s falling back to Demucs: %s", job_id, reason)

        if separator == "demucs":
            load_start = time.perf_counter()
            _load_model(DEMUCS_MODEL)
            load_seconds = time.perf_counter() - load_start

            demucs_start = time.perf_counter()
            stem, sample_rate = separate_guitar_stem(source_path)
            demucs_seconds = time.perf_counter() - demucs_start
            sf.write(stem_path, stem.T, sample_rate, subtype="PCM_16")

            audio_seconds = stem.shape[1] / sample_rate
            logger.info(
                "separate_guitar: job %s separated %.1fs of audio in %.1fs (real-time factor %.2f, "
                "model %s, shifts %d, torch threads %d, model load %.1fs)",
                job_id,
                audio_seconds,
                demucs_seconds,
                demucs_seconds / audio_seconds if audio_seconds else 0.0,
                DEMUCS_MODEL,
                DEMUCS_SHIFTS,
                torch.get_num_threads(),
                load_seconds,
            )

        # Everything this step spent separating: for "high", Mega 53's whole
        # process (model load included), plus Demucs after a fallback.
        separation_seconds = time.perf_counter() - start
        # Same conversion ingest applies, so Basic Pitch gets its usual format.
        transcription_path = normalize_to_wav(stem_path, os.path.join(directory, "guitar_stem_normalized.wav"))

        update_job(
            job_id,
            stem_audio_path=stem_path,
            transcription_audio_path=transcription_path,
            separation_seconds=round(separation_seconds, 2),
            separator=separator,
            separation_note=note,
        )
    except Exception as exc:
        logger.exception("separate_guitar failed for job %s", job_id)
        update_job(job_id, status="failed", error=f"separate_guitar failed: {exc}")
        raise
    return job_id
