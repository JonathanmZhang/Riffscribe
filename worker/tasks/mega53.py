"""High-quality guitar separation: MVSep Mega 53 Stems (ZFTurbo, BS-RoFormer),
the optional "high" separation quality. Needs an NVIDIA GPU: on a CPU it is
20-38x real time (experiments/full_mix/README.md, branch
experiment/full-mix-benchmark).

Only in images built with HQ_SEPARATION=1 (docker-compose.gpu.yml): those
hold MSST at /opt/msst and a two-head checkpoint under MEGA53_DIR.

The model runs in a subprocess of the separation worker, not in the Celery
process: MSST's top-level `utils` package would shadow BTC's, the GPU
memory is released when the process exits, and a crash or an out-of-memory
kill is an error the task can catch (it then falls back to Demucs).

    python -m tasks.mega53 probe                      # is it usable here? (JSON)
    python -m tasks.mega53 separate <input> <out.wav> # writes the stem, prints JSON
    python mega53.py strip <full.ckpt> <config.yaml> <out dir>   # image build only

This module imports only the stdlib at the top, so the worker can import it
without loading torch, and the Dockerfile can run it before tasks/ is copied.
"""

import json
import os
import shutil
import subprocess
import sys
import time

MSST_DIR = "/opt/msst"
MEGA53_DIR = os.environ.get("MEGA53_DIR", "/app/models/mega53")
CHECKPOINT = os.path.join(MEGA53_DIR, "guitar_heads.ckpt")
CONFIG = os.path.join(MEGA53_DIR, "config.yaml")

# The two guitar stems of the 53. Only their mask estimators are kept: the
# shared transformer runs in full, so each stem is what the 53-stem model
# gives. "guitar" is the one the benchmark's headline numbers use.
HEADS = ["guitar", "electric-guitar"]
HEAD = os.environ.get("HQ_SEPARATION_HEAD", "guitar")
SAMPLE_RATE = 44100

NOT_BUILT = (
    "the separation worker was built without high-quality separation. Start the stack with "
    "docker-compose.gpu.yml on a computer with an NVIDIA GPU (see the README)"
)
NO_GPU = "no NVIDIA GPU is available to the separation worker"


class HqSeparationError(Exception):
    """High-quality separation didn't produce a stem; the message is the
    reason, readable by a user."""


# --- in the worker process -------------------------------------------------


def _run(args: list[str], timeout: float | None) -> dict:
    """Runs this module's CLI in a subprocess and returns the JSON object it
    printed last. subprocess.run kills the child on a timeout and on any
    exception raised here (e.g. Celery's soft time limit)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "tasks.mega53", *args],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise HqSeparationError(f"it took longer than {timeout:.0f} seconds") from None
    result = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            try:
                result = json.loads(line)
            except ValueError:
                continue
            break
    if result is not None and result.get("error"):
        raise HqSeparationError(result["error"])
    if proc.returncode < 0:
        # Killed by a signal; 9 is usually the kernel's out-of-memory killer.
        raise HqSeparationError(
            f"the separator was killed (signal {-proc.returncode}; the computer may have run out of memory)"
        )
    if proc.returncode != 0 or result is None:
        tail = (proc.stderr.strip().splitlines() or ["no output"])[-1].strip()
        raise HqSeparationError(f"the separator stopped unexpectedly (exit code {proc.returncode}: {tail})")
    return result


def probe_in_subprocess(timeout: float = 120) -> dict:
    """{"available", "reason", "gpu", "gpu_memory_mib"} for this worker.
    Never raises. A subprocess, so CUDA is never initialized in the Celery
    parent process (its forked children couldn't use it afterwards)."""
    try:
        return _run(["probe"], timeout)
    except Exception as exc:
        return {"available": False, "reason": f"the GPU check failed ({exc})", "gpu": None, "gpu_memory_mib": None}


def separate_to_file(input_path: str, output_path: str, timeout: float) -> dict:
    """Writes the guitar stem of input_path to output_path (44.1kHz stereo
    PCM_16). Returns timing and GPU memory figures. Raises HqSeparationError."""
    return _run(["separate", input_path, output_path], timeout)


# --- in the subprocess -----------------------------------------------------


def probe() -> dict:
    if not os.path.isfile(CHECKPOINT):
        return {"available": False, "reason": NOT_BUILT, "gpu": None, "gpu_memory_mib": None}
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "reason": NO_GPU, "gpu": None, "gpu_memory_mib": None}
    return {
        "available": True,
        "reason": None,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
    }


def _build_model(config):
    """The two-head BSRoformer for config, on the CPU, weights not loaded."""
    import torch
    from packaging import version

    if MSST_DIR not in sys.path:
        sys.path.insert(0, MSST_DIR)
    import models.bs_roformer.attend as attend
    from models.bs_roformer import BSRoformer

    # On CUDA, MSST's attention calls sdpa_kernel(..., set_priority=True),
    # which needs torch >= 2.6. On older torch use its legacy path (math /
    # memory-efficient kernels): the same attention, another kernel.
    if version.parse(torch.__version__.split("+")[0]) < version.parse("2.6"):
        attend._HAS_SDPA_KERNEL = False

    config.model.num_stems = len(HEADS)
    config.training.instruments = HEADS
    config.training.target_instrument = None
    # fp32, 20s chunks, 2 overlaps: the release's inference settings, and
    # the ones measured (~2.8 GiB of GPU memory).
    config.training.use_amp = False
    return BSRoformer(**dict(config.model)).eval()


def _read_config(path: str):
    import yaml
    from ml_collections import ConfigDict

    with open(path) as f:
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


def strip(full_checkpoint: str, config_path: str, out_dir: str) -> None:
    """Image build: cuts the release's 53-stem checkpoint (1.37 GB) down to
    the shared transformer and the two guitar heads, so a job loads ~200 MB
    and not all 53 heads. The kept tensors are unchanged."""
    import torch

    config = _read_config(config_path)
    ids = [list(config.training.instruments).index(h) for h in HEADS]
    ckpt = torch.load(full_checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    prefix = "mask_estimators."
    kept = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            kept[key] = value
            continue
        index, rest = key[len(prefix):].split(".", 1)
        if int(index) in ids:
            kept[f"{prefix}{ids.index(int(index))}.{rest}"] = value
    # Strict: fails the build if anything but the dropped heads is missing.
    _build_model(config).load_state_dict(kept)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": kept, "heads": HEADS}, os.path.join(out_dir, os.path.basename(CHECKPOINT)))
    shutil.copyfile(config_path, os.path.join(out_dir, os.path.basename(CONFIG)))
    print(f"mega53: kept {len(kept)} of {len(state)} tensors (heads {dict(zip(HEADS, ids))})")


def separate(input_path: str, output_path: str) -> dict:
    status = probe()
    if not status["available"]:
        return {"error": status["reason"]}
    if HEAD not in HEADS:
        return {"error": f"HQ_SEPARATION_HEAD must be one of {HEADS}, not {HEAD!r}"}

    import librosa
    import numpy as np
    import soundfile as sf
    import torch

    device = torch.device("cuda")
    try:
        load_start = time.perf_counter()
        config = _read_config(CONFIG)
        model = _build_model(config)
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        del ckpt
        model = model.to(device)
        load_seconds = time.perf_counter() - load_start

        from utils.model_utils import demix  # MSST's own chunked inference

        # 44.1kHz stereo, as the model was trained: duplicate mono, keep the
        # first two channels of a wider layout (as for Demucs).
        audio, _ = librosa.load(input_path, sr=SAMPLE_RATE, mono=False)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        audio = audio[:2]

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        stems = demix(config, model, audio, device, "bs_roformer")
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
    except RuntimeError as exc:  # torch.OutOfMemoryError is a RuntimeError
        if "out of memory" in str(exc).lower():
            return {"error": "the GPU ran out of memory"}
        raise

    sf.write(output_path, np.clip(stems[HEAD], -1.0, 1.0).T, SAMPLE_RATE, subtype="PCM_16")
    return {
        "seconds": round(seconds, 2),
        "load_seconds": round(load_seconds, 2),
        "audio_seconds": round(audio.shape[1] / SAMPLE_RATE, 2),
        "head": HEAD,
        "gpu": status["gpu"],
        # allocated = tensors; reserved = what PyTorch's allocator held (what
        # has to fit on the card, on top of the CUDA context).
        "peak_vram_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_vram_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
    }


if __name__ == "__main__":
    command, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else (None, [])
    if command == "probe" and not args:
        print(json.dumps(probe()))
    elif command == "separate" and len(args) == 2:
        outcome = separate(*args)
        print(json.dumps(outcome), flush=True)
        sys.exit(1 if outcome.get("error") else 0)
    elif command == "strip" and len(args) == 3:
        strip(*args)
    else:
        sys.exit(__doc__)
