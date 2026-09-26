"""Guitar-separation duration cap, shared by ingest_audio (early check, so
over-long audio is rejected before normalizing) and separate_guitar (safety
net). Kept free of torch/Demucs imports so ingest doesn't depend on them.
"""

import os

MAX_SEPARATION_DURATION_SECONDS = float(os.environ.get("MAX_SEPARATION_DURATION_SECONDS", "120"))


def separation_duration_error(duration_seconds: float) -> str | None:
    """User-readable reason audio is too long to separate, or None if it's fine."""
    if duration_seconds <= MAX_SEPARATION_DURATION_SECONDS:
        return None
    return (
        f"guitar isolation is limited to {MAX_SEPARATION_DURATION_SECONDS:.0f} seconds of "
        f"audio, and this file is {duration_seconds:.0f} seconds long. Use a shorter clip, or "
        "turn off \"Isolate guitar\"."
    )
