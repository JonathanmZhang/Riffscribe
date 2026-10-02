from typing import Optional

from pydantic import BaseModel


class SeparationCapabilities(BaseModel):
    """What the separation worker reported when it started. Standard
    separation (Demucs) is always available; high quality (Mega 53) needs an
    NVIDIA GPU and the GPU build of the worker."""

    high_quality_available: bool
    # Why not, readable by a user; None when available.
    high_quality_unavailable_reason: Optional[str] = None
    gpu: Optional[str] = None
    gpu_memory_mib: Optional[int] = None


class CapabilitiesResponse(BaseModel):
    separation: SeparationCapabilities
