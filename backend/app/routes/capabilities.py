import json

from fastapi import APIRouter

from app.routes.jobs import redis_client
from app.schemas.capabilities import CapabilitiesResponse, SeparationCapabilities

router = APIRouter(tags=["capabilities"])

# Written by the separation worker when it starts (worker/tasks/storage.py).
SEPARATION_CAPABILITIES_KEY = "separation:capabilities"
NOT_REPORTED = "the separation worker has not started yet"


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities() -> CapabilitiesResponse:
    """What this installation can do beyond the defaults; the upload form
    uses it to enable or disable high-quality guitar separation."""
    raw = await redis_client.get(SEPARATION_CAPABILITIES_KEY)
    reported = json.loads(raw) if raw else {}
    available = bool(reported.get("available"))
    return CapabilitiesResponse(
        separation=SeparationCapabilities(
            high_quality_available=available,
            high_quality_unavailable_reason=None if available else reported.get("reason") or NOT_REPORTED,
            gpu=reported.get("gpu"),
            gpu_memory_mib=reported.get("gpu_memory_mib"),
        )
    )
