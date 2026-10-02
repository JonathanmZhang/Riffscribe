import json
import os

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def get_job(job_id: str) -> dict:
    raw = redis_client.get(_job_key(job_id))
    return json.loads(raw) if raw else {}


def update_job(job_id: str, **fields) -> None:
    job = get_job(job_id)
    job["job_id"] = job_id
    job.update(fields)
    redis_client.set(_job_key(job_id), json.dumps(job))


# What the separation worker can do (written once at its start by
# tasks/separate.py, read by the backend's GET /capabilities).
SEPARATION_CAPABILITIES_KEY = "separation:capabilities"


def get_separation_capabilities() -> dict:
    raw = redis_client.get(SEPARATION_CAPABILITIES_KEY)
    return json.loads(raw) if raw else {}


def set_separation_capabilities(capabilities: dict) -> None:
    redis_client.set(SEPARATION_CAPABILITIES_KEY, json.dumps(capabilities))
