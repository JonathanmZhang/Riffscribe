import os

from celery import Celery

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/1")

app = Celery(
    "riffscribe",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["tasks.pipeline", "tasks.separate", "tasks.transcribe", "tasks.fretboard"],
)

app.conf.task_soft_time_limit = 120

# Separation (~3GB RAM each) runs on its own queue, consumed by the
# worker-separation service at concurrency 1; everything else stays on the
# default "celery" queue. The backend also sets this queue explicitly on the
# signature it sends.
app.conf.task_routes = {"separate_guitar": {"queue": "separation"}}
