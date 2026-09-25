import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import jobs

# Explicit allowlist of frontend origins, comma-separated. A wildcard is
# refused rather than silently opening the API to every origin.
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
if "*" in CORS_ALLOWED_ORIGINS:
    raise ValueError("CORS_ALLOWED_ORIGINS must list explicit origins; '*' is not allowed")

app = FastAPI(title="Riffscribe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
