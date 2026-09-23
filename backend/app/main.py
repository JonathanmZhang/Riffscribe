from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import jobs

app = FastAPI(title="StratoTab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
