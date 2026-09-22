from fastapi import FastAPI

from app.routes import jobs

app = FastAPI(title="StratoTab API")

app.include_router(jobs.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
