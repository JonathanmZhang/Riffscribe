FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pure, stdlib-only notation code shared with the worker (bars, quantization,
# MusicXML export), from the "worker" build context in docker-compose.yml.
# Rebuild the backend too when these files change.
COPY --from=worker tasks/rhythm.py tasks/musicxml.py ./tasks/
COPY app/ ./app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
