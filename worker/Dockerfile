FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY tasks/ ./tasks/
CMD ["celery", "-A", "tasks.celery_app", "worker", "--loglevel=info"]