FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    LOCAL_LLM_MODEL=/opt/models/flan-t5-small \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && pip install -r requirements.txt

RUN TRANSFORMERS_OFFLINE=0 python - <<'PY'
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
src = "google/flan-t5-small"
dst = "/opt/models/flan-t5-small"
AutoTokenizer.from_pretrained(src).save_pretrained(dst)
AutoModelForSeq2SeqLM.from_pretrained(src).save_pretrained(dst)
PY

COPY app ./app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
