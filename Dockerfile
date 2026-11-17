FROM python:3.11.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        'torch==2.13.0+cpu' \
    && python -m pip install -r requirements.txt \
    && python -m pip check

COPY scripts/ scripts/
COPY tests/ tests/
COPY data/ data/
COPY MANIFEST.sha256 ./
RUN sha256sum -c MANIFEST.sha256 \
    && install -d -o 10001 -g 10001 /app/outputs

USER 10001:10001

CMD ["python", "scripts/run_experiments.py"]
