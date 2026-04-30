# HuggingFace Spaces Docker template — based on the root Dockerfile but
# binds to port 7860 (HF default) and runs as the `user` HF expects.

FROM python:3.11-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

RUN mkdir -p data/bronze data/silver data/gold models \
    && chown -R user:user /code

USER user

ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/code \
    PYTHONUNBUFFERED=1 \
    PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
