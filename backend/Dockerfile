# Étape 1 : Build
FROM python:3.9-slim AS builder

WORKDIR /zoom-etf

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --user -r requirements.txt

# Étape 2 : Image finale
FROM python:3.9-slim

WORKDIR /zoom-etf

COPY --from=builder /root/.local /root/.local
COPY backend/app backend/app
COPY backend/tests backend/tests

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/zoom-etf/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
