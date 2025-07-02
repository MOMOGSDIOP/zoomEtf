# === builder stage ===
FROM python:3.9-slim AS builder

WORKDIR /zoom-etf

# 1. Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    g++ \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Installer les packages ML de base
RUN pip install --no-cache-dir --timeout 100 \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu


# 3. Installer les autres dépendances (sans finbert-embedding)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Installer sentence-transformers et configurer FinBERT
RUN pip install --no-cache-dir sentence-transformers==2.6.0 && \
    python -c "from transformers import AutoModel; AutoModel.from_pretrained('yiyanghkust/finbert-tone')"

# === final stage ===
FROM python:3.9-slim

WORKDIR /zoom-etf

# Dépendances runtime
RUN apt-get update && apt-get install -y \
    netcat-openbsd \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copie depuis le builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copie de l'application
COPY backend/app backend/app
COPY backend/tests backend/tests
COPY backend/alembic.ini backend/alembic.ini
COPY backend/start.sh backend/start.sh

ENV PYTHONPATH=/zoom-etf/backend
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

RUN chmod +x backend/start.sh

EXPOSE 8000

CMD ["./backend/start.sh"]