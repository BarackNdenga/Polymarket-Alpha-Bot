# ═══════════════════════════════════════════════════════════════════
# Polymarket Alpha Bot — Production Docker Image
# ═══════════════════════════════════════════════════════════════════
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash bot && \
    mkdir -p /home/bot/app/data /home/bot/app/logs

WORKDIR /home/bot/app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=bot:bot . .

# Create data directories
RUN mkdir -p data logs

# Switch to non-root user
USER bot

# Default environment
ENV BOT_ENV=paper
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from src.data.database import Database; Database(); print('OK')" || exit 1

# Expose metrics port (optional Prometheus exporter)
EXPOSE 8080

# Entrypoint
CMD ["python", "run.py"]
