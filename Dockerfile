FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face Spaces default UID is 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install PyTorch CPU and Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy project files into container with appropriate permissions
COPY --chown=user:user . /app

# Ensure runtime directories exist with full write access
RUN mkdir -p /app/static/uploads && chown -R user:user /app

# Switch to non-root user
USER user

# Expose the designated Hugging Face Spaces port
EXPOSE 7860

# Launch application via Gunicorn production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "1", "--timeout", "300", "app:app"]