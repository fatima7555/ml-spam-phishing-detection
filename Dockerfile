FROM python:3.11-slim

WORKDIR /app

# System deps needed to compile scikit-learn / xgboost extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK corpora into the image
RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True)"

# Copy source code
COPY . .

# Download datasets and train both classifiers at build time.
# Trained model .pkl files are baked into the image layer so the
# container starts serving predictions immediately with no cold-start.
RUN python data/download_data.py && \
    python train_email.py && \
    python train_url.py

EXPOSE 5000

# gunicorn: 2 workers, 120s timeout (generous for large requests)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--chdir", "/app", \
     "webapp.app:app"]
