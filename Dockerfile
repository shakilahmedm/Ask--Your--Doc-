# Dockerfile for RAG API with FastAPI and Streamlit

# Use slim Python image
FROM python:3.10-slim


# Set working directory
WORKDIR /app


# Install system dependencies for OCR and PDF extraction
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY . .


# Expose FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501


# Start both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_frontend.py --server.port 8501"]
