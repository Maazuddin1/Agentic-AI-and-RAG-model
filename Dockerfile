FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/indexes data/memory data/papers

# Expose the port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV API_URL=http://localhost:8000

# Run the application (FastAPI for API and Streamlit for frontend)
# CMD ["sh", "-c", "python -m src.api.fastapi_app & streamlit run app.py"]
CMD ["python", "main.py"]
