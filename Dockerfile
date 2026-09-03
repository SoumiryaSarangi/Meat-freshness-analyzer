FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first to leverage Docker cache
COPY backend/requirements.txt .

# Install dependencies (CPU-only PyTorch to save space, since most cheap hosting doesn't have GPUs)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
# We copy backend, frontend, and models
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY models/ ./models/

# Expose the port FastAPI will run on
EXPOSE 8000

# Set the working directory to backend where app.py lives
WORKDIR /app/backend

# Command to run the application (Uses the PORT environment variable provided by hosting platforms, defaults to 8000)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
