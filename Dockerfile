FROM python:3.11-slim

# System dependencies required by OpenCV and other libs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install CPU-only PyTorch to keep image size manageable, then rest of deps
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Pre-download YOLO models at build time so container starts faster
RUN python -c "from ultralytics import YOLO; YOLO('yolov8x-pose.pt'); YOLO('yolov8x-seg.pt'); print('Models downloaded.')"

# Expose port (Railway injects $PORT at runtime)
EXPOSE 8000

# Run with the PORT env var Railway provides
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
