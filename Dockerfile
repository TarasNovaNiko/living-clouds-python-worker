FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Python + basic deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Install CUDA-enabled PyTorch first (important)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker
COPY worker.py .
COPY model_stub.py .

# Optional: cache folders (helps speed on cold starts if volume supported)
ENV HF_HOME=/tmp/hf
ENV TRANSFORMERS_CACHE=/tmp/hf
ENV TORCH_HOME=/tmp/torch

CMD ["python3", "worker.py"]
