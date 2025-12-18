FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Cache to /tmp to avoid "No space left on device"
ENV HF_HOME=/tmp/hf \
    XDG_CACHE_HOME=/tmp \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    DIFFUSERS_CACHE=/tmp/hf/diffusers \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /tmp/hf/transformers /tmp/hf/diffusers

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY worker.py /app/worker.py

CMD ["python", "-u", "worker.py"]
