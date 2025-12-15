FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

CMD ["python", "-u", "worker.py"]
