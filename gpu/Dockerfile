FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY proto/ proto/
COPY gpu/models/ models/
COPY gpu/training/ training/
COPY gpu/inference/ inference/

CMD ["python3", "inference/server.py"]
