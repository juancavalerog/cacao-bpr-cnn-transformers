# Entorno reproducible del experimento BPR-cacao (PyTorch + timm)
# Driver del servidor: NVIDIA 560.35.03 (CUDA 12.6 como máximo), por eso torch se instala con ruedas cu126.
# Las ruedas de PyTorch traen su propio CUDA y cuDNN: la imagen base no necesita CUDA.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 procps tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --retries 10 --timeout 120 \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        -r /tmp/requirements-docker.txt

WORKDIR /workspace/cacao
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.root_dir=/workspace"]
