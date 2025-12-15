# 1. Usamos la imagen oficial de RunPod como base
# Trae: Python 3.11, PyTorch 2.4, CUDA 12.4.1, Ubuntu 22.04
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 2. Evitamos interacciones en la instalación de apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PANDA_PRC_DIR="/etc/panda3d"

# 3. Instalamos SOLO lo que le falta a la imagen de RunPod
# MetaDrive necesita libgl1 (OpenGL) y librerías gráficas básicas
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Configurar entorno de trabajo
WORKDIR /workspace

# 5. Instalamos tus librerías de Python
# NOTA: no instalamos torch aquí, porque la imagen base ya lo tiene.
RUN pip install --no-cache-dir \
    numpy==1.26.3 \
    "ray[rllib]" \
    seaborn \
    gymnasium \
    PyYAML \
    pydantic \
    GPUtil \
    git+https://github.com/metadriverse/metadrive.git@85e5dadc6c7436d324348f6e3d8f8e680c06b4db

# 6. Descarga de Assets de MetaDrive
RUN python -m metadrive.pull_asset

# 7. Copiamos tu código
COPY . /workspace

# 8. Comando de inicio
CMD ["python", "train.py"]
