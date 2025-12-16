# 1. Usamos la imagen oficial de RunPod como base
# Trae: Python 3.11, PyTorch 2.4, CUDA 12.4.1, Ubuntu 22.04
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 2. Evitamos interacciones en la instalación de apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PANDA_PRC_DIR="/etc/panda3d"

# 3. Instalamos SOLO lo que le falta a la imagen de RunPod
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Configurar entorno de trabajo
WORKDIR /app

# 5. Instalamos tus librerías de Python
# 5. Instalamos tus librerías de Python (versiones fijadas según conda)
RUN pip install --no-cache-dir \
    numpy==2.2.6 \
    "ray[rllib]==2.51.1" \
    tensorboard \
    seaborn==0.13.2 \
    gymnasium==1.1.1 \
    PyYAML==6.0.3 \
    pydantic \
    GPUtil \
    git+https://github.com/metadriverse/metadrive.git@85e5dadc6c7436d324348f6e3d8f8e680c06b4db


# 5.5 Parchear el bug del logger de MetaDrive (Known Pipes)
RUN printf '%s\n' \
"import pathlib" \
"import metadrive.engine.core.engine_core as ec" \
"" \
"p = pathlib.Path(ec.__file__)" \
"t = p.read_text()" \
"old = 'logger.info(\"Known Pipes: {}\".format(*GraphicsPipeSelection.getGlobalPtr().getPipeTypes()))'" \
"new = 'logger.info(\"Known Pipes: {}\".format(GraphicsPipeSelection.getGlobalPtr().getPipeTypes()))'" \
"print('Patching', p)" \
"p.write_text(t.replace(old, new))" \
> /tmp/patch_metadrive.py \
 && python /tmp/patch_metadrive.py \
 && rm /tmp/patch_metadrive.py

# 6. Descarga de Assets de MetaDrive
RUN python -m metadrive.pull_asset

# 7. Copiamos código
COPY . /app

# 8. Comando de inicio
CMD ["python", "train.py"]
