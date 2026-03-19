FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /workspace

ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.9"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    imageio[ffmpeg]

RUN pip install --no-cache-dir fastvideo

RUN python -c "import fastvideo; print('fastvideo installed')"

COPY handler.py /workspace/handler.py

CMD ["python", "-u", "handler.py"]
