ARG PYTORCH="1.7.0"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV DEBIAN_FRONTEND noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN rm -f /etc/apt/sources.list.d/cuda.list && rm -f /etc/apt/sources.list.d/nvidia-ml.list 
RUN apt-get update && apt-get install -y git vim tmux libgl1-mesa-glx libglib2.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MIM
RUN pip install openmim --no-cache-dir

# Install MMCV
RUN mim install mmcv-full==1.2.7

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

RUN pip install tensorboard scikit-learn opencv-python --no-cache-dir
RUN pip install mmpycocotools mmlvis matplotlib --no-cache-dir
RUN apt-get autoclean && apt-get autoremove