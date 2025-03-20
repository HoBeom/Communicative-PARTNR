# Dockerfile for facebookresearch/partnr-planner using the habitat-llm conda environment
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and tools
RUN apt-get update && apt-get install -y \
    build-essential git git-lfs curl vim ca-certificates \
    libjpeg-dev libglm-dev libegl1-mesa-dev \
    ninja-build xorg-dev freeglut3-dev pkg-config \
    wget zip lcov libhdf5-dev libomp-dev unzip \
    texlive-base texlive-latex-extra \
    texlive-fonts-extra texlive-fonts-recommended \
 && git lfs install \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda (for Python package management)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p /opt/conda \
 && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create conda environment named "habitat-llm" with Python 3.9.2 and cmake 3.14.0
RUN conda create -y -n habitat-llm python=3.9.2 cmake=3.14.0

# Install conda dependencies in the 'habitat-llm' env
# withbullet ?
RUN conda run -n habitat-llm conda install -y -c pytorch -c nvidia \
      pytorch=2.4.1 torchvision=0.19.1 torchaudio=2.4.1 pytorch-cuda=12.4 \
 && conda run -n habitat-llm conda install -y -c conda-forge -c aihabitat \
      habitat-sim=0.3.3 headless \ 
 && conda run -n habitat-llm conda install -y -c conda-forge \
      numpy=1.26.4 pytest pytest-cov hypothesis pytest-mock cython mock \
 && conda run -n habitat-llm pip install opencv-python \
 && conda clean -afy

# Clone the partnr-planner repository and initialize submodules
WORKDIR /partnr-planner
RUN git clone https://github.com/facebookresearch/partnr-planner.git . \
 && git submodule update --init --recursive

# Install project dependencies
RUN conda run -n habitat-llm pip install -e third_party/habitat-lab/habitat-lab \
 && conda run -n habitat-llm pip install -e third_party/habitat-lab/habitat-baselines \
 && conda run -n habitat-llm pip install -e third_party/transformers-CFG \
 && conda run -n habitat-llm pip install -r requirements.txt \
 && conda run -n habitat-llm pip install -e . \
 && rm -rf /root/.cache/pip

# Set environment PATH to use the habitat-llm conda environment by default
ENV PATH="/opt/conda/envs/habitat-llm/bin:/opt/conda/bin:$PATH"

# Default entry point
CMD ["/bin/bash"]
