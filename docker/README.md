# PARTNR Planner Docker Setup

This repository provides a Docker image for the PARTNR Planner environment, built on Ubuntu 22.04 with NVIDIA CUDA 12.4.1, using Miniconda with Python 3.9.2. A dedicated conda environment named **habitat-llm** is created and configured with the required dependencies.

## Prerequisites

- Docker (with NVIDIA Container Toolkit installed for GPU support)
- (Optional) Docker Compose

## Building the Docker Image

From the root directory (where the Dockerfile is located), run:

```bash
docker build -t habitat-llm:latest .
```

This command builds an image tagged `habitat-llm:latest`.

## Running the Docker Container with GPU Support

To run the container with GPU support and mount host directories for outputs and data, use the following command:

```bash
docker run --gpus all -it --name habitat-llm-container \
  -v $(pwd)/../docker-outputs:/partnr-planner/outputs \
  -v $(pwd)/../data:/partnr-planner/data \
  habitat-llm:latest
```
```bash
docker run --gpus all -it --name habitat-llm-container \
  -v $(pwd)/../docker-outputs:/partnr-planner/outputs \
  -v $(pwd)/../data:/partnr-planner/data \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  habitat-llm:latest
```

- `--gpus all` enables GPU support.
- `-v $(pwd)/../docker-outputs:/partnr-planner/outputs` mounts the host directory `../docker-outputs` to `/partnr-planner/outputs` in the container.
- `-v $(pwd)/../data:/partnr-planner/data` mounts the host directory `../data` to `/partnr-planner/data` in the container.
- The container starts with an interactive Bash shell.



## Environment Details

- **Base Image:** `nvidia/cuda:12.4.1-devel-ubuntu22.04`
- **Python Version:** 3.9.2 (installed via Miniconda)
- **Conda Environment:** `habitat-llm`
- **Key Dependencies:**
  - PyTorch 2.4.1, torchvision 0.19.1, torchaudio 2.4.1, and pytorch-cuda=12.4 (installed from `pytorch` and `nvidia` channels)
  - Habitat-Sim 0.3.3 with Bullet and headless support (from `conda-forge` and `aihabitat`)
  - Additional Python packages (numpy, pytest, etc.) as specified in the installation instructions.
- **Repository Setup:**  
  The Dockerfile clones the PARTNR Planner repository, updates its submodules, and installs:
  - `third_party/habitat-lab/habitat-lab`
  - `third_party/habitat-lab/habitat-baselines`
  - `third_party/transformers-CFG`
  - The project's requirements (via `requirements.txt`) and the project itself using `pip install -e .`

Below is the new separate **Dataset Download** section:

---

## Dataset Download

After setting up the environment and installing dependencies, download and link the required datasets using the following commands:

1. **Habitat-Sim Datasets:**  
   ```bash
   python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets hab_spot_arm hab3-episodes habitat_humanoids --data-path data/ --no-replace --no-prune
   ```

2. **OVMM Objects:**  
   ```bash
   git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects data/objects_ovmm --recursive
   ```

3. **HSSD Scene Dataset:**  
   ```bash
   git clone -b partnr https://huggingface.co/datasets/hssd/hssd-hab data/versioned_data/hssd-hab
   cd data/versioned_data/hssd-hab
   git lfs pull
   cd ../../..
   ln -s versioned_data/hssd-hab data/hssd-hab
   ```

4. **Task Datasets and Skill Checkpoints:**  
   ```bash
   git clone https://huggingface.co/datasets/ai-habitat/partnr_episodes data/versioned_data/partnr_episodes
   cd data/versioned_data/partnr_episodes
   git lfs pull
   cd ../../..
   mkdir -p data/datasets
   ln -s ../versioned_data/partnr_episodes data/datasets/partnr_episodes
   ln -s versioned_data/partnr_episodes/checkpoints data/models
   ```

## Run the Tests

After setting up the environment and downloading datasets, run the test suite to verify the installation:

1. **Download and link additional testing datasets:**
   ```bash
   git clone https://huggingface.co/datasets/ai-habitat/hssd-partnr-ci data/versioned_data/hssd-partnr-ci
   ln -s versioned_data/hssd-partnr-ci data/hssd-partnr-ci
   cd data/hssd-partnr-ci
   git lfs pull
   cd ../..

   # Link RAG testing data
   ln -s versioned_data/partnr_episodes/test_rag data/test_rag
   ```

2. **Run the test suite:**
   ```bash
   python -m pytest -s habitat_llm/tests
   ```
