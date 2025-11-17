# GPU-Computation-of-Persistent-Homology-for-Image-Data
CUDA-accelerated persistent homology for 2D/3D image data using cubical complexes.

## Google Colab


A **TopoGPU-demo** notebook is provided to make it easy to try TopoGPU without installing it locally.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seravee08/GPU-Computation-of-Persistent-Homology-for-Image-Data/blob/main/TopoGPU_demo.ipynb)

The notebook provides a step-by-step workflow:

1. **Environment setup**
   - Installs the precompiled `topogpu` wheel for Python 3.12 on Linux.
   - Installs `cripser` (which provides the `tcripser` interface).

2. **Data loading**
   - Downloads a sample 3D grayscale volume (`.raw`) from this repository.

3. **Running TopoGPU**
   - Demonstrates `topogpu.create3D(...)` and explains its key parameters.
   - Runs persistent homology on the sample volume and retrieves the resulting pairs.

4. **Running Cubical Ripser (tcripser)**
   - Computes persistence with `tcripser.computePH(...)` on the same volume.
   - Extracts the persistence pairs from the `tcripser` output.

5. **Result comparison**
   - Compares TopoGPU and Cubical Ripser outputs (up to 3 decimal places).
   - Reports any nontrivial persistence pairs that differ between the two methods.

## Docker Image

### Prerequisites
**Windows 10/11**
- Docker Desktop with WSL 2 backend enabled.
- NVIDIA GPU + up-to-date Windows NVIDIA driver.
- User in local docker-users group.
- Verify using command in Windows PowerShell: ```docker run --rm --gpus=all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi```

**Linux**
- NVIDIA GPU + driver installed on the host.
- NVIDIA Container Toolkit installed and configured for Docker.
- Verify using command: ```docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi```

**macOS**
- No NVIDIA GPU pass-through on macOS. CUDA execution will **NOT** work on Mac.

### Run Docker
**Windows 10/11**
- Pull image: ```docker pull seravee08/topogpu:latest```
- Create container and enter shell (Replace PATH_TO_DATA with an absolute Windows path):
```command
docker run --name topogpu -it --gpus=all `
  --mount type=bind,source="PATH_TO_DATA",target=/data `
  -w /opt/app --entrypoint /bin/bash `
  seravee08/topogpu:latest
```
- Example commands inside the container:
```
./build/topoGPU -help
./build/topoGPU -filename /data/mrt_angio_416x512x112_uint16.raw -datatype ushort -height 512 -width 416 -depth 112 -bufSize 3000 -out /result.txt
```

**Linux**
- Pull image: ```docker pull seravee08/topogpu:latest```
- Create container and enter shell (Replace /abs/path/to/data with your absolute path):
```command
docker run --name topogpu -it --gpus all \
  -v /abs/path/to/data:/data \
  -w /opt/app --entrypoint /bin/bash \
  seravee08/topogpu:latest
```
- Run program on your data.

### Common failure messages
- readArrayFromBin: failure!  
  The input file path is wrong or not mounted at /data. Verify your host path and -v/--mount mapping.

- Error: boundary matrix buffer limit reached...  
  Increase the buffer, e.g.: -bufSize 3000 (or larger as needed. Default 2000).

## Pyton Module

Prebuilt binary wheels of TopoGPU are available for Python 3.10 and 3.12 on Linux (x86_64).  
Install the wheel that matches your Python version:
```
# Python 3.10
pip install https://github.com/seravee08/GPU-Computation-of-Persistent-Homology-for-Image-Data/raw/main/releases/topogpu-0.1.0-cp310-cp310-linux_x86_64.whl

# Python 3.12
pip install https://github.com/seravee08/GPU-Computation-of-Persistent-Homology-for-Image-Data/raw/main/releases/topogpu-0.1.0-cp312-cp312-linux_x86_64.whl
```

## Source Codes

To be released upon paper acceptance.

