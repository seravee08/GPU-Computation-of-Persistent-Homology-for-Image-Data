# GPU-Computation-of-Persistent-Homology-for-Image-Data
CUDA-accelerated persistent homology for 2D/3D image data using cubical complexes.

## Acknowledgments
This software was developed by Fan Wang as part of a research collaboration with Hubert Wagner, Rezaul Chowdhury, and Chao Chen. I thank Dr. Hubert Wagner for valuable discussions that informed the design of this work. A manuscript describing this work is in preparation and will be linked here upon publication.

## 1️⃣ Google Colab

We provide a **TopoGPU-demo** notebook on Google Colab that demonstrates how to use TopoGPU.

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

## 2️⃣ Docker Image

The image `seravee08/topogpu:v2.0.0` (alias `latest`) contains the pre-built C++ binary, all source files, the build scripts, a CUDA 12.6 toolchain, Python 3.10–3.13 and Jupyter, so you can run, rebuild, or develop TopoGPU without installing anything on the host.

### 📌 Prerequisites
**Windows 10/11**
- Docker Desktop with WSL 2 backend enabled.
- NVIDIA GPU + up-to-date Windows NVIDIA driver (≥ 525).
- User in the local `docker-users` group.
- Verify in Windows PowerShell: ```docker run --rm --gpus=all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi```

**Linux**
- NVIDIA GPU + driver (≥ 525) installed on the host.
- NVIDIA Container Toolkit installed and configured for Docker.
- Verify: ```docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi```

### 📌 Run Docker
**Windows 10/11**
- Pull image: ```docker pull seravee08/topogpu:v2.0.0```
- Create container and enter shell (replace `PATH_TO_DATA` with an absolute Windows path):
```command
docker run --name topogpu -it --gpus=all -p 8888:8888 `
  --mount type=bind,source="PATH_TO_DATA",target=/data `
  seravee08/topogpu:v2.0.0
```

**Linux**
- Pull image: ```docker pull seravee08/topogpu:v2.0.0```
- Create container and enter shell (replace `/abs/path/to/data` with your absolute path):
```command
docker run --name topogpu -it --gpus all -p 8888:8888 \
  -v /abs/path/to/data:/data \
  seravee08/topogpu:v2.0.0
```

### 📌 Inside the container
The shell starts in `/opt/topogpu`.

- Run the C++ binary on your data:
```
./build/topoGPU -help
./build/topoGPU -filename /data/mrt_angio_416x512x112_uint16.raw -datatype ushort -height 512 -width 416 -depth 112 -bufSize 3000 -out /result.txt
```

### 📌 Common failure messages
- readArrayFromBin: failure!  
  The input file path is wrong or not mounted at /data. Verify your host path and -v/--mount mapping.

- Error: boundary matrix buffer limit reached...  
  Increase the buffer, e.g.: -bufSize 3000 (or larger as needed. Default 2000).

## 3️⃣ Python Module

Pre-built wheels of TopoGPU are available for **Python 3.10–3.13 on Linux x86_64** (manylinux_2_28; works on Ubuntu, Rocky/RHEL, conda environments, and Google Colab). The CUDA runtime is installed automatically as a pip dependency; the only host requirement is an NVIDIA driver (≥ 525).

### 📌 Install
Pick the wheel matching your Python version from the [Releases page](https://github.com/seravee08/GPU-Computation-of-Persistent-Homology-for-Image-Data/releases) (`cp310` = Python 3.10, `cp311` = 3.11, …) and install it with pip, e.g. for Python 3.12:
```
pip install https://github.com/seravee08/GPU-Computation-of-Persistent-Homology-for-Image-Data/releases/download/v2.0.0/topogpu-2.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```
Verify:
```
python -c "import topogpu; print(topogpu.__file__)"
```

### 📌 Usage
See the Google Colab notebook in [Section 1](#1️⃣-google-colab) for complete examples: single-image / single-volume computation from files and NumPy arrays, batch mode for processing many same-sized images per call, and validation against Cubical Ripser. The full API reference is in [`docs/PYTHON_API.md`](docs/PYTHON_API.md).

## 4️⃣ Source Codes

To be released upon paper acceptance.

