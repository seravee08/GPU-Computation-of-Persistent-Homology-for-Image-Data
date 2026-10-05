# TopoGPU Python API

GPU-accelerated persistent homology of 2D images and 3D volumes (cubical complexes, sublevel-set filtration). Morse matching, topological sort and boundary-path counting run on the GPU; the final boundary-matrix reduction runs multi-threaded on the CPU (PHAT).

## Installation

Pre-built wheels are published under **Releases**. Pick the one matching your Python version (`cp310` = Python 3.10, …), 64-bit Linux, CUDA 12.x runtime and an NVIDIA GPU:

```bash
pip install https://github.com/<user>/<repo>/releases/download/<tag>/topogpu-<ver>-cp310-cp310-linux_x86_64.whl
python -c "import topogpu; print('ok')"
```

If you see `GLIBCXX_3.4.30 not found` inside a conda environment, update its C++ runtime: `conda install -c conda-forge "libstdcxx-ng>=12"`.

## Concepts

| Term | Meaning |
|---|---|
| Image size | 2D: `(imgSize_x, imgSize_y)` = **(height, width)**. 3D: `(imgSize_x, imgSize_y, imgSize_z)` = **(height, width, depth)**. |
| `dtype` | Type of the **on-disk / input data**: `'uchar'`, `'ushort'`, `'int'`, `'float'` (aliases `uint8`, `uint16`, `int32`, `float32`). Internally `'uchar'` is processed as `uint16`, `'ushort'` and `'int'` as `int32`, `'float'` as `float32`. |
| `bndmat_num` | Capacity (pairs per GPU block) of the boundary-relation buffer. Too small → `Error: boundary matrix buffer limit reached`. Increase for noisy / high-detail data; default 2000. |
| `bx, by (, bz)` | Optional CUDA block size override (each 4/8/16/32). `0` = choose automatically. |
| `deviceID` | CUDA device index, `-1` = auto. |

### Result format

Each result is a `float32` NumPy array with **one row per persistence pair**:

* 2D — shape `(N, 7)`: `[dim, birth, death, birth_y, birth_x, death_y, death_x]`
* 3D — shape `(N, 9)`: `[dim, birth, death, birth_z, birth_y, birth_x, death_z, death_y, death_x]`

`dim` is the homology dimension (0, 1 for 2D; 0, 1, 2 for 3D). `birth`/`death` are filtration values. The coordinate columns give the pixel/voxel that creates and destroys the feature. The last row is the **essential pair** of H0: `[0, min_value, inf, 0, …]`. Pairs with `birth == death` are omitted.

---

## Single-image API

### Create

```python
import topogpu
tg = topogpu.create2D(imgSize_x, imgSize_y, dtype, bndmat_num=2000, bx=0, by=0, deviceID=-1, debug=False, verbose=False)
tg = topogpu.create3D(imgSize_x, imgSize_y, imgSize_z, dtype, bndmat_num=2000, bx=0, by=0, bz=0, deviceID=-1, debug=False, verbose=False)
```

Allocation happens here; reuse the object for many images of the same size (or smaller width).

### Methods

| Method | Description |
|---|---|
| `configure(imgSize_x, imgSize_y[, imgSize_z])` | Set the size of the next input. **Call before every `compute_*`.** Width may not exceed the width given at creation. |
| `compute_PH_from_file(filename, dtype)` | Load a `.jpg`/`.png` (2D only) or raw binary file and compute. `dtype` is the element type stored in the file. |
| `compute_PH_from_array(arr)` | Compute from a NumPy array of shape `(H, W)` or `(X, Y, Z)`; converted to the backend dtype if needed. |
| `get_pairNum()` | Number of persistence pairs. |
| `get_results()` | `(N, 7)` / `(N, 9)` array. **Returns a view** into internal memory — call `.copy()` before `reset()` or the next compute. |
| `reset()` | Free the results of the last computation. |
| `get_blockDims()`, `get_chunkDims()` | Chosen CUDA block size and chunk size (diagnostics). |

### Example

```python
import numpy as np, topogpu
from PIL import Image

img = np.asarray(Image.open("scan.jpg").convert("L"))          # (H, W) uint8
tg  = topogpu.create2D(img.shape[0], img.shape[1], "uchar")

tg.configure(img.shape[0], img.shape[1])
tg.compute_PH_from_array(img)
diag = tg.get_results().copy()
tg.reset()

h0 = diag[diag[:, 0] == 0]
h1 = diag[diag[:, 0] == 1]
print(len(h0), "H0 pairs,", len(h1), "H1 pairs")
```

Raw binary volume (3D):

```python
tg = topogpu.create3D(128, 128, 64, "ushort")
tg.configure(128, 128, 64)
tg.compute_PH_from_file("volume_128x128x64.raw", "ushort")
```

---

## Batch API

Processes many **same-sized** images per call. GPU work on one half of the batch overlaps CPU reduction of the other half, giving much higher throughput for small images (typical deep-learning inputs). Results for all images are ready when the call returns.

### Create

```python
tg = topogpu.create2D_batch(batchSize, imgSize_x, imgSize_y, dtype, bndmat_num=2000, bx=0, by=0, deviceID=-1, debug=False, verbose=False)
tg = topogpu.create3D_batch(batchSize, imgSize_x, imgSize_y, imgSize_z, dtype, bndmat_num=2000, bx=0, by=0, bz=0, deviceID=-1, debug=False, verbose=False)
```

`batchSize` is the maximum number of images per `compute_*` call (GPU/host memory is allocated for `ceil(batchSize/2)` images at once). There is no `configure()`: image size is fixed at creation.

### Methods

| Method | Description |
|---|---|
| `compute_PH_from_files(filenames, dtype)` | `filenames`: list of ≤ `batchSize` paths (`.jpg`/`.png` or raw binary; 3D: raw only). |
| `compute_PH_from_array(arr)` | `arr`: C-contiguous `(B, H, W)` or `(B, X, Y, Z)`, `B ≤ batchSize`. |
| `get_fileNum()` | Number of images in the last call. |
| `get_pairNum(fileIdx)` | Pairs of image `fileIdx`. |
| `get_results(fileIdx=None)` | `fileIdx` given → `(N, 7)`/`(N, 9)` **copy** for that image. `None` → list of copies for all images. |
| `reset()` | Free results (also done automatically by the next `compute_*`). |
| `get_blockDims()`, `get_chunkDims()` | Diagnostics. |

### Example — dataset loop

```python
import glob, numpy as np, topogpu

files = sorted(glob.glob("data/*.jpg"))
B = 20
tg = topogpu.create2D_batch(B, 512, 512, "uchar", bndmat_num=700)

diagrams = {}
for i in range(0, len(files), B):
    batch = files[i:i + B]                      # last batch may be shorter
    tg.compute_PH_from_files(batch, "uchar")
    for fn, d in zip(batch, tg.get_results()):  # list of (N, 7) copies
        diagrams[fn] = d
```

### Example — PyTorch tensors

```python
x = torch.rand(16, 512, 512)                                     # (B, H, W)
arr = np.ascontiguousarray((x * 65535).to(torch.int32).cpu().numpy())
tg  = topogpu.create2D_batch(16, 512, 512, "int")
tg.compute_PH_from_array(arr)
diags = tg.get_results()                                         # list of 16 arrays
```

Pass the array in the backend dtype (`'uchar'`→`uint16`, `'ushort'`/`'int'`→`int32`, `'float'`→`float32`) to avoid a hidden conversion copy.

---

## Notes and limitations

* **Lower-star filtration.** Pixels are 2-cells (voxels 3-cells); the filtration is the sublevel set of pixel values. Pad or invert values yourself for superlevel-set / dual complexes.
* **Reserved maximum value.** The maximum value of the input type (`255` for raw 8-bit input, `65535` for `uint16`, `INT_MAX`, `FLT_MAX`) is used as the boundary padding; inputs containing it are rejected. Use a wider dtype if your data hits the maximum.
* **JPEG decoding.** `compute_PH_from_file(s)` decodes with `stb_image`; feeding the same JPEG through PIL/OpenCV in array mode gives pixels differing by ±1 and therefore slightly different diagrams. For bit-exact agreement between modes use lossless formats (PNG, raw) or a single decoder.
* **Memory.** Each object allocates host (pinned) and device buffers proportional to `chunk size × streams (× ceil(batchSize/2))`. Create one object per size/dtype and reuse it.
* **Threads.** CPU reduction uses all hardware threads; inside containers, limit CPUs with care (`--cpus`) to avoid oversubscription.
* **Determinism.** Results are deterministic for a given input and build; the order of rows may vary between runs.
