# Statistical Gradient Filtering

This repository contains the official implementation of the following paper:

> **[Statistical Gradient Filtering for Geometry Optimization Under Limited Observations (ACM SIGGRAPH 2026)](https://wonjongg.me/assets/pdf/SGF.pdf)**<br>
> Wonjong Jang, Gwangjin Ju, Seungyong Lee


## Demo

![Vertex optimization on two example frames](media/demo.gif)

## Installation

### 1. Clone the Repository

Make sure to clone this repository with the --recursive flag to properly fetch the Pixel3DMM submodule:

```bash
git clone --recursive https://github.com/wonjongg/stat-grad.git
cd stat-grad
```

Note: If you already cloned the repository without the `--recursive` flag, you can fetch the submodule by running:
```bash
git submodule update --init --recursive
```

### 2. Install Pixel3DMM

Please follow the official [Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm/tree/master) installation guide to install all required dependencies and download the necessary assets (e.g., the FLAME model).

Only two downloads from [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/) are needed:

| Item on the download page | Provides |
| --- | --- |
| **FLAME 2020** (154 MB) | `FLAME2020/generic_model.pkl` |
| **FLAME 2023** — the 103 MB one, *"versions w/ and w/o jaw rotation"* | `FLAME2023/flame2023_no_jaw.pkl` |

Unzip both into `pixel3dmm/src/pixel3dmm/preprocessing/MICA/data/` so that the two `.pkl` files
sit directly in `FLAME2020/` and `FLAME2023/`.

### 3. Configuration

Copy `env_paths.py` over `pixel3dmm/src/pixel3dmm/env_paths.py`, and copy the contents of `assets/` into `pixel3dmm/assets/`. This registers `head_template_noeye`, `EYEHOLE_MASK` and `MOUTHHOLE_MASK`, which `tracker.py` needs.

You may also need to adjust `frame_dst` in `tracker.py`'s `__init__` so that it matches the folder `track.py` produced on your machine (see below).

## Run

### 1. Extract Surface Normal Maps
First, run Pixel3DMM to extract the surface normal maps from your input data. Please refer to the instructions provided in the Pixel3DMM repository for detailed steps.

```bash
cd pixel3dmm
python scripts/run_preprocessing.py --video_or_images_path $PATH_TO_VIDEO
python scripts/network_inference.py model.prediction_type=normals video_name=$VID_NAME
python scripts/network_inference.py model.prediction_type=uv_map  video_name=$VID_NAME
python scripts/track.py video_name=$VID_NAME use_flame2023=True ignore_mica=True
cd ..
```

### 2. Run tracker.py
Our `tracker.py` is based on `pixel3dmm/src/pixel3dmm/tracking/tracker.py`. Instead of their PCA coefficients tracker, use our 3D vertex tracker with statistical gradient filtering:
```bash
python tracker.py video_name=$VID_NAME use_flame2023=True
```

- **Rendering resolution.** `tracker.py` defaults `size` to 512 and builds the renderer from it, since pixel3dmm predicts its normal maps at 512 (its own `tracking.yaml` defaults to 256). Pass `size=...` to override; the ground truth and the renderer always follow the same value.
- **`frame_dst`** (in `__init__`) must match the folder `track.py` actually produced. Upstream builds that postfix from `no_lm` / `no_pho` / `ignore_mica` / `uv` / `normal` only, so the command above yields `_nV1_noPho_noMICA_uv2000.0_n1000.0` rather than `_nV1_noPho_no_jaw_...`.

