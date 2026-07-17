# Statistical Gradient Filtering

This repository contains the official implementation of the following paper:

> **[Statistical Gradient Filtering for Geometry Optimization Under Limited Observations (ACM SIGGRAPH 2026)](https://wonjongg.me/assets/pdf/SGF.pdf)**<br>
> Wonjong Jang, Gwangjin Ju, Seungyong Lee

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

### 3. Configuration

Before running the tracker, open `tracker.py` and modify the paths in the `__init__` function to match your local environment settings.

## Run

### 1. Extract Surface Normal Maps
First, run Pixel3DMM to extract the surface normal maps from your input data. Please refer to the instructions provided in the Pixel3DMM repository for detailed steps.

### 2. Run tracker.py
Our `tracker.py` is based on `pixel3dmm/src/pixel3dmm/tracking/tracker.py`. Instead of their PCA coefficients tracker, use our 3D vertex tracker with statistical gradient filtering:
```bash
python tracker.py
```
