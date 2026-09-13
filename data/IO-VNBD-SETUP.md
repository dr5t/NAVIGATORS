# IO-VNBD Dataset Setup

This project uses the official [IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark Dataset)](https://github.com/onyekpeu/IO-VNBD) as the primary training dataset for the ML speed estimation pipeline.

## Requirements
* `git`
* `git-lfs` (Git Large File Storage)
* `python3` with `pandas`, `numpy`, and `scipy`

## Download Instructions

Due to the size of the dataset, it is not checked into this repository. Instead, you should clone the official repository using Git LFS directly into the `data` directory.

```bash
# 1. Install Git LFS if you haven't already
brew install git-lfs      # macOS
# sudo apt install git-lfs # Linux

# 2. Initialize Git LFS
git lfs install

# 3. Clone the IO-VNBD repository into the data/IO-VNBD folder
cd data
git clone https://github.com/onyekpeu/IO-VNBD.git

# 4. Pull all large files (CSVs)
cd IO-VNBD
git lfs pull
```

## Dataset Structure

The pipeline expects the data to be structured as it is in the official repository. Specifically, it looks for paired sensor and ground-truth data in the `csv/` directory:
- IMU/Sensor data files starting with `S_`
- Ground truth vehicle files starting with `V_`

```
data/IO-VNBD/csv/
├── S_2020-04-20_17-23-38.csv
├── V_2020-04-20_17-23-38.csv
├── S_2020-04-22_12-45-10.csv
├── V_2020-04-22_12-45-10.csv
...
```

## Parsing and Features

The training pipeline uses the `src.data_prep.iovnbd_parser` to automatically parse these files:
- Accelerometer (`ax`, `ay`, `az`) in $m/s^2$
- Gyroscope (`gx`, `gy`, `gz`) in $rad/s$
- Ground Truth North & East Velocities (`v_n`, `v_e`) computed from `Velocity` and `Heading` in $m/s$

The data is split on a **session-level** (each CSV pair is treated as an independent session) to prevent data leakage between adjacent windows.
