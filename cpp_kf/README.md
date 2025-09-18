# GPFilterKF (Kalman Filter for XYZ Trajectory)

This is a minimal C++ framework to filter 3D position (x, y, z) measurements from a CSV file using a constant-acceleration Kalman Filter. The state for each axis is `[pos, vel, acc]`. The measurement is position only.

The project is designed for Ubuntu (g++/CMake), but it is cross-platform and should build on Windows as well.

## Build (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake

cd cpp_kf
mkdir -p build && cd build
cmake ..
cmake --build . -j
```

## Run

By default it will try to read the CSV file at `../data/trajectory.csv` relative to the project root.

```bash
./gpfilter_kf --csv "../../data/trajectory.csv" --dt 0.01 \
  --q_pos 1e-4 --q_vel 1e-3 --q_acc 1e-2 --r 1e-3 --out "filtered_xyz.csv"
```

Arguments:
- `--csv` path to input CSV with at least 3 columns (x,y,z). Extra columns will be ignored.
- `--dt` sampling period seconds.
- `--q_pos`, `--q_vel`, `--q_acc` diagonal process noise for each state element per axis.
- `--r` measurement noise variance (position).
- `--out` output CSV path. If omitted, prints summary to stdout.

## CSV format

Each row must contain at least three floating point numbers: `x,y,z,...`

## Notes

- This repository provides just the framework and a simple reference implementation. You can tune Q/R according to your sensor characteristics.
- The transition matrix models constant acceleration with jerk as process noise.