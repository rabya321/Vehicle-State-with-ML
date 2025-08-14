# Vehicle Trajectory Prediction using Machine Learning Models

This project implements a **Recurrent neural networks** to predict vehicle trajectory using vehicle sensor data. The model estimates local velocity changes and reconstructs the vehicle’s global trajectory over time.

## Features

- Multi-step trajectory prediction using GRU.
- Standardizes input and target features for better learning.
- Converts local velocity deltas to global coordinates using yaw information.
- Visualizes predicted vs. actual trajectory (example plots included using synthetic data).
- Computes positional error metrics (MSE, RMSE, MAE, R², Euclidean distance).

---

## Usage

1. **Install requirements:**

```bash
pip install pandas numpy matplotlib scikit-learn torch

2. **Run the example workflow:**

The code can be executed with synthetic data generated automatically for testing the model and plotting routines.

3. **Adapt the code (optional):**
Users can modify the preprocessing and model scripts to fit their own data pipeline if they have private datasets.

