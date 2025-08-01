# CHANGELOG
All updates will be recorded here

## [0.2.1] - 2025-08-01

### Fixed
- **include/neural.hpp**:
  - Corrected improper use of non-existent `activation_type` in `make_mlp()` factory.
  - Replaced erroneous activation type logic with `relu_layer` as default.
  - Removed unused `identity_layer()` to resolve compilation errors.

- **src/train.cpp**:
  - Fixed undefined `mse_loss` error by explicitly including `tiny_dnn/loss/loss_function.h`.
  - Adjusted activation handling to match updated neural scaffolding.
  - Confirmed CLI training workflow compiles, executes, and saves `.tnn` weights on valid data.

- **src/data_loader.cpp**:
  - Hardened dataset loading against malformed CSV entries.
  - Added line-by-line validation with `std::stof` inside `try/catch` to skip non-numeric data.
  - Added warning output for corrupt or empty lines, improving transparency in dataset preprocessing.

### Notes
- This patch resolves critical bugs in **Checkpoint 2.2: Neural Network Scaffolding**, ensuring that:
  - The entire training pipeline compiles cleanly across macOS Clang environments.
  - CLI execution fails gracefully if input data is missing or malformed.
  - Future checkpoints (e.g., 2.3–2.4) can safely build on top of this corrected base.

### Next
- **Checkpoint 2.3: Neural Training Enhancements**:
  - Add CLI parameters for architecture tuning (`--hidden 128 64`).
  - Implement validation-based early stopping and learning rate decay.
  - Export predictions vs. ground truth to `.csv` for post-training diagnostics and visualization.


## [0.2.0] - 2025-07-30

### Added
- **external/tiny-dnn**:
  - Added as a Git submodule to provide a lightweight, header-only neural network library.
  - Enables fast prototyping of regression-style networks for SDE approximation.

- **include/neural.hpp**:
  - Neural scaffolding for time-series forecasting.
  - Provides:
    - StandardScaler struct for per-feature normalization.
    - `make_mlp()` factory function for building fully connected regression networks with custom hidden layers.
  - Designed to integrate seamlessly with tiny-dnn and the existing data loader.

- **src/train.cpp**:
  - End-to-end training pipeline for Neural SDE approximation:
    - Loads processed time-series data via `data_loader`.
    - Applies standardization to improve numerical stability.
    - Defines and trains a feedforward MLP on sliding-window input samples.
    - Logs train/validation MSE per epoch.
    - Saves trained model weights (`.tnn`) and scaler stats (`xscaler_stats.csv`, `yscaler_stats.csv`) to `/models`.

### Enhanced
- **CMakeLists.txt**:
  - Extended build config to include `external/tiny-dnn` headers.
  - Added new `trainer` executable for running neural approximation workflows independently of the simulator.
  - Maintains modular separation between simulation (Euler–Maruyama) and learning (MLP).

### Notes
- This release completes **Checkpoint 2.2: Neural Network Scaffolding**:
  - The project now supports **data generation (GBM, OU)** → **dataset preprocessing** → **neural network training** entirely in C++.
  - Sets the foundation for future experiments in **data-driven drift/diffusion learning** and potential real-time forecasting.

### Next
- **Checkpoint 2.3: Neural Training Enhancements**:
  - Add early stopping, learning rate scheduling, and CLI-configurable architecture (`--hidden 128 64`).
  - Export predictions vs. true SDE paths to CSV for visual comparison.
  - Prepare for experimentation with stochastic volatility processes and Neural SDE variants.


## [0.1.2] - 2025-07-29

### Added
- **include/data_loader.hpp**:
  - Declares reusable `Dataset` and `Sample` typedefs for training format.
  - Provides forward declaration for the dataset loader used in neural training.
  - Serves as a standardized interface for all sliding window–based training pipelines.

- **src/data_loader.cpp**:
  - Loads time-series data from GBM/OU simulation CSV files.
  - Extracts overlapping sliding windows of length `n` as inputs, with the `n+1`th value as label.
  - Normalizes feature shape to be compatible with FCNs or MLP-style networks in future phases.

### Enhanced
- **CMakeLists.txt**:
  - Updated build configuration to compile `data_loader.cpp` alongside the main simulator.
  - Enables integration of neural preprocessing into full pipeline.

### Notes
- This completes **Checkpoint 2.1: Data Preprocessing for Neural Approximation**.
- The simulator can now export raw paths, and the loader can consume them into training-ready samples.
- Prepares foundation for neural net scaffolding and training loop in C++ via `tiny-dnn`.

### Next
- **Checkpoint 2.2: Neural Network Scaffolding**:
  - Import `tiny-dnn` and build a lightweight feedforward architecture.
  - Define training loop, loss function, and optimizer.
  - Train on preprocessed GBM data and validate output against ground truth.


## [0.1.1] - 2025-07-28

### Added
- **include/OU.hpp**:
  - Introduced **Ornstein–Uhlenbeck (OU)** stochastic process class.
  - Implements mean-reverting dynamics via `theta`, `mu`, and `sigma` parameters.
  - Subclass of the generic `SDE` interface for interchangeable simulation.

- **plot_paths.py**:
  - Python visualization script using `matplotlib`.
  - Reads simulation outputs (`gbm_path.csv`, `ou_path.csv`) and plots trajectories for side-by-side comparison.
  - Saves rendered output to `sde_paths.png`.

### Enhanced
- **src/main.cpp**:
  - Modularized simulation logic into a reusable `simulate()` function accepting any `SDE`-compliant object.
  - Extended main routine to run both GBM and OU processes, each outputting to its own CSV.
  - Provides clean console output upon file generation.

### Notes
- This completes **Phase 1.3: OU Integration + Visualization**:
  - The simulator now supports both **trend-following** (GBM) and **mean-reverting** (OU) dynamics.
  - Users can easily extend support to additional processes via the `SDE` interface.
  - Visualization script improves interpretability and quick debugging of stochastic paths.

### Next
- **Phase 2.0: Neural SDE Approximation**:
  - Create a neural net that learns the underlying drift and diffusion functions of a process from simulated data.
  - Implement training pipeline using `tiny-dnn` or `libtorch`.
  - Compare learned process outputs against true SDE samples for forecast accuracy and realism.


## [0.1.0] - 2025-07-27

### Added
- **include/SDE.hpp**:
  - Abstract base class for all stochastic differential equations.
  - Declares virtual `drift()` and `diffusion()` functions to support multiple process types.

- **include/GBM.hpp**:
  - Implements **Geometric Brownian Motion** as a subclass of `SDE`.
  - Encodes process dynamics with parameters `mu` (drift) and `sigma` (volatility).

- **src/main.cpp**:
  - First working simulation of GBM using the **Euler–Maruyama scheme**.
  - Outputs CSV file (`gbm_path.csv`) containing time-series path for plotting or validation.

- **CMakeLists.txt**:
  - Basic build system for the project.
  - Enables clean compilation with C++17 and `include/` header support.

### Notes
- This release marks the completion of **Phase 1.2: GBM Simulation Engine** using the Euler–Maruyama method.
- The project scaffolding is now fully initialized, and simulation output is functional.
- GBM serves as a test case for later neural approximation and generalization to other SDEs (e.g., OU, Jump Diffusion).

### Next
- **Phase 1.3**:
  - Implement Ornstein–Uhlenbeck (OU) process simulation.
  - Add optional command-line arguments to control process parameters.
  - Generate Python-based visualizations using `matplotlib` for output inspection.
