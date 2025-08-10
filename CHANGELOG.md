# CHANGELOG
All updates will be recorded here

## [0.2.7] - 2025-08-10

### Added
- **scripts/dashboard_app.py**:
  - Fully interactive Streamlit dashboard for exploring model performance.
  - Allows users to upload `val_loss_log.csv`, `full_val_preds.csv`, and forecast outputs for dynamic visualization.
  - Includes toggles for run selection, forecast smoothing, and epoch slicing.

- **scripts/run_exporter.py**:
  - Utility script for exporting forecast runs to a standardized format.
  - Automatically relocates `val_loss_log.csv`, `full_val_preds.csv`, and forecast CSVs to `logs/{run_id}/` and `predictions/{run_id}_forecast.csv`.
  - Enables leaderboard-style run tracking and compatibility with the dashboard interface.

### Enhanced
- **src/train.cpp**:
  - Introduced `--run_id` CLI argument for clean log and forecast export under a unified identifier.
  - Automatically creates `logs/{run_id}/` and copies `val_loss_log.csv`, `full_val_preds.csv`, and forecast outputs.
  - Prepares run structure for ingestion by `dashboard_app.py`.

### Notes
- This release completes **Checkpoint 2.6–2.7: Dashboard Integration and Exportable Runs**.
  - Dashboard now enables real-time monitoring and comparison of multiple model runs.
  - All logs and forecast outputs are structured for seamless integration and public display.

### Next
- **Checkpoint 2.8: Forecast Leaderboard + Ensemble Evaluation**:
  - Add dashboard comparison of multiple models over same horizon.
  - Compute ensemble statistics (MAE, RMSE, directional accuracy) across runs.
  - Enable exporting summaries for model reports or notebooks.


## [0.2.6] - 2025-08-05

### Fixed
- **src/train.cpp**:
  - Patched bug in `train_val_split` logic and tensor misalignment during dataset conversion.
  - Ensured proper shape consistency between training and validation batches in tiny-dnn.
  - Resolved compilation failure due to incorrect call to `tiny_dnn::mse::f()`.

- **CMakeLists.txt**:
  - Linked `neural.cpp` to `trainer` target to expose implementation of core training functions.

### Added
- **src/neural.cpp**:
  - Implemented `build_mlp()`, `save_scaler()`, and `standardize_dataset()` to support modular neural training.
  - Enables clean linkage and scalable reuse across training and forecasting pipelines.

- **scripts/plot_paths.py**:
  - Forecast visualization scaffold to display predicted vs. actual values over a multi-step forecast horizon.
  - Ingests `forecast_out.csv` and provides intuitive overlay for evaluation.

### Enhanced
- **include/neural.hpp**:
  - Streamlined MLP architecture generation with named activation support (`relu`, `tanh`, etc.).
  - Added inverse-transform helpers to `StandardScaler` for debugging predictions in natural scale.

### Notes
- This release finalizes the diagnostics infrastructure and debugging improvements begun in [0.2.5].
  - All core neural utilities are now modularized and build-ready.
  - Future checkpoints will build on this stable foundation for real-time dashboards and ensemble training.

### Next
- **Checkpoint 2.6: Interactive Dashboard (Optional)**:
  - Integrate Streamlit for on-the-fly inspection of training logs and forecast files.
  - Allow upload of `.csv` model predictions and visualize comparisons.
  - Introduce model toggles, slider-based epoch selection, and overlay customization.


## [0.2.5] - 2025-08-04

### Added
- **scripts/plot_predictions.py**:
  - Visualization script to plot predicted vs. true values across training epochs.
  - Consumes `val_loss_log.csv` and `full_val_preds.csv` to track model performance and detect overfitting.

- **scripts/residual_analysis.py**:
  - Residual diagnostics tool for evaluating model error patterns.
  - Computes residuals (true - predicted), plots rolling mean/variance, and supports qualitative analysis of bias or drift.

### Enhanced
- **src/train.cpp**:
  - Logs validation loss per epoch to `val_loss_log.csv` for training diagnostics.
  - Exports full validation set predictions at every epoch to `full_val_preds.csv`, aiding in longitudinal model inspection.
  - Preserves backwards compatibility with one-shot validation export and forecast output.

### Notes
- This release completes **Checkpoint 2.5: Visualization & Diagnostics**, making it easier to interpret training behavior over time.
  - Residual analysis helps surface systematic prediction errors (e.g., trend-following lag or under-reactivity).
  - Epoch-level logs support charting validation curves and aligning prediction patterns to training dynamics.

- These additions enable future:
  - Regression diagnostic tests (e.g., heteroscedasticity, autocorrelation).
  - Visualizations for model comparison across multiple architectures or SDE types.
  - Reporting templates for Jupyter or Streamlit-based dashboards.

### Next
- **Checkpoint 2.6: Interactive Dashboard (Optional)**:
  - Wrap existing plots and logs into a single exploratory UI using Streamlit.
  - Allow upload of `.csv` prediction files and compare different trained models interactively.
  - Integrate real-time training logs for active monitoring.


## [0.2.3] - 2025-08-03

### Added
- **docs/EVALUATION.md**:
  - New evaluation guide for neural approximator results.
  - Details how to interpret validation predictions, compute key metrics (MSE, MAE, RMSE), and plot predicted vs. true paths.
  - Includes troubleshooting tips for diagnosing underfitting, overfitting, or poor generalization.

### Enhanced
- **README.md**:
  - Added **Neural Training & Evaluation** section to project overview.
  - Provides quick instructions for generating data, running the trainer, and evaluating learned models.
  - Links to detailed training and evaluation docs for full workflows.

- **src/train.cpp**:
  - Introduced CLI flag `--forecast_horizon` to generate multi-step predictions.
  - Automatically saves forecasted trajectories to `predictions/` for later analysis.
  - Prepares output data for visualization scripts referenced in `docs/EVALUATION.md`.

### Notes
- This release completes **Checkpoint 2.4: Extended Evaluation & Forecasting**, expanding the project beyond basic training:
  - Users can now produce both one-step and multi-step forecasts.
  - Evaluation docs and README improvements make it easier to interpret and showcase results.
- These additions set the foundation for:
  - Automated comparison plots of learned vs. true SDE paths.
  - Advanced evaluation metrics (e.g., R², directional accuracy).
  - Future neural SDE experiments with stochastic volatility processes.

### Next
- **Checkpoint 2.5: Visualization & Diagnostics**:
  - Implement Python scripts to plot training loss curves and forecast trajectories.
  - Add residual error heatmaps for qualitative analysis.
  - Build reporting utilities to summarize model performance across multiple SDE types.



## [0.2.2] - 2025-08-02

### Enhanced
- **src/train.cpp**:
  - Added CLI support for variable hidden layers (`--hidden`) and activation functions (`--act`) for flexible network architectures.
  - Implemented **early stopping** based on validation MSE with configurable patience.
  - Added **learning rate decay on plateau** with `--lr_decay`, `--lr_patience`, and `--min_lr` flags.
  - Integrated **best model checkpointing**, automatically saving the lowest validation loss model.
  - Exports **validation predictions** (inverse-scaled) to CSV via `--pred_out`.
  - Auto-creates necessary output directories for models and predictions, improving usability.

- **include/neural.hpp**:
  - Extended `StandardScaler` with inverse transformation methods for restoring original scale.
  - Enhanced `make_mlp()` to accept arbitrary hidden layer configurations.
  - Provided configurable activation layers to match CLI options for the trainer.

- **docs/TRAINING.md**:
  - Added dedicated neural training guide detailing usage of the trainer CLI, argument table, example commands, output descriptions, and tips for running from different working directories.

### Notes
- This release completes **Checkpoint 2.3: Neural Training Enhancements**, introducing advanced training features for better generalization and usability.
- Early stopping and LR scheduling improve convergence behavior, while prediction exports aid diagnostics and visualization.
- Training guide makes the neural approximation workflow clearer for contributors and recruiters.

### Next
- **Checkpoint 2.4: Extended Evaluation & Forecasting**:
  - Add support for test-set evaluation, multi-step forecasting, and residual analysis.
  - Build visualization scripts to compare predicted vs. true SDE paths over time.
  - Prepare codebase for experimenting with stochastic volatility models and neural SDE variants.


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
