# CHANGELOG
All updates will be recorded here

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
