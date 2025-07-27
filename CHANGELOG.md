## [0.1.0] - 2025-07-25

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
