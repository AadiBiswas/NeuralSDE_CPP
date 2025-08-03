# Neural SDE Simulator (C++)

**Modular · Deterministic · Forecastable Stochastic Dynamics**

## Overview  
This project implements a **C++ simulation and visualization engine** for learning and generating time-series data from **stochastic differential equations (SDEs)**.

It supports canonical processes such as **Geometric Brownian Motion (GBM)** and **Ornstein–Uhlenbeck (OU)**, with planned neural extensions to **approximate or forecast** stochastic dynamics from raw trajectories.

Ideal for researchers and developers working at the intersection of **quant finance**, **ML modeling**, and **numerical stochastic simulation**.

---

## 🧠 Core Goals
- Simulate canonical SDEs using numerical solvers (Euler–Maruyama)
- Visualize the behavior of trend-following vs. mean-reverting systems
- Build a modular framework for expanding to custom SDEs
- Extend into neural approximators to forecast drift/diffusion from data
- Prepare for fast C++ inference in trading, modeling, or generative finance

---

## ⚙️ Architecture  

- **SDE Interface**: Abstract class with `drift()` and `diffusion()` virtual methods
- **GBM Process**: Simulates stochastic exponential growth (e.g., stock prices)
- **OU Process**: Simulates mean-reverting stochastic dynamics (e.g., interest rates)
- **Simulator Engine**: Euler–Maruyama scheme for path generation
- **Plotter**: Python script to visualize paths from CSV output

---

## 📈 Metrics Tracked
- ✅ Continuous-time path generation  
- ✅ CSV output of time vs. simulated value  
- ✅ Visualization of multi-process trajectories  
- ✅ Ready for extension to neural networks and uncertainty quantification

## 🔬 Neural Training & Evaluation
In addition to classical SDE simulations, this project includes a neural approximation module that learns drift/diffusion dynamics directly from generated paths.

### 1️⃣ Data Generation
Run the simulator to produce raw paths:
```bash 
./build/NeuralSDE_CPP  
```

This creates ```gbm_path.csv``` and/or ```ou_path.csv``` for use as training data.

### 2️⃣ Train Neural MLP
Use the trainer to learn one-step predictions from sliding windows:
```bash 
./build/trainer \
  --data gbm_path.csv \
  --window 20 \
  --epochs 100 \
  --batch 64 \
  --lr 0.001 \
  --hidden "128,64" \
  --early_stop true \
  --model_out models/gbm_mlp.tnn \
  --pred_out predictions/gbm_val_preds.csv
```

### 3️⃣ Evaluate Model
Quickly visualize validation predictions vs. ground truth:
```bash 
python3 docs/evaluation_plot.py
```

Alternatively, follow the instructions in:

```docs/TRAINING.md``` — Full CLI guide

```docs/EVALUATION.md``` — Metrics, plotting scripts, and diagnostics



---

## 🛠️ How to Build and Run (Mac/Linux)

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/NeuralSDE_CPP.git
cd NeuralSDE_CPP 
```

### Step 2: Compile with CMake
```bash
mkdir build
cd build
cmake ..
make
```

### Step 3: Run the Simulator
```bash
./NeuralSDE_CPP
```
- This generates:

    - gbm_path.csv — Simulated GBM path

    - ou_path.csv — Simulated OU path


### Step 4: Visualize Output with Python
From the root directory:
```bash
python3 plot_paths.py
```
This opens a Matplotlib window and saves the chart to  ```sde_paths.png ```

It shows side-by-side trajectories of GBM vs. OU

## 📁 File Structure Overview
```bash
NeuralSDE_CPP/
├── include/
│   ├── SDE.hpp         # Abstract interface for all stochastic processes
│   ├── GBM.hpp         # Geometric Brownian Motion implementation
│   └── OU.hpp          # Ornstein–Uhlenbeck implementation
│
├── src/
│   └── main.cpp        # Simulation logic and CLI entry point
│
├── build/              # CMake build output
│
├── gbm_path.csv        # Simulated GBM path (generated)
├── ou_path.csv         # Simulated OU path (generated)
├── sde_paths.png       # Output visualization (generated)
│
├── plot_paths.py       # Python script to visualize results
├── CMakeLists.txt      # Build configuration
├── README.md           # You're here!
└── .gitignore

```

## 🔧 Planned Extensions

 - Neural network to learn drift/diffusion from sample paths

 - Extend to Neural SDEs with learned stochastic generators

 - Add Heston / Jump Diffusion models

 - CUDA acceleration (GPU-enabled SDE simulation)

 - QuantLib-style extensions for stochastic volatility modeling

 ## 🧠 Ideal For

- Students exploring stochastic modeling and forecasting

- Researchers testing learned dynamics vs. true processes

- Quants simulating microstructure or synthetic markets

- Engineers experimenting with C++ for scientific computing

 ## License

 MIT License