# Evaluation Guide

This document explains quick checks to evaluate the neural approximator on SDE paths.

## 1) Files Produced
- **Validation one-step predictions:** `predictions/*_val_preds.csv` (columns: `idx,true,pred`)
- **Recursive multi-step forecast:** `predictions/*_val_forecast.csv` (columns: `step,true,pred`)
- **Model + scalers:** under `models/` or your custom `--model_out` directory

> Note: `true` may be `NaN` in the forecast CSV once the horizon exceeds available ground truth.

---

## 2) Quick Metrics (Python)
```bash
python3 - << 'PY'
import pandas as pd, numpy as np

preds = pd.read_csv("predictions/gbm_val_preds.csv").dropna()

y_true = preds["true"].values
y_pred = preds["pred"].values

mse = np.mean((y_pred - y_true)**2)
mae = np.mean(np.abs(y_pred - y_true))
rmse = np.sqrt(mse)

print(f"MSE:  {mse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
PY
```

## 3) Quick Plots (Python)
```bash
python3 - << 'PY'
import pandas as pd
import matplotlib.pyplot as plt

preds = pd.read_csv("predictions/gbm_val_preds.csv").dropna()

plt.figure(figsize=(9,4))
plt.plot(preds["idx"], preds["true"], label="True")
plt.plot(preds["idx"], preds["pred"], label="Pred")
plt.title("Validation — One-step Predictions")
plt.xlabel("Validation index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("predictions/gbm_val_preds_plot.png")
print("Saved predictions/gbm_val_preds_plot.png")

fc = pd.read_csv("predictions/gbm_val_forecast.csv")
plt.figure(figsize=(9,4))
plt.plot(fc["step"], fc["pred"], label="Forecast (pred)")
if fc["true"].notna().any():
    plt.plot(fc["step"], fc["true"], label="Ground truth", alpha=0.7)
plt.title("Validation Tail — Recursive Multi-step Forecast")
plt.xlabel("Forecast step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("predictions/gbm_val_forecast_plot.png")
print("Saved predictions/gbm_val_forecast_plot.png")
PY
```

## 4) Reading the Plots
- One-step predictions: curves should broadly track each other; large gaps imply underfit/overfit.

- Recursive forecast: expect drift over long horizons; smaller drift indicates better learned dynamics.


## 5) Common Tweaks
Increase ```--window``` or capacity ```(--hidden "128,64")``` if underfitting.

Lower ```--lr```, enable early stopping, or reduce capacity if overfitting.

For OU (mean-reversion), shorter windows often suffice; for GBM, a bit longer can help.