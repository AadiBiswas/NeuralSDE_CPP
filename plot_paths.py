import pandas as pd
import matplotlib.pyplot as plt

gbm = pd.read_csv("gbm_path.csv", names=["t", "x"])
ou  = pd.read_csv("ou_path.csv",  names=["t", "x"])

plt.figure(figsize=(10, 5))
plt.plot(gbm["t"], gbm["x"], label="GBM")
plt.plot(ou["t"], ou["x"], label="OU")
plt.title("Simulated Paths: GBM vs. OU")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sde_paths.png")
plt.show()
