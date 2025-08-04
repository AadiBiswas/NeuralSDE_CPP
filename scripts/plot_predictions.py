import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_predictions(pred_file, output_dir="evaluation"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(pred_file)

    plt.figure(figsize=(10, 5))
    plt.plot(df["idx"], df["true"], label="True", linewidth=2)
    plt.plot(df["idx"], df["pred"], label="Predicted", linestyle="--")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Predicted vs. True Values")
    plt.legend()
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "pred_vs_true.png")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to prediction CSV")
    parser.add_argument("--out", type=str, default="evaluation", help="Directory to save output plot")
    args = parser.parse_args()
    plot_predictions(args.file, args.out)