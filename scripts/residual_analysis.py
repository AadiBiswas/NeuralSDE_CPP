import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_residuals(pred_file, output_dir="evaluation"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(pred_file)
    df["residual"] = df["true"] - df["pred"]

    # Line plot of residuals
    plt.figure(figsize=(10, 4))
    plt.plot(df["idx"], df["residual"], label="Residuals")
    plt.axhline(0, color="gray", linestyle="--")
    plt.title("Residuals over Index")
    plt.xlabel("Index")
    plt.ylabel("Residual")
    plt.tight_layout()
    out_path1 = os.path.join(output_dir, "residuals_line.png")
    plt.savefig(out_path1)
    print(f"Saved: {out_path1}")

    # Histogram of residuals
    plt.figure(figsize=(6, 4))
    sns.histplot(df["residual"], kde=True, bins=30)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.tight_layout()
    out_path2 = os.path.join(output_dir, "residuals_hist.png")
    plt.savefig(out_path2)
    print(f"Saved: {out_path2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to prediction CSV")
    parser.add_argument("--out", type=str, default="evaluation", help="Directory to save output plots")
    args = parser.parse_args()
    plot_residuals(args.file, args.out)
