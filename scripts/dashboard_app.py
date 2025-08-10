import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Neural SDE Dashboard", layout="wide")
st.title("📊 Neural SDE Training Dashboard")

st.markdown("""
Compare validation losses, predictions, and forecasts from different model runs.  
Each run should live in a subdirectory under `logs/`, e.g., `logs/run1/`, `logs/run2/`.
""")

# --- Detect available runs ---
log_root = "logs"
available_runs = sorted([d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))])

selected_run = st.sidebar.selectbox("📁 Select Model Run", available_runs)
run_path = os.path.join(log_root, selected_run)

val_loss_path = os.path.join(run_path, "val_loss_log.csv")
preds_path = os.path.join(run_path, "full_val_preds.csv")
forecast_path = f"predictions/{selected_run}_forecast.csv"

# Allow upload override
st.sidebar.markdown("---")
val_loss_file = st.sidebar.file_uploader("Override: Validation Loss CSV", type="csv", key="loss")
preds_file = st.sidebar.file_uploader("Override: Validation Predictions CSV", type="csv", key="preds")
forecast_file = st.sidebar.file_uploader("Override: Forecast CSV", type="csv", key="forecast")

# Fallback to run defaults
if not val_loss_file and os.path.exists(val_loss_path):
    val_loss_file = open(val_loss_path, "r")
if not preds_file and os.path.exists(preds_path):
    preds_file = open(preds_path, "r")
if not forecast_file and os.path.exists(forecast_path):
    forecast_file = open(forecast_path, "r")

# --- Visualize Loss ---
if val_loss_file:
    val_loss_df = pd.read_csv(val_loss_file)
    st.subheader("📉 Validation Loss Curve")
    fig, ax = plt.subplots()
    ax.plot(val_loss_df["epoch"], val_loss_df["val_loss"], label="Val Loss", color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Validation MSE per Epoch")
    ax.grid(True)
    st.pyplot(fig)

# --- Predictions View ---
if preds_file:
    preds_df = pd.read_csv(preds_file)
    epochs = preds_df["epoch"].unique()
    selected_epoch = st.slider("🔁 Select Epoch to View Predictions", int(epochs.min()), int(epochs.max()), int(epochs.max()))
    filtered = preds_df[preds_df["epoch"] == selected_epoch]

    st.subheader(f"📈 Predictions vs. Ground Truth (Epoch {selected_epoch})")
    fig, ax = plt.subplots()
    ax.scatter(filtered["idx"], filtered["true"], label="True", alpha=0.6)
    ax.scatter(filtered["idx"], filtered["pred"], label="Predicted", alpha=0.6)
    ax.set_xlabel("Validation Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --- Forecast Output ---
if forecast_file:
    st.subheader("🔮 Forecast Output")
    forecast_df = pd.read_csv(forecast_file)
    fig, ax = plt.subplots()
    ax.plot(forecast_df["true"], label="True")
    ax.plot(forecast_df["pred"], label="Forecasted")
    ax.set_title("Forecast: True vs Predicted")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.markdown("---")
st.markdown(f"📁 Currently viewing: `{selected_run}`")
