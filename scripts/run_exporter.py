import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", required=True, help="Unique name for this training run (e.g., mlp_relu_64x64)")
args = parser.parse_args()

os.makedirs(f"logs/{args.run_id}", exist_ok=True)

files_to_move = ["logs/val_loss_log.csv", "logs/full_val_preds.csv"]
for f in files_to_move:
    if os.path.exists(f):
        shutil.copy(f, f"logs/{args.run_id}/{os.path.basename(f)}")

forecast_src = "predictions/gbm_val_forecast.csv"
forecast_dst = f"predictions/{args.run_id}_forecast.csv"
if os.path.exists(forecast_src):
    shutil.copy(forecast_src, forecast_dst)

print(f"Export complete. Run: {args.run_id}")
