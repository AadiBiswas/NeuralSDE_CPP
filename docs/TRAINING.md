# Neural Training Guide

This guide explains how to train the MLP that approximates SDE dynamics (GBM/OU) using sliding-window samples.

## Quick Start
```bash
# From project root
./build/trainer \
  --data gbm_path.csv \
  --window 20 \
  --epochs 100 \
  --batch 64 \
  --lr 0.001 \
  --hidden "128,64" \
  --act relu \
  --early_stop true \
  --patience 15 \
  --lr_decay 0.5 \
  --lr_patience 7 \
  --min_lr 1e-5 \
  --model_out models/gbm_mlp.tnn \
  --pred_out predictions/gbm_val_preds.csv
