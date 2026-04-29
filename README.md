# NumPy-only 3-layer MLP on EuroSAT_RGB

This project implements a **3-layer MLP** (Input -> Hidden1 -> Hidden2 -> Output) for EuroSAT land-cover classification using only `numpy` for model math and backpropagation.

## Project Structure
- `script.py`: main entry point and CLI
- `mlp_model.py`: model definition (`MLP3Layer`) and model-level utilities
- `data_pipeline.py`: dataset loading, split, normalization, mini-batch iterator
- `training.py`: training loop (`run_training`)
- `hp_search.py`: grid/random hyperparameter search
- `checkpoint_eval.py`: checkpoint save/load, test evaluation, confusion matrix export
- `encode_loss_calc.py`: loss and gradient utilities
- `activate_functions.py`: activation functions and derivatives
- `utils.py`: list parsing and CSV output helpers

## Implemented Features
- Configurable hidden dimension via `--hidden-dim`
- Activation switch: `relu`, `sigmoid`, `tanh`
- Loss switch: `cross_entropy`, `mse`
- Mini-batch SGD + inverse-time LR decay + L2 weight decay
- Train/val/test split with train-stat normalization
- Save/load best checkpoint by validation accuracy
- Test-set accuracy + confusion matrix export
- Optional export of misclassified test images with true/predicted labels
- Hyperparameter tuning via grid search or random search
- Optional training curve export (`--plot-curves`)

## Environment
Required Python packages:
- `numpy`
- `Pillow`
- `matplotlib` (for curve plotting)

## Common Commands

### 1) Train + save best model + test evaluation
```powershell
python .\script.py --data-dir .\EuroSAT_RGB --epochs 20 --batch-size 64 --lr 0.01 --lr-decay 0.01 --weight-decay 0.01 --checkpoint-path .\best_model.npz --confusion-matrix-csv .\confusion_matrix.csv
```

### 2) Quick sanity run (small sample)
```powershell
python .\script.py --data-dir .\EuroSAT_RGB --epochs 1 --batch-size 16 --max-per-class 2 --checkpoint-path .\best_model_smoke.npz --confusion-matrix-csv .\confusion_matrix_smoke.csv
```

### 3) Load best checkpoint and evaluate only
```powershell
python .\script.py --data-dir .\EuroSAT_RGB --checkpoint-path .\best_model.npz --load-best --confusion-matrix-csv .\confusion_matrix_loaded.csv
```

### 4) Grid search
```powershell
python .\script.py --data-dir .\EuroSAT_RGB --search-mode grid --search-epochs 3 --grid-lrs 0.01,0.001 --grid-hidden-dims 128,256 --grid-weight-decays 0.0,0.0001 --results-csv .\search_results_grid.csv
```

### 5) Random search
```powershell
python .\script.py --data-dir .\EuroSAT_RGB --search-mode random --search-epochs 3 --num-trials 10 --results-csv .\search_results_random.csv
```

### 6) Train and save loss/accuracy curves
```powershell
python .\script.py --data-dir .\EuroSAT_RGB --epochs 20 --plot-curves --loss-plot .\loss_curve.png --acc-plot .\val_acc_curve.png
```

### 7) Save misclassified test images
```powershell
python .\script.py --data-dir .\EuroSAT_RGB --checkpoint-path .\best_model.npz --load-best --save-misclassified --misclassified-dir .\misclassified_examples
```

## Key CLI Arguments (Current Defaults)
- `--data-dir`: `EuroSAT_RGB`
- `--hidden-dim`: `256`
- `--activation`: `relu` (`relu|sigmoid|tanh`)
- `--epochs`: `20`
- `--batch-size`: `64`
- `--lr`: `0.01`
- `--lr-decay`: `0.01`  
	effective learning rate at epoch `t`: `lr_t = lr / (1 + lr_decay * (t - 1))`
- `--weight-decay`: `0.01`
- `--loss`: `cross_entropy` (`cross_entropy|mse`)
- `--val-ratio`: `0.2`
- `--test-ratio`: `0.1`
- `--image-size`: `32`
- `--max-per-class`: `None` (disabled by default)
- `--seed`: `42`
- `--search-mode`: `none` (`none|grid|random`)
- `--search-epochs`: `5`
- `--grid-lrs`: `0.01,0.001`
- `--grid-hidden-dims`: `128,256`
- `--grid-weight-decays`: `0.0,0.0001`
- `--num-trials`: `10`
- `--random-lr-min`: `1e-4`
- `--random-lr-max`: `1e-1`
- `--random-hidden-min`: `64`
- `--random-hidden-max`: `512`
- `--random-weight-decay-min`: `1e-6`
- `--random-weight-decay-max`: `1e-2`
- `--results-csv`: `search_results.csv`
- `--checkpoint-path`: `best_model.npz`
- `--load-best`: disabled by default
- `--confusion-matrix-csv`: `confusion_matrix.csv`
- `--plot-curves`: disabled by default
- `--save-misclassified`: disabled by default
- `--misclassified-dir`: `misclassified_examples`
- `--loss-plot`: `loss_curve.png`
- `--acc-plot`: `val_acc_curve.png`

## Output Files
- Checkpoint: `best_model.npz` (or your custom `--checkpoint-path`)
- Search log: `search_results.csv` (or your custom `--results-csv`)
- Confusion matrix: `confusion_matrix.csv` (or your custom `--confusion-matrix-csv`)
- Curves (optional): `loss_curve.png`, `val_acc_curve.png`

## Notes
- Default image size is `32x32` via `--image-size`.
- `--load-best` evaluates with checkpoint metadata (split/normalization settings) for reproducibility.
- In `--load-best` mode, model structure and data preprocessing settings are restored from checkpoint metadata.
