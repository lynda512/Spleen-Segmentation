# Spleen Segmentation — 3D U-Net on CT Volumes

A 3D segmentation pipeline for automated spleen delineation in abdominal CT scans, built on the
[Medical Segmentation Decathlon](http://medicaldecathlon.com/) Task09_Spleen dataset.

## Problem

Manual organ segmentation in CT volumes is slow and inter-observer variability is well documented
in the radiology literature. This project trains a 3D U-Net to automatically segment the spleen
from a full CT volume, and evaluates it the way a safety-relevant imaging system should be
evaluated: on a held-out test set, with per-case results and explicit failure-case review, not
just a single headline number.

## Dataset

- **Source:** Medical Segmentation Decathlon, Task09_Spleen
- **41 labelled 3D CT volumes** (portal venous phase), variable resolution and slice count
- **Split:** 70% train / 15% validation / 15% test, seeded random shuffle (not file order) — see
  `src/data/msd_spleen_datamodule.py`. The test split is held out entirely from training and from
  checkpoint selection, so the reported test Dice is an honest generalization estimate rather than
  a number implicitly optimized for via "pick the best val epoch."

## Method

- **Architecture:** 3D U-Net (MONAI implementation), 5 levels, channels 16→256, 2 residual units
  per block (`src/models/unet_3d.py`)
- **Preprocessing:** resample to 1.5mm isotropic spacing, RAS orientation, intensity clipping to a
  soft-tissue HU window, foreground cropping (`src/data/transforms.py`)
- **Training:** patch-based (96³ patches sampled with 1:1 positive:negative ratio), Dice + cross-entropy
  loss, Adam optimizer, early stopping on validation Dice (`src/train.py`)
- **Inference:** sliding-window prediction over full volumes at validation/test/deploy time, since
  the model is trained on fixed-size patches but needs to predict on variable-size volumes
  (`src/engine.py`)
- **Experiment tracking:** Weights & Biases (loss/Dice curves, run config, best-checkpoint metric)

## Results

*To be filled in after the current training run — see `reports/test_results.json` and the linked
W&B run for the actual numbers. Reporting honestly here rather than backfilling a number before the
model has actually been trained.*

| Metric | Value |
|---|---|
| Test Dice (mean, n=6) | TBD |
| Best case Dice | TBD |
| Worst case Dice | TBD |
| Training epochs | TBD |

## Repository structure

```
src/
  config.py                    # all paths and hyperparameters, env-var overridable
  engine.py                    # shared loss/metric/sliding-window inference logic
  train.py                     # training loop, checkpointing, W&B logging
  evaluate.py                  # held-out test set evaluation, per-case Dice
  predict.py                   # inference on a single new (unlabeled) volume
  data/
    msd_spleen_datamodule.py   # dataset loading, train/val/test split
    transforms.py               # preprocessing + augmentation pipelines
  models/
    unet_3d.py                  # model definition
  qc/
    check_dataset.py            # data quality checks (shape/spacing/intensity sanity)
notebooks/
  00_explore_dataset.ipynb      # initial data exploration
  01_train_and_evaluate.ipynb   # end-to-end Colab notebook: download, train, evaluate, visualize
reports/
  data_quality/                 # output of qc/check_dataset.py
  training_history.json         # per-epoch loss/Dice log
  test_results.json             # final held-out test results, per-case
```

## Running this project

This was trained on Google Colab (free-tier T4 GPU). The Colab notebook
(`notebooks/01_train_and_evaluate.ipynb`) handles the full flow — clone, install, download data,
QC, train, evaluate, visualize — and is the easiest way to reproduce results.

To run locally or adapt to another environment:

```bash
pip install -r requirements.txt

# Point config at wherever your data/checkpoints/reports should live
export SPLEEN_DATA_DIR=/path/to/data
export SPLEEN_CKPT_DIR=/path/to/checkpoints
export SPLEEN_REPORTS_DIR=/path/to/reports

# Download Task09_Spleen into $SPLEEN_DATA_DIR/raw/Task09_Spleen (imagesTr/, labelsTr/, dataset.json)
# then:
python -m src.qc.check_dataset   # sanity-check the data
python -m src.train --epochs 60
python -m src.evaluate
python -m src.predict --input path/to/volume.nii.gz --output prediction.nii.gz
```

## Known limitations / honest gaps

- **41 cases total** is a small dataset by deep learning standards; results should be read as a
  proof-of-concept on this benchmark, not a clinically validated system.
- `predict.py` outputs predictions in the *resampled* (1.5mm isotropic, foreground-cropped) space
  used internally for inference, not the original volume's native grid — for deployment, this would
  need to be resampled back onto the input image's original space.
- No uncertainty/confidence estimate is currently produced alongside the predicted mask — for a
  safety-relevant use case, knowing *when the model is unsure* matters as much as the prediction
  itself.
- Dice score alone doesn't capture boundary-level accuracy; Hausdorff distance or average surface
  distance would be a more complete evaluation, especially for borderline cases.

## What's next

- [ ] Add Hausdorff distance / average surface distance as secondary metrics
- [ ] Resample predictions back to native image space in `predict.py`
- [ ] MC-dropout or ensemble-based uncertainty estimation
- [ ] Cross-validation instead of a single train/val/test split, given the small dataset size
- [ ] Compare against a 2D slice-based baseline to quantify the benefit of full 3D context

## Stack

Python · PyTorch · MONAI · nibabel · Weights & Biases
