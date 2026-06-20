"""
Shared training/evaluation building blocks.

Kept separate from train.py / evaluate.py / predict.py so that all three
scripts use exactly the same loss, metric, and inference logic. This
matters in segmentation work specifically: it's easy to accidentally
report a different (better-looking) Dice at "evaluation time" than what
training actually optimized against, if the two are implemented twice.
"""
from typing import Dict, Tuple

import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from src import config


def get_loss_fn() -> DiceCELoss:
    """
    Combined Dice + cross-entropy loss. Dice alone can produce unstable
    gradients early in training when the prediction is all-background
    (common for a 2-class problem with a small foreground organ like the
    spleen); CE stabilizes early training while Dice drives the metric
    that actually matters for this task.
    """
    return DiceCELoss(to_onehot_y=True, softmax=True)


def get_dice_metric() -> DiceMetric:
    return DiceMetric(include_background=False, reduction="mean", get_not_nans=False)


def get_postprocessing():
    """
    Converts raw 2-channel logits into a one-hot discrete prediction, and
    converts the integer label into one-hot form so DiceMetric can compare
    them directly.
    """
    post_pred = AsDiscrete(argmax=True, to_onehot=config.OUT_CHANNELS)
    post_label = AsDiscrete(to_onehot=config.OUT_CHANNELS)
    return post_pred, post_label


def sliding_window_predict(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """
    Run inference on a full (variable-size) volume using a sliding window
    of size config.PATCH_SIZE -- the model was trained on fixed-size
    patches, so it can't be applied to a full volume in one forward pass.

    overlap=0.5 trades compute for smoother predictions at patch
    boundaries; reduce to 0.25 if inference is too slow on CPU/Colab.
    """
    return sliding_window_inference(
        inputs=image,
        roi_size=config.PATCH_SIZE,
        sw_batch_size=4,
        predictor=model,
        overlap=0.5,
    )


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    return_per_case: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Runs full-volume sliding-window inference + Dice over a dataloader.

    Returns (mean_dice, per_case_dice) where per_case_dice maps a case
    identifier to its Dice score -- needed for failure-case analysis,
    not just an aggregate number.
    """
    model.eval()
    dice_metric = get_dice_metric()
    post_pred, post_label = get_postprocessing()
    per_case: Dict[str, float] = {}

    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        pred = sliding_window_predict(model, image)
        pred_list = [post_pred(p) for p in torch.unbind(pred, dim=0)]
        label_list = [post_label(l) for l in torch.unbind(label, dim=0)]

        dice_metric(y_pred=pred_list, y=label_list)

        if return_per_case:
            case_dice = get_dice_metric()
            case_dice(y_pred=pred_list, y=label_list)
            case_id = _extract_case_id(batch)
            per_case[case_id] = float(case_dice.aggregate().item())
            case_dice.reset()

    mean_dice = float(dice_metric.aggregate().item())
    dice_metric.reset()
    return mean_dice, per_case


def _extract_case_id(batch: dict) -> str:
    """Pull a human-readable case identifier out of a MONAI batch dict."""
    meta = batch.get("image_meta_dict") or batch.get("image").meta
    path = meta.get("filename_or_obj") if isinstance(meta, dict) else meta["filename_or_obj"]
    if isinstance(path, (list, tuple)):
        path = path[0]
    return str(path).split("/")[-1].split("\\")[-1]
