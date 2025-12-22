from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, EnsureTyped
)
from src import config


def get_train_transforms():
    keys_img = ["image"]
    keys_seg = ["label"]
    return Compose([
        LoadImaged(keys=keys_img + keys_seg),
        EnsureChannelFirstd(keys=keys_img + keys_seg),
        Spacingd(
            keys=keys_img + keys_seg,
            pixdim=config.TARGET_SPACING,
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=keys_img + keys_seg, axcodes="RAS"),
        ScaleIntensityRanged(
            keys=keys_img,
            a_min=config.INTENSITY_CLIP[0],
            a_max=config.INTENSITY_CLIP[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=keys_img + keys_seg, source_key="image"),
        RandCropByPosNegLabeld(
            keys=keys_img + keys_seg,
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
        ),
        RandFlipd(keys=keys_img + keys_seg, prob=0.5, spatial_axis=0),
        RandRotate90d(keys=keys_img + keys_seg, prob=0.5, max_k=3),
        EnsureTyped(keys=keys_img + keys_seg),
    ])


def get_val_transforms():
    keys_img = ["image"]
    keys_seg = ["label"]
    return Compose([
        LoadImaged(keys=keys_img + keys_seg),
        EnsureChannelFirstd(keys=keys_img + keys_seg),
        Spacingd(
            keys=keys_img + keys_seg,
            pixdim=config.TARGET_SPACING,
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=keys_img + keys_seg, axcodes="RAS"),
        ScaleIntensityRanged(
            keys=keys_img,
            a_min=config.INTENSITY_CLIP[0],
            a_max=config.INTENSITY_CLIP[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=keys_img + keys_seg, source_key="image"),
        EnsureTyped(keys=keys_img + keys_seg),
    ])
