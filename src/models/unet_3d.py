from monai.networks.nets import UNet
from src import config


def build_unet_3d() -> UNet:
    """
    Construct a 3D U-Net via MONAI for binary spleen segmentation
    (background + spleen).

    Architecture: 5-level encoder/decoder, channels doubling 16->256,
    2 residual units per block. This is a deliberately modest-capacity
    network (not MONAI's full 'spleen tutorial' config) chosen to train
    in a reasonable time on free-tier Colab/Kaggle GPUs.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    )
    return model
