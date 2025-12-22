from monai.networks.nets import UNet
from src import config


def build_unet_3d():
    """
    Construct a simple 3D U-Net via MONAI.
    """
    model = UNet(
        dimensions=3,
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    )
    return model
