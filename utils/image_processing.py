

import numpy as np
from torchvision.transforms import transforms as tr

def inv_normalize(image):
    inv_normal = tr.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    return inv_normal(image).data

def uw_inv_normalize(image):
    inv_normal = tr.Normalize(
        mean=[-0.13553666/0.04927989, -0.41034216/0.10722694, -0.34636855/0.10722694],
        std=[1/0.04927989, 1/0.10722694, 1/0.10722694]
    )
    return inv_normal(image).data

def normalize_result(value, vmin=None, vmax=None):
    
    try:
        value = value.cpu().numpy()[0, :, :]
    except:
        pass

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.
    return np.expand_dims(value, 0)