import random

import numpy as np
from PIL import Image


def train_augment(image, mask, crop_size, scale_range=(0.5, 2.0), flip_prob=0.5):
    """SegFormer / AttnFD-style random scale, flip, and crop."""
    crop_h, crop_w = crop_size
    w, h = image.size
    scale = random.uniform(scale_range[0], scale_range[1])
    new_w, new_h = int(w * scale), int(h * scale)
    new_w = max(new_w, crop_w)
    new_h = max(new_h, crop_h)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    mask = mask.resize((new_w, new_h), Image.NEAREST)

    if random.random() < flip_prob:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    x0 = random.randint(0, new_w - crop_w)
    y0 = random.randint(0, new_h - crop_h)
    image = image.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    mask = mask.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    return image, mask


def to_tensor(image, mask, mean, std):
    image = np.array(image, dtype=np.float32) / 255.0
    mask = np.array(mask, dtype=np.int64)
    image = (image - mean) / std
    return image.transpose(2, 0, 1), mask
