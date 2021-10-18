'''
Some utilities for working with images.
---
code by Yuri Balasanov, Mihail Tselishchev; iLykei 2018-2020
'''

import numpy as np
import tensorflow as tf


def normalize(img):
    min = img.min()
    max = img.max()
    return 2.0 * (img - min) / (max - min) - 1.0


def denormalize(img, orig_img):
    min = orig_img.min()
    max = orig_img.max()
    denormalized_img = ((img + 1.0) * (max - min)) / 2 + min
    return (np.round(denormalized_img)).astype(int)


def picture_from_mask(mask, threshold=0):
    colors = {
        0: [150, 150, 150],  # Buildings (grey)
        1: [223, 194, 125],  # Roads & Tracks (light orange)
        2: [27, 120, 55],    # Trees (green)
        3: [166, 219, 160],  # Crops (greyish-green)
        4: [116, 173, 209]   # Water (blue)
    }
    z_order = {
        1: 3,
        2: 4,
        3: 0,
        4: 1,
        5: 2
    }
    pict = 255*np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8)
    for i in range(1, 6):
        cl = z_order[i]
        for ch in range(3):
            pict[ch,:,:][mask[cl,:,:] > threshold] = colors[cl][ch]
    return pict
