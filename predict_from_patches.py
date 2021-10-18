'''
Author: Tamer Abousoud
'''
# Standard modules
import math
import random
from itertools import product

# 3rd party modules
import numpy as np
import tensorflow as tf


# Function to determine no. of patches required when patches overlap
def calculate_npatches(img_dim, patch_size, overlap):
    '''
    Calculate number of patches required when patches overlap
    '''
    n_patches = 1
    unpatched = img_dim - patch_size
    
    while unpatched > 0:
        n_patches += 1
        unpatched -= (patch_size - overlap)
        
    return n_patches

# --------------------------------------------------------------------------- #

''' 
The function `predict_patch` predicts the segmentation for a full-size image 
from a model trained on smaller patches. 
--- --- ---
- Takes the image to segment and trained model as arguments, breaks image down
  into patches.
- Returns patch predictions and reassembles fully segmented image from the 
  predicted patches. 
- Allows patches to overlap and can reconstruct final segmented image from 
  overlapping pixels to remove edge artifacts that can occur when using this 
  method with CNNs.
'''

def predict_patch(img, model, patch_size, overlap=0, n_classes=5, batch_size=4):
    '''
    Divide image into patches, run prediction then reassemble final
    image from square patches of size `patch_size` x `patch_size`.
    ---
    img: full-size image for prediction
    model: U-Net model to use for prediction
    patch_size: length of patch side
    overlap: int; no. of pixels to overlap adjacent patches, used to 
             adjust for edge artifacts when re-composing image.
             Should be an even number.
    n_classes: no. of classes to predict
    '''    
    img_height = img.shape[0]
    img_width = img.shape[1]
    n_channels = img.shape[2]
    
    # verify patch sizes and overlaps are compatible with image
    if patch_size > img_height or patch_size > img_width:
        raise ValueError("`patch_size` cannot be larger than image size")
    if overlap % 2 != 0:
        raise ValueError("`overlap` should be an even integer")
    if overlap >= patch_size // 2:
        raise ValueError("`overlap` should be less than half `patch_size`")
    
    # Determine number of patches needed for width, height, account for overlap
    npatches_vertical = calculate_npatches(img_height, patch_size, overlap)
    npatches_horizontal = calculate_npatches(img_width, patch_size, overlap)

    # make extended img containing integer number of patches accounting for overlap
    extended_height = (npatches_vertical * patch_size) -\
                      (npatches_vertical - 1) * (overlap)
    extended_width = (npatches_horizontal * patch_size) -\
                     (npatches_horizontal - 1) * (overlap)
    ext_img = np.zeros(shape=(extended_height, extended_width, n_channels), 
                       dtype=np.float32)
    
    # fill extended image with mirror reflections of neighbors:
    ext_img[:img_height, :img_width, :] = img
    for i in range(img_height, extended_height):
        ext_img[i, :, :] = ext_img[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_img[:, j, :] = ext_img[:, 2 * img_width - j - 1, :]
    
    # Create array for images to predict from patches
    patches_list = []
    
    # Adjust for overlapping regions
    if overlap > 0:
        step = patch_size - overlap
        adj_height, adj_width = extended_height - step, extended_width - step
    else:
        step = patch_size
        adj_height, adj_width = extended_height, extended_width
    coords = product(range(0, adj_height, step), 
                     range(0, adj_width, step))
    
    for i, j in coords:
        # patches for prediction
        x0 = i
        x1 = x0 + patch_size
        y0 = j
        y1 = y0 + patch_size
        patches_list.append(ext_img[x0:x1, y0:y1, :])
    
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    
    # predictions:
    patches_predict = model.predict(patches_array)

    repatch_out = tf.unstack(patches_predict)
    
    repatch_out = [tf.concat(repatch_out[i - npatches_horizontal:i], axis=1) 
                   for i in range(npatches_horizontal, len(repatch_out) + npatches_horizontal, npatches_horizontal)]
    
    repatch_out = tf.concat(repatch_out, axis=0)
    
    # vertical and horizontal masks to filter adjusted patches
    mask_v = np.ones(repatch_out.shape[1])
    mask_h = np.ones(repatch_out.shape[0])
     
    for p in range(patch_size, repatch_out.shape[1], patch_size):
        mask_v[p - overlap // 2 : p + overlap // 2] = 0
        
    for q in range(patch_size, repatch_out.shape[0], patch_size):
        mask_h[q - overlap // 2 : q + overlap // 2] = 0
    
    prediction = np.compress(mask_v, repatch_out, axis=1)
    prediction = np.compress(mask_h, prediction, axis=0)
    
    return prediction[:img_height, :img_width, :]

