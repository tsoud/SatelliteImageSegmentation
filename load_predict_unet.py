#!/usr/bin/env python
# coding: utf-8

'''
Author: Tamer Abousoud
---
Load a trained U-Net model and use it for prediction
'''
# Standard libraries
import math
import os
import random
from itertools import product

# 3rd party libraries
import imageio
import matplotlib.pyplot as plot
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'tk')
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.exposure import adjust_gamma
from skimage.transform import rotate

# Tensorflow
import tensorflow as tf
if tf.__version__ >= '2.0':
    # Avoids issues on some GPUs and prevents hogging all GPU memory
    gpu = tf.config.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], enable=True)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model, model_to_dot

# Local imports
from UNet_model import UNet
from img_utils import normalize, picture_from_mask
from img_generator import load_img_data, augment_img, get_sample_patch,\
     get_imgs_labels

from predict_from_patches import predict_patch
from validation_metrics import prediction_metrics, overall_scores



# --------------------------------------------------------------------------- #

img_dir = './data/mband/'
mask_dir = './data/gt_mband/'

saved_unet_model = './saved_models/latest_model'

model = load_model(saved_unet_model, compile=False)
# model.summary()
# model.compile()
# model.load_weights('saved_models/last_saved_weights.hdf5')
# model.compile(loss='binary_crossentropy')
# model2 = Model().from_config(model.get_config())
# model2.load_weights('./saved_models/last_saved_weights.hdf5')
# m2 = model2.get_weights()[0][0, 0, 0, :]
# m = model.get_weights()[0][0, 0, 0, :]

# np.equal(m, m2)
# np.allclose(m, m2, atol=1e-5)

# Prediction parameters
PATCH_SIZE = 160       # patch size
OVERLAP = 32

sample_idx = '22'  # Don't use 'test.tif'!!
# load a test image to predict
test_img = tiff.imread(f'{img_dir}{sample_idx}.tif')
# load the original mask
actual_mask = tiff.imread(f'{mask_dir}{sample_idx}.tif')

# Original image
tiff.imshow(test_img.copy()[(4,2,1), :, :])

# Predictions
test_img = normalize(test_img)
img_t = test_img.transpose([1,2,0])  # keras uses last dimension for channels by default
predicted_mask = predict_patch(img_t, model, 
                               patch_size=PATCH_SIZE, 
                               overlap=OVERLAP).transpose([2,0,1])  # channels first to plot
y_pict_2 = picture_from_mask(predicted_mask, threshold = 0.5)
tiff.imshow(y_pict_2)



img_list = [img for img in os.listdir(img_dir) if not img.startswith('test')]
# overall_scores(model, img_list, img_dir, mask_dir, PATCH_SIZE, OVERLAP)

model.load_weights('/home/tamer/MSCA/Adv_ML_Summer2021/image_segmentation/saved_models/last_saved_weights.hdf5')

model_scores = overall_scores(model, img_list, img_dir, 
                              mask_dir, PATCH_SIZE, OVERLAP)

scores = np.array(list(model_scores.get('IoU').values())); scores
weights = 1 / scores
weights = weights / weights.sum()
weights = np.round(weights, 2)
weights.sum()
weights


test_idx = 'test'  # Don't use 'test.tif'!!
# load a test image to predict
test_img = tiff.imread(f'{img_dir}{test_idx}.tif')
# tiff.imshow(test_img.copy()[(4,2,1), :, :])
test_img = normalize(test_img)
img_t = test_img.transpose([1,2,0])  # keras uses last dimension for channels by default
test_prediction = predict_patch(img_t, model, 
                                patch_size=PATCH_SIZE, 
                                overlap=OVERLAP).transpose([2,0,1])  # channels first to plot
test_labels = picture_from_mask(test_prediction, threshold=0.5)
tiff.imshow(test_labels)
tiff.imsave('result.tif', (255 * test_prediction).astype('uint8'))
