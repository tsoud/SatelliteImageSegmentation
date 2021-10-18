'''
Author: Tamer Abousoud
---
Generate sample image patches for training
'''

# Standard libraries
import os
import random
import threading

# Third party libraries
import tensorflow as tf
if tf.__version__ >= '2.0':
    gpu = tf.config.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], enable=True)
import numpy as np
import tifffile as tiff
from skimage.transform import rotate
from skimage.exposure import adjust_gamma

# User libraries
from img_utils import normalize


def load_img_data(img_dir:str, mask_dir:str, output_channels='all', pad_images=True):
    '''
    Load images from disk and convert to arrays for processing.
    ---
    img_dir: directory of tiff images
    mask_dir: directory of ground truth image masks
    output_channels: Output channels (labels) for segmentation masks
                     'all' -> include all mask channels
                     int, tuple -> channel or channels (if tuple) to include
    pad_images: If images have different sizes, pad all images so all elements in
                returned array have the same 2D shape corresponding to 
                (largest height * largest width)
    ---
    returns -> tuple: image array with shape `n_images * H * W * channels`, 
                      mask array with shape `n_masks * H * W * channels`, 
                      shape array with original image 2D shapes
    If padded H, W == padded H, padded W
    '''
    
    # NOTE: Remember to skip test image(s) in image directory
    img_ids = list(set(os.listdir(img_dir)) & set(os.listdir(mask_dir)))
    img_ids.sort()

    images = [img_dir + im for im in img_ids]
    masks = [mask_dir + im for im in img_ids]

    data = []

    for img, mask in zip(images, masks):
        # Read and process images
        img = tiff.imread(img)
        img = normalize(img)
        img = img.transpose([1, 2, 0])
        mask = tiff.imread(mask) / 255
        mask = mask.transpose([1, 2, 0])

        if output_channels != 'all':
            mask = mask[:, :, output_channels]
            if isinstance(output_channels, int):
                # change from 2D to 3D array if 3rd dim is 1
                mask = np.expand_dims(mask, axis=2)

        data.append((img, mask))

    if pad_images:
        shapes = [img[0].shape[:2] for img in data]
        heights, widths = zip(*shapes)
        padded_H, padded_W = max(heights), max(widths)
        data = [(tf.image.pad_to_bounding_box(img, 0, 0, padded_H, padded_W), 
                 tf.image.pad_to_bounding_box(mask, 0, 0, padded_H, padded_W)) for\
                 (img, mask) in data]

    images, masks = zip(*data)
    return tf.convert_to_tensor(images, dtype=tf.float32),\
           tf.convert_to_tensor(masks, dtype=tf.float32),\
           tf.convert_to_tensor(shapes, dtype=tf.int32)


@tf.function
def get_sample_patch(patch_size:tf.Tensor, 
                     images:tf.Tensor, 
                     masks:tf.Tensor, 
                     shapes:tf.Tensor):
    '''
    ---
    Create samples from the existing images and masks for training.
    ---
    patch_size: Sample patch size, must be input as numpy array
    images, masks: image and mask arrays from `load_img_data()`
    shapes: array of original image shapes
    exposure_step: Positive value <= 1.0 for applying color exposure augmentation.
    ---
    Returns: A random patch from an image and its corresponding mask
    '''
    random_id = tf.random.uniform((), minval=0, maxval=images.shape[0], 
                                  dtype=tf.int32)
    image, mask, shape = images[random_id], masks[random_id], shapes[random_id]
    int_type = shape.dtype  # ensure int types are consistent for calculations
    patch_size = tf.cast(patch_size, dtype=int_type)
    # Select random x, y coordinates for the sample patch
    x = tf.random.uniform((), minval=0, maxval=(shape[0] - patch_size), 
                          dtype=int_type)
    y = tf.random.uniform((), minval=0, maxval=(shape[1] - patch_size), 
                          dtype=int_type)
    # fetch same patch from image and mask
    img_patch = image[x:(x + patch_size), y:(y + patch_size)]
    mask_patch = mask[x:(x + patch_size), y:(y + patch_size)]
    # return augmented image and mask patch
    return img_patch, mask_patch


@tf.function
def augment_img(image:tf.Tensor, mask:tf.Tensor):
    '''
    Apply random augmentations (flips and rotations) to an image and mask.
    '''
    # Random values for transformations
    rot = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    flip_v = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
    flip_h = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)

    image = tf.image.rot90(image, k=rot)
    mask = tf.image.rot90(mask, k=rot)
    if tf.cast(flip_v, tf.bool):
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    if tf.cast(flip_h, tf.bool):
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    return image, mask


@tf.function
def get_imgs_labels(patch_size:tf.Tensor, 
                    images:tf.Tensor, 
                    masks:tf.Tensor, 
                    shapes:tf.Tensor, 
                    batch_size:tf.Tensor):
    '''
    Fetch a number of patches from the training images
    returning augmented image and corresponding mask as tensors.
    ---
    patch_size: Sample patch size
    images, masks: Array of images and masks from `load_img_data()`
    shapes: Array of original image shapes before padding
    batch_size: Number of patches to generate.
    '''
    img_channels = tf.constant(images[0].shape[-1])
    mask_channels = tf.constant(masks[0].shape[-1])

    img_patches = tf.TensorArray(dtype=tf.float32, size=batch_size)
    label_patches = tf.TensorArray(dtype=tf.float32, size=batch_size)

    for idx in tf.range(batch_size):
        img_patch, label_patch = get_sample_patch(patch_size=patch_size, 
                                                  images=images, 
                                                  masks=masks, 
                                                  shapes=shapes)
        img_patch, label_patch = augment_img(img_patch, label_patch)
        img_patches = img_patches.write(idx, img_patch)
        label_patches = label_patches.write(idx, label_patch)

    return img_patches.stack(), label_patches.stack()


# Turn `get_imgs_labels` into a thread-safe generator so it can be used with
# keras functions like `GeneratorEnqueuer` to speed up training.
class threadsafe_iterator:
    ''' 
    ---
    Makes a thread-safe iterator
    ---
    it: a function that can be used as a generator
    '''
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    def __iter__(self):
        return self
    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(func):
    ''' 
    Wrapper to apply `threadsafe_iterator` to a generator
    '''
    def make_threadsafe(*args, **kwargs):
        return threadsafe_iterator(func(*args, **kwargs))
    return make_threadsafe

@threadsafe_generator
def generate_imgs_labels(patch_size:tf.Tensor, 
                         images:tf.Tensor, 
                         masks:tf.Tensor, 
                         shapes:tf.Tensor, 
                         batch_size:tf.Tensor):
    while True:
        yield get_imgs_labels(patch_size, images, masks, shapes, batch_size)