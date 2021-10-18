#!/usr/bin/env python
# coding: utf-8

'''
Author: Tamer Abousoud
---
Train the U-Net model for satellite image segmentation
'''

# Standard libraries
import math
import os
import threading
from datetime import datetime
from zipfile import ZipFile

# 3rd party libraries
import imageio
import numpy as np
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
from tensorflow.keras import backend as K
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, \
    TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model, model_to_dot, GeneratorEnqueuer

# Local imports
from UNet_model import UNet
from img_utils import normalize, picture_from_mask
from img_generator import load_img_data, augment_img, get_sample_patch,\
     get_imgs_labels, generate_imgs_labels

# --------------------------------------------------------------------------- #

# Choose layers to freeze when fine-tuning
def freeze_layers(model, freeze_up_to_layer:str):
    ''' 
    Freezes all model layers before the given layer.
    '''
    layer_names = [layer.name for layer in model.layers]
    freeze_up_to = layer_names.index(freeze_up_to_layer)

    for layer in model.layers[:freeze_up_to]:
        layer.trainable = False


# Define the loss function
def weighted_bin_crossentropy(class_weights:tf.Tensor):
    @tf.function
    def loss(y_true, y_pred):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return K.sum(class_loglosses * class_weights)
    return loss


# Learning rate scheduler
def scheduler(epoch, lr):
    limits = [40, 80, 120]
    # limits = [60, 120, 160]
    # step_size = [0.001, 0.0003, 0.0001]
    step_size = [0.0001, 0.00003, 1e-5]
    if epoch in range(limits[0]):
        return step_size[0]
    elif epoch in range(limits[0], limits[1]):
        return step_size[1]
    elif epoch in range(limits[1], limits[2]):
        return step_size[2]
    else:
        return max(lr * tf.math.exp(-0.05), 1e-6)


# --------------------------------------------------------------------------- #

if __name__=='__main__':

    continue_training = str(input(f"Continue training model "\
        "(Y for YES, *any key* for NO)?")).lower()
    continue_training = True if continue_training == 'y' else False
    if continue_training:
        model_path = str(input("Enter saved model path: "))
        tune_model = str(input(f"Tune model "\
            "(Y for YES, *any key* for NO)?")).lower()
        tune_model = True if tune_model == 'y' else False

    # with ZipFile('./data.zip') as zf:
    #     zf.extractall()
    # img_dir = './data/mband/'
    # mask_dir = './data/gt_mband/'
    
    img_dir = './data/mband/'
    mask_dir = './data/gt_mband/'

    images, labels, shapes = load_img_data(img_dir=img_dir, mask_dir=mask_dir)

    if continue_training:
        unet_model = load_model(model_path, compile=False)
        # Make sure to use the patch size of existing model
        PATCH_SIZE = unet_model.get_config().get('layers')[0]\
            .get('config').get('batch_input_shape')[1]
        if tune_model:
            # Freeze all layers in descending branch
            freeze_layers(unet_model, freeze_up_to_layer='conv2d_transpose')
    else:
        # Model parameters
        N_CLASSES = 5                    # no. of classes to predict
        CHANNELS = 8                     # no. of image channels
        N_LEVELS = 5                     # depth of U-Net
        TOP_LEVEL_FLTRS = 36             # top layer filters
        N_FILTER_SETS = 2                # sets of filters in conv block
        GROWTH_FACTOR = 2                # growth factor
        PATCH_SIZE = 160                 # patch size
        KERNEL_SIZE = (3, 3)             # conv kernel size
        BATCH_NORM = False

        # Configure the model
        unet = UNet(image_size=(PATCH_SIZE, PATCH_SIZE, CHANNELS), 
                    n_classes=N_CLASSES, 
                    unet_depth=N_LEVELS, 
                    growth_factor=GROWTH_FACTOR, 
                    filter_sets_per_conv_layer=N_FILTER_SETS, 
                    filters_per_set_start=TOP_LEVEL_FLTRS, 
                    conv_layer_kernel_size=KERNEL_SIZE, 
                    use_batch_norm=BATCH_NORM)

        # Assemble the U-Net NN
        unet_model = unet.network()

    # Training parameters:
    EPOCHS = 200
    BATCH_SIZE = tf.constant(24)
    val_batch = max([0.25 * BATCH_SIZE.numpy(), 1.0])
    VAL_BATCH_SIZE = tf.constant(val_batch)
    N_STEPS = 150                                            # steps per epoch
    VAL_STEPS = int(np.round(0.25 * N_STEPS, decimals=0))    # validation steps
    LR = 0.001                                               # learning rate
    if tune_model:
        LR = LR * 0.1

    # Using `GeneratorEnqueuer` improves GPU  utilization when using
    # a generator dataset; recommended for faster performance.
    # Increasing `N_WORKERS` and `MAX_Q_SZ` improves performance but
    # also uses more system memory.
    # Set to True to use `GeneratorEnqueuer`:
    USE_GENQ = False               
    N_WORKERS = 4
    MAX_Q_SZ = 500

    # Class weights for loss calculation
    # class_weights = np.array([0.2, 0.3, 0.1, 0.1, 0.3])
    class_weights = tf.constant([0.2, 0.3, 0.1, 0.1, 0.3], dtype=tf.float32)
    # class_weights = tf.constant([0.15, 0.42, 0.16, 0.09, 0.18], dtype=tf.float32)

    # Select optimizer
    # scheduled_lr = optimizers.schedules.PiecewiseConstantDecay(
    #                                     boundaries=[40, 80, 120], 
    #                                     values=[0.001, 0.0003, 0.0001, 1e-5])
    # optimizer = optimizers.Adam(clipnorm=1.0, clipvalue=0.5)
    optimizer = optimizers.SGD(learning_rate=LR, momentum=0.95, 
                               clipnorm=1.0, clipvalue=0.5)
    # optimizer = optimizers.Adam(0.00003, clipnorm=1.0, clipvalue=0.5)

    # Set up directory for saving checkpoint weights
    save_dir = './saved_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_weights_path = save_dir + 'last_saved_weights.hdf5'

    # Checkpoints
    model_checkpoint = ModelCheckpoint(save_weights_path, 
                                       monitor='val_loss', 
                                       save_best_only=True, 
                                       mode='min'
                                       )

    # Define early stopping criteria
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=15, 
                                   restore_best_weights=True)
                                   
    # Learning rate schedule
    scheduler = LearningRateScheduler(scheduler)

    # Learning rate reduction when model stops improving
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_lr=1e-6, patience=8)

    # Tensorboard callbacks
    tensorboard = TensorBoard(log_dir='tb_logs', write_graph=False, write_images=True)

    if USE_GENQ:
        gen_enqueuer = GeneratorEnqueuer(generate_imgs_labels(
                                         tf.constant(PATCH_SIZE), 
                                         images, labels, shapes, 
                                         BATCH_SIZE), 
                                        #  use_multiprocessing=True
                                         )
        gen_enqueuer.start(workers=N_WORKERS, max_queue_size=MAX_Q_SZ)
        data_generator = gen_enqueuer.get()

    else:
        data_generator = generate_imgs_labels(tf.constant(PATCH_SIZE), 
                                              images, labels, shapes,  
                                              BATCH_SIZE)


    # TRAIN THE MODEL
    # ---------------
    LOSS_FN = weighted_bin_crossentropy(class_weights)
    # LOSS_FN = 'binary_crossentropy'
    unet_model.compile(optimizer=optimizer, 
                       loss=LOSS_FN)

    # Train
    unet_model.fit(data_generator, 
                   epochs=EPOCHS, steps_per_epoch=N_STEPS, 
                   validation_data=data_generator, 
                   validation_steps=VAL_STEPS, 
                   callbacks=[
                              model_checkpoint, 
                              lr_reduction, 
                              early_stopping, 
                              tensorboard
                             ], 
                   verbose=1)

    # Save the full model when done
    save_model_path = save_dir + 'latest_model'
    print(f"\nFinished training. Saving trained model to "\
        f"{os.path.abspath(save_model_path)}")
    unet_model.save(save_model_path)
    # model_json = unet_model.to_json()
    # with open(save_model_path + '.json', 'w') as json_file:
    #     json_file.write(model_json)
    # unet_model.save_weights(save_model_path + 'h5')
    print("\nModel saved.")

    # if USE_GENQ:
    #     # Try to stop the generator if it doesn't exit.
    #     if gen_enqueuer.is_running():
    #         print(f"\nShutting down GeneratorEnqueuer..."\
    #             "\nUse 'Ctrl-C' to exit if this takes too long.")
    #     gen_enqueuer.stop()

