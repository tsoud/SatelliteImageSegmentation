'''
Author: Tamer Abousoud
---
Creates a configurable U-Net network using TF Keras functional API
'''

# 3rd party libraries
import tensorflow as tf
if tf.__version__ >= '2.0':
    gpu = tf.config.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], enable=True)
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# --------------------------------------------------------------------------- #


class UNet:
    '''
    ---
    Create a U-Net model for image segmentation.
    --- 
    image_size: Input image size (H, W, n_channels).  
    n_classes: Number of classes to predict.  
    unet_depth: How many levels "deep" to make the network.  
    growth_factor: Factor by which to increase the number of filters
                    in a level when going deeper in the network.
                    Values < 1.0 default to 1.0  
    filter_sets_per_conv_layer: Number of sets of filters in a block of 
                                convolutional layers.  
    filters_per_set_start: Number of filters per set at top level (before
                            applying growth factor).  
    conv_layer_kernel_size: Kernel size for convolutional layers.  
    conv_layer_strides: Kernel strides for convolutional layers.  
    conv_layer_maxpool_size: Pool size for Maxpooling layer after 
                                convolutions.  
    use_batch_norm: Applies batch normalization after Maxpooling.
    padding: Padding to use (`'same'` or `'valid'`).  
    activation: Convolutional layer activation function.  
    output_kernel_size: Kernel size for output layer.  
    use_upconv: Use `Conv2DTranspose` instead of upsampling for ascending
                branch of U-Net.  
    upconv_activation: Apply an activation function in `Conv2DTranspose` 
                        layers.  
    output_activation: Activation for output layer.  
    '''
    def __init__(self, image_size:tuple, n_classes:int, 
                 unet_depth:int, growth_factor:float, 
                 filter_sets_per_conv_layer:int, filters_per_set_start:int, 
                 conv_layer_kernel_size:tuple, conv_layer_strides=(1, 1), 
                 conv_layer_maxpool_size=(2, 2), use_batch_norm=False, 
                 padding='same', activation='relu', output_kernel_size=(1, 1), 
                 use_upconv=True, upconv_activation=None, 
                 output_activation='sigmoid'):
        
        self.depth = unet_depth
        self.growth = max(growth_factor, 1.0)
        self.n_filter_sets = filter_sets_per_conv_layer
        self.n_filters_start = filters_per_set_start
        self._n_filters = filters_per_set_start
        self.kernel_size = conv_layer_kernel_size
        self.strides = conv_layer_strides
        self.pool_size = conv_layer_maxpool_size
        self.batch_norm = use_batch_norm
        self.padding = padding
        self.activation = activation
        self.upconv = use_upconv
        self.up_activation = upconv_activation
        # Inputs/Outputs
        self.input_layer = layers.Input((image_size))
        self.output_layer = layers.Conv2D(n_classes, output_kernel_size, 
                                          activation=output_activation)
        # Track last convolutional layer in each level
        self._conv_last = []

    def network(self):
        '''
        Creates the model of the network based on the given parameters
        and input/output specification.
        '''
        x = self.input_layer

        # Build descending branch of network
        for i in range(self.depth):
            # Add convolutional layers
            for j in range(self.n_filter_sets):
                x = layers.Conv2D(filters=self._n_filters, 
                                  kernel_size=self.kernel_size, 
                                  strides=self.strides, 
                                  padding=self.padding)(x)
                if (j == self.n_filter_sets - 1) and\
                    (i != self.depth - 1):  # track last conv layer
                    self._conv_last.append(x)
            # Add MaxPool layer
            if self.pool_size and (i != self.depth - 1):
                x = layers.MaxPool2D(self.pool_size, padding=self.padding)(x)
            # Add batch normalization if applicable
            if self.batch_norm and (i != self.depth - 1):
                x = layers.BatchNormalization()(x)
            # Increase filters for next level
            if i < (self.depth - 1):
                self._n_filters *= self.growth

        # Build ascending branch of network
        for i in range(self.depth - 1):
            self._n_filters //= self.growth
            # Add upconv or upsampling layers
            if self.upconv:
                y = layers.Conv2DTranspose(filters=self._n_filters, 
                                           kernel_size=self.pool_size, 
                                           strides=self.pool_size, 
                                           padding=self.padding)(x)
            else:
                y = layers.UpSampling2D(self.pool_size)(x)
            # Concatenate with adjacent convolutions from "down" branch
            x = layers.concatenate([y, self._conv_last.pop()])
            # Add convolutional layers
            for j in range(self.n_filter_sets):
                x = layers.Conv2D(filters=self._n_filters, 
                                  kernel_size=self.kernel_size, 
                                  strides=self.strides, 
                                  padding=self.padding)(x)

        return Model(self.input_layer, self.output_layer(x))

