"""
####################################################################################
Fully Convolution Network (FCN) Image Segmentation Models with Tensorflow and Keras.
####################################################################################
# Supported FCN Models :
- FCN8
- FCN16
- FCN32

Pre-trained models eligible to be model encoder
- DenseNet121, DenseNet169
- EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
- EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
- MobileNet, MobileNetV2
- ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2
- VGG16, VGG19
"""

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine import training
from tensorflow.python.lib.io import file_io
import warnings

from ..lib.encoders import encoder_models
from ..lib.utils import get_encoder_model
from ..lib.utils import get_encoder_model_output_layer
from ..lib.utils import get_skip_connection_layers



################################################################################
# Default Convolution Block Function
################################################################################
def convolution_block(block_input,
                      num_filters=256,
                      kernel_size=3,
                      dilation_rate=1,
                      padding="same",
                      use_bias=False,
                      use_batchnorm=True,
                      activation='relu'):
    """
    Instantiates default convolution block.
    """
    if activation == None:
        activation = 'linear'

    x = Conv2D(num_filters,
               kernel_size=kernel_size,
               padding="same",
               use_bias=use_bias,
               kernel_initializer=tf.keras.initializers.HeNormal()
              )(block_input)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x



################################################################################
# FCN Encoder
################################################################################
def fcn_encoder(encoder_type='Default',
                input_tensor=None,
                encoder_weights=None,
                encoder_freeze=False,
                encoder_filters=[32, 64, 128, 256, 512]):

    """
    Instantiates the encoder model for the Fully Convolutional Network (FCN)
    architecture.

    Args:
        encoder_type: type of model to build upon. One of 'Default',
                      'DenseNet121', 'DenseNet169' 'EfficientNetB0',
                      'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                      'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6',
                      'EfficientNetB7', 'MobileNet', 'MobileNetV2',
                      'ResNet50', 'ResNet101', 'ResNet152',
                      'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
                      'VGG16', 'VGG19'. Default encoder type is  'Default'.
        input_tensor: a tensorflow tensor input of a tuple containg image width,
                      height and channels respectively.
        encoder_weights: One of `None` (random initialization), `imagenet`,
                         or the path to the weights file to be loaded.
                         Default to None.
        encoder_freeze: Boolean Specifying whether to train encoder parameters
                        or not. Default to False.
        encoder_filters: a list or tuple containing the number of filters to
                         use for each encoder block. Default to
                         [32, 64, 128, 256, 512].
    Returns:
        encoder_model: keras model
        encoder_model_output: Encoder model output.
        skip_connection_layers: Skip connection layers/encoder
                                block outputs before pooling.
    """
    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. encoder_type
    if encoder_type.lower() != 'default':
        if not encoder_type in encoder_models:
            raise ValueError("The `encoder_type` argument is not not properly defined. "
                        "Kindly use one of the following encoder names: 'Default', "
                        "'DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', "
                        "'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', "
                        "'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', "
                        "'EfficientNetB67', 'MobileNet', 'MobileNetV2', 'ResNet50', "
                        "'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', "
                        "'ResNet152V2', 'VGG16', 'VGG19'")

    # 2. encoder_weights
    if encoder_type.lower() == 'default':
        if not (encoder_weights in {None} or file_io.file_exists_v2(encoder_weights)):
            warnings.warn('Ensure the `encoder_weights` argument is either '
                          '`None` (random initialization), '
                          'or the path to the weights file to be loaded. ')
    else:
        if not (encoder_weights in {'imagenet', None} or file_io.file_exists_v2(encoder_weights)):
            warnings.warn('The `encoder_weights` argument should be either '
                          '`None` (random initialization), `imagenet` '
                          '(pre-training on ImageNet), or the path to the '
                          'weights file to be loaded.')

    # 3. encoder_freeze
    if not isinstance(encoder_freeze, bool):
        raise ValueError("The `encoder_freeze` argument should either be True or False.")

    # 5. fcn_encoder
    if not (isinstance(encoder_filters, tuple) or isinstance(encoder_filters, list)):
        raise ValueError('The `encoder_filters` argument should be a list of tuple.')
    elif len(encoder_filters) <= 0:
        raise ValueError('The `encoder_filters` argument cannot be an empty list.')
    
    #--------------------------------------------------------------------------#
    # Build the encoding blocks from the arguments specified
    #--------------------------------------------------------------------------#
    # 1. Default encoding blocks
    #--------------------------------------------------------------------------#
    num_blocks = 5
    if encoder_type.lower() == 'default':
        x = input_tensor
        kernel_size = 3
        pool_size = 2
        skip_connection_layers = []

        # Design the model
        for filter_id in range(num_blocks):
            num_filters = encoder_filters[filter_id]
            x = convolution_block(x,
                                  num_filters=num_filters,
                                  kernel_size=kernel_size,
                                  dilation_rate=1,
                                  padding="same",
                                  use_bias=False,
                                  use_batchnorm=True,
                                  activation='relu')

            x = convolution_block(x,
                                  num_filters=num_filters,
                                  kernel_size=kernel_size,
                                  dilation_rate=1,
                                  padding="same",
                                  use_bias=False,
                                  use_batchnorm=True,
                                  activation='relu')

            # save block output layer before pooling
            skip_connection_layers.append(x)

            x = MaxPooling2D((pool_size, pool_size))(x)

        encoder_model_output = x

        # Create model
        encoder_model = Model(input_tensor, encoder_model_output)

    #--------------------------------------------------------------------------#
    # 2. Pretrained Model encoding blocks
    #--------------------------------------------------------------------------#
    else:
        encoder_model = get_encoder_model(encoder_type,
                                          input_tensor,
                                          encoder_weights)
        skip_connection_layers = get_skip_connection_layers(encoder_type,
                                                            encoder_model)
        encoder_model_output = get_encoder_model_output_layer(encoder_type,
                                                              encoder_model,
                                                              num_blocks)

    # Make the model parameters trainable or non trainable
    encoder_model.trainable = not(encoder_freeze)

    return encoder_model, encoder_model_output, skip_connection_layers



################################################################################
# FCN 8, 16 and 32 Encoder
################################################################################
#------------------------------------------------------------------------------#
# 1. FC32 Decoder
#------------------------------------------------------------------------------#
def fcn_decoder(num_classes,
                model_type,
                decoder_input,
                skip_connection_layers,
                decoder_type='upsampling',
                output_activation=None):

    """
    Instantiates the decoder function for the Fully Convolutional Network (FCN)
    architecture.

    Args:
        num_classes: number of the segmentation classes.
        model_type: one of FCN32, FCN16 or FCN8
        decoder_input: decoder inpit or encoder model output.
        skip_connection_layers: skip connection layers
        decoder_type: (optional) one of 'transpose' 
                      (to use Conv2DTanspose operation for
                      deconvolution operation) or 'upsampling' 
                      (to use UpSampling2D operation for 
                      deconvolution operation). Default to upsampling.
        output_activation: (optional) activation for output layer.
                            Default to either 'sigmoid' or 'softmax' based on
                            the value of the 'num_classes' argument.
    Returns:
        output: Decoder output.
    """
    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. number of classes specified
    if num_classes < 2:
        raise ValueError("The `num_classes` argument cannot be less than 2.")
    elif not(isinstance(num_classes, int)):
        raise ValueError("The `num_classes` argument can only be an integer.")

    # 2. model_type
    if not model_type.lower() in {'fcn32','fcn-32','fcn16',
                                  'fcn-16','fcn8','fcn-8'}:
        raise ValueError("The `model_type` should be one of "
                         "'FCN32','FCN16','FCN8'.")

    # 3. decoder_type
    if not decoder_type.lower() in {'transpose', 'upsampling'}:
        raise ValueError("The `decoder_type` should be one of 'transpose' "
                         "or 'upsampling'.")

    # 4. output activation
    if output_activation == None:
        if num_classes == 2:
            output_activation = 'sigmoid'
        else:
            output_activation = 'softmax'

    r1, r2 = skip_connection_layers[-2:]

    #--------------------------------------------------------------------------#
    # Build the decoding blocks from the arguments specified
    #--------------------------------------------------------------------------#
    #--------------------------------------------------------------------------#
    # FCN-32 Decoder
    #--------------------------------------------------------------------------#
    if model_type.lower() in {'fcn32', 'fcn-32'}:
        if decoder_type.lower() == 'transpose':
            x = Conv2D(num_classes,  (1, 1), kernel_initializer='he_normal',
                       name="decoder_conv")(decoder_input)
            x = Conv2DTranspose(num_classes, kernel_size=(64, 64),
                                strides=(32, 32), padding='same',
                                name='decoder_conv2dtranspose')(x)
            x = Activation(output_activation, name='decoder_activation')(x)

        elif decoder_type.lower() == 'upsampling':
            x = Conv2D(num_classes,  (1, 1), kernel_initializer='he_normal',
                       padding='same', name="decoder_conv_1")(decoder_input)
            x = UpSampling2D(size=(32, 32), interpolation='bilinear',
                             name='decoder_upsampling2d')(x)
            x = Conv2D(num_classes,  (64, 64), kernel_initializer='he_normal',
                       padding='same', name="decoder_conv_2")(x)
            x = Activation(output_activation, name='decoder_activation')(x)
    
    #--------------------------------------------------------------------------#
    # FCN-16 Decoder
    #--------------------------------------------------------------------------#
    elif model_type.lower() in {'fcn16', 'fcn-16'}:
        if decoder_type.lower() == 'transpose':
            x = Conv2D(num_classes,  (1, 1), kernel_initializer='he_normal',
                       name="decoder_conv_1")(decoder_input)
            x = Conv2DTranspose(num_classes, kernel_size=(3,3),
                                strides=(2,2), padding='same',
                                name='decoder_conv2dtranspose_1')(x)
            y = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), 
                       padding='same', name='decoder_conv_2')(r2)
            x = Add(name='decoder_skip_connection_1')([x, y])

            x = Conv2D(num_classes,  (1, 1), kernel_initializer='he_normal',
                       name="decoder_conv_3")(x)
            x = Conv2DTranspose(num_classes, kernel_size=(32,32), strides=(16,16), 
                                padding='same', name='decoder_conv2dtranspose_2')(x)
            x = Activation(output_activation, name='outut_activation')(x)

        elif decoder_type.lower() == 'upsampling':
            x = Conv2D(num_classes,  (1, 1), kernel_initializer='he_normal',
                       name="decoder_conv_1")(decoder_input)
            x = UpSampling2D(size=(2, 2), interpolation='bilinear',
                             name='decoder_upsampling2d_1')(x)
            x = Conv2D(num_classes,  (3, 3), kernel_initializer='he_normal',
                       padding='same', name="decoder_conv_2")(x)
            y = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), 
                       padding='same', name='decoder_conv_3')(r2)
            x = Add(name='decoder_skip_connection_1')([x, y])
            x = Conv2D(num_classes,  (1, 1), kernel_initializer='he_normal', 
                       name="decoder_conv_4")(x)

            x = UpSampling2D(size=(16, 16), interpolation='bilinear',
                             name='decoder_upsampling2d_2')(x)
            x = Conv2D(num_classes,  (32, 32), kernel_initializer='he_normal',
                       padding='same', name="decoder_conv_5")(x)
            x = Activation(output_activation, name='outut_activation')(x)
    
    #--------------------------------------------------------------------------#
    # FCN-8 Decoder
    #--------------------------------------------------------------------------#
    elif model_type.lower() in {'fcn8', 'fcn-8'}:
        if decoder_type.lower() == 'transpose':
            x = Conv2D(num_classes,  (1, 1), kernel_initializer='he_normal',
                       name="decoder_conv_1")(decoder_input)
            x = Conv2DTranspose(num_classes, kernel_size=(3,3),
                                strides=(2,2), padding='same',
                                name='decoder_conv2dtranspose_1')(x)
            y = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1),
                       padding='same', name='decoder_conv_2')(r2)
            x = Add(name='decoder_skip_connection_1')([x, y])
            x = Conv2D(num_classes,  (1, 1), kernel_initializer='he_normal',
                       name="decoder_conv_3")(x)
            x = Conv2DTranspose(num_classes, kernel_size=(3,3),
                                strides=(2,2), padding='same',
                                name='decoder_conv2dtranspose_2')(x)
            z = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1),
                       padding='same', name='decoder_conv_4')(r1)
            x = Add(name='decoder_skip_connection_2')([x, z])
            x = Conv2DTranspose(num_classes, kernel_size=(16,16),
                                strides=(8,8), padding='same',
                                name='decoder_conv2dtranspose_3')(x)
            x = Activation(output_activation, name='output_activation')(x)

        elif decoder_type.lower() == 'upsampling':
            x = Conv2D(num_classes,  (1, 1), kernel_initializer='he_normal',
                       name="decoder_conv_1")(decoder_input)
            x = UpSampling2D(size=(2, 2), interpolation='bilinear',
                             name='decoder_upsampling2d_1')(x)
            x = Conv2D(num_classes,  (3, 3), kernel_initializer='he_normal',
                       padding='same', name="decoder_conv_2")(x)
            y = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1),
                       padding='same', name='decoder_conv_3')(r2)
            x = Add(name='decoder_skip_connection_1')([x, y])
            x = Conv2D(num_classes,  (1, 1), kernel_initializer='he_normal',
                       name="decoder_conv_4")(x)
            x = UpSampling2D(size=(2, 2), interpolation='bilinear',
                             name='decoder_upsampling2d_2')(x)
            x = Conv2D(num_classes,  (3, 3), kernel_initializer='he_normal',
                       padding='same', name="decoder_conv_5")(x)
            z = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1),
                       padding='same', name='decoder_conv_6')(r1)
            x = Add(name='decoder_skip_connection_2')([x, z])
            x = UpSampling2D(size=(8, 8), interpolation='bilinear',
                             name='decoder_upsampling2d_3')(x)
            x = Conv2D(num_classes,  (16, 16), kernel_initializer='he_normal',
                       padding='same', name="decoder_conv_7")(x)
            x = Activation(output_activation, name='output_activation')(x)
    
    # Output layer
    output = x

    return output

################################################################################
# FCN Segmentation Model
################################################################################
#------------------------------------------------------------------------------#
# 1. FCN Model
#------------------------------------------------------------------------------#
def fcn(num_classes,
        model_type,
        encoder_type='Default',
        input_shape=(224,224,3),
        model_weights=None,
        encoder_filters=[32, 64, 128, 256, 512],        
        encoder_weights=None,
        encoder_freeze=False,
        decoder_type='upsampling',
        output_activation=None):
    
    '''
    Fully Convolution Network (FCN) Model for Semantic Segmentation.
        Args:
            num_classes: number of the segmentation classes.
            model_tyoe: one of FCN model type i.e. 'FCN32', 'FCN16' or 'FCN8'.
            encoder_type: type of model to build upon. One of 'Default',
                          'DenseNet121', 'DenseNet169' 'EfficientNetB0',
                          'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                          'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6',
                          'EfficientNetB7', 'MobileNet', 'MobileNetV2',
                          'ResNet50', 'ResNet101', 'ResNet152',
                          'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
                          'VGG16', 'VGG19'. Default to  'Default'.
            input_shape: a tuple containing image height, width and channel
                         respectively. Default to (224,224,3).
            model_weights: (optional) Link to pre-trained weights.
                           Default to None.
            encoder_filters: (optional) number of filters to use for the encoder 
                             model blocks. Only useful with 'Default' encoder_type
                             input.
            encoder_weights: (optional) pre-trained weights for encoder function.
                             one of None (random initialization),
                             'imagenet' (pre-training on ImageNet),
                             or the path to the weights file to be loaded.
                             Default to None.
            encoder_freeze: (optional) boolean to specify whether to train encoder
                            model parameters or not. Default to False.
            output_activation: (optional) Activation for output layer.
                                Default to 'sigmoid' for 2 segmentation classes
                                or 'softmax' for more than 2 segmentation classes.
        Returns:
            (keras FCN Model)
    '''
    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. num_classes - check fcn32_decoder functon
    # 2. encoder_type - check fcn_encoder functon
    # 3. input_shape - check fcn_encoder functon
    # 4. model_weights
    if not (model_weights in {None} or file_io.file_exists_v2(model_weights)):
        warnings.warn('The `model_weights` argument should either be '
                      '`None` (random initialization), '
                      'or the path to the weights file to be loaded.')

    # 5. encoder_weights - check fcn_encoder functon
    # 6. encoder_freeze - check fcn_encoder functon
    # 7. decoder_type - check fcn32_decoder functon
    # 8. output_activation - check fcn32_decoder functon

    #--------------------------------------------------------------------------#
    # Build Model
    #--------------------------------------------------------------------------#
    input_tensor = Input(shape=(input_shape), name='input_tensor')
    # Get block output layers and encoder model functions
    encoder_model, encoder_model_output, skip_connection_layers = fcn_encoder(
        encoder_type=encoder_type, 
        input_tensor=input_tensor, 
        encoder_weights=encoder_weights,
        encoder_freeze=encoder_freeze, 
        encoder_filters=encoder_filters)
    
    # Get model output from the decoder function
    output = fcn_decoder(
        num_classes=num_classes,
        model_type=model_type, 
        decoder_input = encoder_model_output, 
        skip_connection_layers = skip_connection_layers, 
        decoder_type = decoder_type,
        output_activation = output_activation)

    ## Build the image segmentation model using the encoder 
    ## input and decoder output layers.
    model = Model(input_tensor, output)

    ## Load model weights (if any)
    if model_weights is not None:
        model.load_weights(model_weights)

    return model



#------------------------------------------------------------------------------#
# 2. FCN32 Model
#------------------------------------------------------------------------------#
def fcn32(num_classes,
          encoder_type='Default',
          input_shape=(224,224,3),
          encoder_filters=[32, 64, 128, 256, 512],
          model_weights=None,
          encoder_weights=None,
          encoder_freeze=False,
          decoder_type='upsampling',
          output_activation=None):
    
    """
    Fully Convolution Network-32 (FCN32) Model for Semantic Segmentation.
        Args:
            num_classes: number of the segmentation classes.
            encoder_type: type of model to build upon. One of 'Default',
                          'DenseNet121', 'DenseNet169' 'EfficientNetB0',
                          'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                          'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6',
                          'EfficientNetB7', 'MobileNet', 'MobileNetV2',
                          'ResNet50', 'ResNet101', 'ResNet152',
                          'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
                          'VGG16', 'VGG19'. Default to  'Default'.
            input_shape: a tuple containing image height, width and channel
                         respectively. Default to (224,224,3).
            model_weights: (optional) Link to pre-trained weights.
                           Default to None.
            encoder_filters: (optional) number of filters to use for the encoder 
                             model blocks. Only useful with 'Default' encoder_type
                             input.
            encoder_weights: (optional) pre-trained weights for encoder function.
                             one of None (random initialization),
                             'imagenet' (pre-training on ImageNet),
                             or the path to the weights file to be loaded.
                             Default to None.
            encoder_freeze: (optional) boolean to specify whether to train encoder
                            model parameters or not. Default to False.
            output_activation: (optional) Activation for output layer.
                                Default to 'sigmoid' for 2 segmentation classes
                                or 'softmax' for more than 2 segmentation classes.
        Returns:
            (keras FCN-32 Model)
    """
    
    model = fcn(
        num_classes=num_classes,
        model_type='fcn32',
        encoder_type=encoder_type,
        input_shape=input_shape,
        encoder_filters=encoder_filters,
        model_weights=model_weights,
        encoder_weights=encoder_weights,
        encoder_freeze=encoder_freeze,
        decoder_type=decoder_type,
        output_activation=output_activation)
    
    return model



#------------------------------------------------------------------------------#
# 3. FCN16 Model
#------------------------------------------------------------------------------#
def fcn16(num_classes,
          encoder_type='Default',
          input_shape=(224,224,3),
          encoder_filters=[32, 64, 128, 256, 512],
          model_weights=None,
          encoder_weights=None,
          encoder_freeze=False,
          decoder_type='upsampling',
          output_activation=None):
    
    """
    Fully Convolution Network-16 (FCN16) Model for Semantic Segmentation.
        Args:
            num_classes: number of the segmentation classes.
            encoder_type: type of model to build upon. One of 'Default',
                          'DenseNet121', 'DenseNet169' 'EfficientNetB0',
                          'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                          'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6',
                          'EfficientNetB7', 'MobileNet', 'MobileNetV2',
                          'ResNet50', 'ResNet101', 'ResNet152',
                          'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
                          'VGG16', 'VGG19'. Default to  'Default'.
            input_shape: a tuple containing image height, width and channel
                         respectively. Default to (224,224,3).
            model_weights: (optional) Link to pre-trained weights.
                           Default to None.
            encoder_filters: (optional) number of filters to use for the encoder 
                             model blocks. Only useful with 'Default' encoder_type
                             input.
            encoder_weights: (optional) pre-trained weights for encoder function.
                             one of None (random initialization),
                             'imagenet' (pre-training on ImageNet),
                             or the path to the weights file to be loaded.
                             Default to None.
            encoder_freeze: (optional) boolean to specify whether to train encoder
                            model parameters or not. Default to False.
            output_activation: (optional) Activation for output layer.
                                Default to 'sigmoid' for 2 segmentation classes
                                or 'softmax' for more than 2 segmentation classes.
        Returns:
            (keras FCN-16 Model)
    """
    
    model = fcn(
        num_classes=num_classes,
        model_type='fcn16',
        encoder_type=encoder_type,
        input_shape=input_shape,
        encoder_filters=encoder_filters,
        model_weights=model_weights,
        encoder_weights=encoder_weights,
        encoder_freeze=encoder_freeze,
        decoder_type=decoder_type,
        output_activation=output_activation)
    
    return model



#------------------------------------------------------------------------------#
# 4. FCN8 Model
#------------------------------------------------------------------------------#
def fcn8(num_classes,
         encoder_type='Default',
         input_shape=(224,224,3),
         encoder_filters=[32, 64, 128, 256, 512],
         model_weights=None,
         encoder_weights=None,
         encoder_freeze=False,
         decoder_type='upsampling',
         output_activation=None):
    
    """
    Fully Convolution Network-8 (FCN8) Model for Semantic Segmentation.
        Args:
            num_classes: number of the segmentation classes.
            encoder_type: type of model to build upon. One of 'Default',
                          'DenseNet121', 'DenseNet169' 'EfficientNetB0',
                          'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                          'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6',
                          'EfficientNetB7', 'MobileNet', 'MobileNetV2',
                          'ResNet50', 'ResNet101', 'ResNet152',
                          'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
                          'VGG16', 'VGG19'. Default to  'Default'.
            input_shape: a tuple containing image height, width and channel
                         respectively. Default to (224,224,3).
            model_weights: (optional) Link to pre-trained weights.
                           Default to None.
            encoder_filters: (optional) number of filters to use for the encoder 
                             model blocks. Only useful with 'Default' encoder_type
                             input.
            encoder_weights: (optional) pre-trained weights for encoder function.
                             one of None (random initialization),
                             'imagenet' (pre-training on ImageNet),
                             or the path to the weights file to be loaded.
                             Default to None.
            encoder_freeze: (optional) boolean to specify whether to train encoder
                            model parameters or not. Default to False.
            output_activation: (optional) Activation for output layer.
                                Default to 'sigmoid' for 2 segmentation classes
                                or 'softmax' for more than 2 segmentation classes.
        Returns:
            (keras FCN-8 Model)
    """
    
    model = fcn(
        num_classes=num_classes,
        model_type='fcn8',
        encoder_type=encoder_type,
        input_shape=input_shape,
        encoder_filters=encoder_filters,
        model_weights=model_weights,
        encoder_weights=encoder_weights,
        encoder_freeze=encoder_freeze,
        decoder_type=decoder_type,
        output_activation=output_activation)
    
    return model


####################################################################################################
# Model Aliases
####################################################################################################
FCN = fcn
FCN32 = fcn32
FCN16 = fcn16
FCN8 = fcn8
