"""
####################################################################################
DeepLabV3Plus Image Segmentation Model with Tensorflow and Keras.
####################################################################################
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
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
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
# Atrous/Dilated Spatial Pyramid Pooling Function
################################################################################
def DilatedSpatialPyramidPooling(dspp_input, 
                                 num_filters=256, 
                                 dilation_rates=[1,6,12,18]):
    """
    Instantiates the Atrous/Dilated Spatial Pyramid Pooling (ASPP/DSPP)
    architecture for the DoubleU-Net segmentation model.

    Args:
        dspp_input: DSPP input or encoder model ouput
        num_filters: Number of convolution filters.
        dilation_rates: a list containing dilate rates.
    Returns:
     dspp_output: dspp block output
    """
    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. num_filters
    if not isinstance(num_filters, int):
        num_filters = int(num_filters)
        warnings.warn("The `num_filters` argument is not an integer. "
                      "It will be rounded to the nearest integer "
                      "(if it's data type is float). ")

    # 2. dilation_rates
    if not (isinstance(dilation_rates, tuple) or isinstance(dilation_rates, list)):
        raise ValueError("The `dilation_rates` argument should either a list "
                         "or tuple.")

    #--------------------------------------------------------------------------#
    # Build the DSSP function from the arguments specified
    #--------------------------------------------------------------------------#
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, num_filters=num_filters, kernel_size=1, use_bias=True)
    out_pool = UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
                            interpolation="bilinear",)(x)

    out_1 = convolution_block(dspp_input, num_filters=num_filters,
                              kernel_size=1, dilation_rate=dilation_rates[0])
    out_6 = convolution_block(dspp_input, num_filters=num_filters,
                              kernel_size=3, dilation_rate=dilation_rates[1])
    out_12 = convolution_block(dspp_input, num_filters=num_filters,
                               kernel_size=3, dilation_rate=dilation_rates[2])
    out_18 = convolution_block(dspp_input, num_filters=num_filters,
                               kernel_size=3, dilation_rate=dilation_rates[3])

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])

    dspp_output = convolution_block(x, num_filters=num_filters,
                               kernel_size=1, dilation_rate=1)

    return dspp_output



################################################################################
# DeepLabV3Plus Encoder
################################################################################
def deeplabV3plus_encoder(encoder_type='Default',
                          input_tensor=None,
                          encoder_weights=None,
                          encoder_freeze=False, 
                          num_dspp_filters=256, 
                          dspp_dilation_rates=[1,6,12,18]):
    """
    Instantiates  encoder architecture for the DeepLabV3Plus segmmantation model.

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
        encoder_weights: (optional) One of `None` (random initialization), 
                         `imagenet`, or the path to the weights file to be
                          loaded. Default to None.
        encoder_freeze: (optional) Boolean Specifying whether to train encoder 
                        parameters or not. Default to False.
        num_dspp_filters: (optional) number of dspp blocks convolution filters. 
                          Default to 256.
        dspp_dilation_rates: (optional) dspp block dilation rates. Default to  
                              [1, 6, 12, 18].
    Returns:
        encoder_model: keras model
        encoder_model_output_1: DCNN block output.
        encoder_model_output_2: ASPP/DSPP block output.
    """
    #------------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #------------------------------------------------------------------------------#
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
        if not (encoder_weights in {None} or 
                file_io.file_exists_v2(encoder_weights)):
            warnings.warn('Ensure the `encoder_weights` argument is either '
                          '`None` (random initialization), '
                          'or the path to the weights file to be loaded. ')
    else:
        if not (encoder_weights in {'imagenet', None} or 
                file_io.file_exists_v2(encoder_weights)):
            warnings.warn('The `encoder_weights` argument should be either '
                          '`None` (random initialization), `imagenet` '
                          '(pre-training on ImageNet), or the path to the '
                          'weights file to be loaded.')

    # 3. encoder_freeze
    if not isinstance(encoder_freeze, bool):
        raise ValueError("The `encoder_freeze` argument should either be True or "
                         "False.")
        
    # 4. num_dspp_filters
    if not isinstance(num_dspp_filters, int):
        raise ValueError("The `num_dspp_filters` argument should an integer.")
        
    #--------------------------------------------------------------------------#
    # Build the encoding blocks from the arguments specified
    #--------------------------------------------------------------------------#
    # 1. Default encoding blocks
    #--------------------------------------------------------------------------#
    if encoder_type.lower() == 'default':
        x = input_tensor
        kernel_size = 3
        pool_size = 2
        num_blocks = 5
        encoder_filters = [32, 64, 128, 256, 512]
        encoder_model_outputs = []

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

            encoder_model_outputs.append(x)
            x = MaxPooling2D((pool_size, pool_size))(x)
        
        final_encoder_model_output = x

        # Create model
        encoder_model = Model(input_tensor, final_encoder_model_output)
        encoder_model_output_1 = encoder_model_outputs[2]
        dspp_input = encoder_model_outputs[4]
        encoder_model_output_2 = DilatedSpatialPyramidPooling(
            dspp_input=dspp_input, 
            num_filters = num_dspp_filters, 
            dilation_rates=dspp_dilation_rates
        )

    #--------------------------------------------------------------------------#
    # 2. Pretrained model encoding blocks
    #--------------------------------------------------------------------------#
    else:
        
        encoder_model = get_encoder_model(
            encoder_type, input_tensor, encoder_weights
        )
        encoder_model_output_1 = get_encoder_model_output_layer(
            encoder_type, encoder_model, num_blocks=2
        )
        dspp_input = get_encoder_model_output_layer(
            encoder_type, encoder_model, num_blocks=4
        )
        encoder_model_output_2 = DilatedSpatialPyramidPooling(
            dspp_input=dspp_input, 
            num_filters = num_dspp_filters, 
            dilation_rates=dspp_dilation_rates
        )
        # encoder_model_output_2 = x

    # Make the model parameters trainable or non trainable
    encoder_model.trainable = not(encoder_freeze)

    return encoder_model, encoder_model_output_1, encoder_model_output_2



################################################################################
# DeepLabV3Plus Decoder
################################################################################
def deeplabV3plus_decoder(num_classes,
                          decoder_input_1,
                          decoder_input_2,
                          input_shape=(224,224,3),
                          decoder_type='upsampling',
                          num_decoder_filters=256,
                          decoder_activation=None,
                          decoder_use_batchnorm=True,
                          decoder_dropout_rate=0,
                          output_activation=None):
    
    """
    Instantiates decoder architecture for the DeepLabV3Plus segmmantation model.
    Args:
        num_classes: number of the segmentation classes.
        decoder_input_1: first decoder input or Deep CNN output. 
        decoder_input_2: second decoder input or DSPP/ASPP output.
        input_shape: a tuple containing image height, width and channels
                      respectively. Default to (224,224,3).
        decoder_type: (optional) one of 'transpose' (to use Conv2DTanspose 
                       layer for deconvolution  operation) or 'upsampling' 
                       (to use UpSampling2D layer for deconvolution  operation). 
                       Default to upsampling.
        num_decoder_filters: (optional) number of decoder blocks convolution 
                             filters. Default to 256.
        decoder_activation: (optional) decoder activation name or function.
        decoder_use_batchnorm: (optional) boolean to specify whether decoder
                                layers should use BatchNormalization or not. 
                                Default to True.
        decoder_dropout_rate: (optional) dropout rate. Float between 0 and 1.
        output_activation: (optional) activation for output layer.
                            Default is either 'sigmoid' or 'softmax' based on
                            the value of the 'num_classes' argument.        
    """
    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. number of classes specified
    if num_classes < 2:
        raise ValueError("The `num_classes` argument cannot be less than 2.")
    elif not (isinstance(num_classes, int)):
        raise ValueError("The `num_classes` argumentcan only be an integer.")
    
    # 2. decoder_type
    if not decoder_type.lower() in {'upsampling', 'transpose'}:
        raise ValueError("The `decoder_type` argument should be oe of "
                         "'upsampling' or 'transpose'.")

    # 3. num_dspp_filters
    if not isinstance(num_decoder_filters, int):
        raise ValueError("The `num_decoder_filters` argument should an integer.")
    
    # 4. decoder_activation=None
    if decoder_activation == None:
        decoder_activation = 'relu'

    # 5. decoder_use_batchnorm=False,
    if not isinstance(decoder_use_batchnorm, bool):
        raise ValueError("The `decoder_use_batchnorm` argument should "
                         "either be True or False.")
    
    # 6. decoder_dropout_rate
    if not (isinstance(decoder_dropout_rate, int) or 
            isinstance(decoder_dropout_rate, float)):
        raise ValueError('The `decoder_use_dropout` argument should be'
                         ' an integer or float between 0 and 1')
    elif decoder_dropout_rate < 0 or decoder_dropout_rate > 1:
        raise ValueError("The `decoder_use_dropout` argument cannot be "
                         "less than 0 or greater than 1.")

    # 7. output activation
    if output_activation == None:
        if num_classes == 2:
            output_activation = 'sigmoid'
        else:
            output_activation = 'softmax'

    #--------------------------------------------------------------------------#
    # Build the decoder blocks from the arguments specified
    #--------------------------------------------------------------------------#    
    image_width = input_shape[0]
    image_height = input_shape[1]    
    x = decoder_input_2
    kernel_size = 3
    
    decoder_input_a = UpSampling2D(
        size=(image_width // 4 // x.shape[1], image_height // 4 // x.shape[2]),
        interpolation="bilinear")(x)
    
    decoder_input_b = convolution_block(
        decoder_input_1, num_filters=num_decoder_filters, kernel_size=1
    )

    x = Concatenate(axis=-1)([decoder_input_a, decoder_input_b])
    x = convolution_block(x,
                          num_filters=num_decoder_filters,
                          kernel_size=kernel_size,
                          dilation_rate=1,
                          padding="same",
                          use_bias=False,
                          use_batchnorm=decoder_use_batchnorm,
                          activation=decoder_activation)

    x = convolution_block(x,
                          num_filters=num_decoder_filters,
                          kernel_size=kernel_size,
                          dilation_rate=1,
                          padding="same",
                          use_bias=False,
                          use_batchnorm=decoder_use_batchnorm,
                          activation=decoder_activation)
    
    if decoder_type.lower() == 'upsampling':
        x = UpSampling2D(
            size=(image_width // x.shape[1], image_height // x.shape[2]),
            interpolation="bilinear",
        )(x)
        x = Conv2D(num_decoder_filters, kernel_size=(3, 3), padding="same")(x)
    elif decoder_type.lower() == 'transpose':
        x = Conv2DTranspose(
            num_decoder_filters, 3, 
            strides=(image_width // x.shape[1], image_height // x.shape[2]), 
            padding="same",
        )(x)
    
    if decoder_dropout_rate > 0:
        x = Conv2D(num_decoder_filters, kernel_size=(3, 3), padding="same")(x)
        x = Dropout(decoder_dropout_rate)(x)
    
    model_output = Conv2D(num_classes, kernel_size=(1, 1), 
                          padding="same", activation=output_activation)(x)
            
    return model_output



################################################################################
# DeepLabV3Plus Model
################################################################################
def deeplabV3plus(num_classes,
                  encoder_type='ResNet50',
                  input_shape=(224,224,3),
                  model_weights=None,
                  encoder_weights=None,
                  encoder_freeze=False,
                  num_dspp_filters=256,
                  dspp_dilation_rates=[1,6,12,18],                  
                  decoder_type='upsampling',
                  num_decoder_filters = 256,
                  decoder_use_batchnorm=True,
                  decoder_activation=None,
                  decoder_dropout_rate=0,
                  output_activation=None):
    """
    Instantiates the DeepLabV3Plus architecture for semantic segmantation
    tasks.
    
    Args:
        num_classes: number of the segmentation classes.
        encoder_type: type of model to build upon. One of 'Default',
                      'DenseNet121', 'DenseNet169' 'EfficientNetB0',
                      'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                      'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6',
                      'EfficientNetB7', 'MobileNet', 'MobileNetV2',
                      'ResNet50', 'ResNet101', 'ResNet152',
                      'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
                      'VGG16', 'VGG19'. Default to 'Default'.
        input_shape: a tuple containing image height, width and channels
                      respectively. Default to (224,224,3).
        model_weights: (optional) link to pre-trained weights.
        encoder_weights: (optional) pre-trained weights for encoder 
                         function. One of None (random initialization),
                         'imagenet' (pre-training on ImageNet),
                         or the path to the weights file to be loaded
        encoder_freeze: (optional) boolean to specify whether to train
                        encoder model parameters or not. Default is False.
        num_dspp_filters: (optional) number of dspp block convlution filters.
                         Default to 256.
        dspp_dilation_rates: (optional) a list containing  dspp block dlation  
                         rates. Default to [1, 6, 12, 18].
        decoder_type: (optional) one of 'transpose' (to use Conv2DTanspose 
                       layer for deconvolution  operation) or 'upsampling' 
                       (to use UpSampling2D layer for deconvolution  operation). 
                       Default to upsampling.
        num_decoder_filters: (optional) number of decoder blocks convlution 
                             filters. Default to 256.
        decoder_activation: (optional) decoder activation name or function.
        decoder_use_batchnorm: (optional) boolean to specify whether decoder
                                layers should use BatchNormalization or not. 
                                Default is False.
        decoder_dropout_rate: (optional) dropout rate. Float between 0 and 1.
        output_activation: (optional) activation for output layer.
                            Default is either 'sigmoid' or 'softmax' based on
                            the value of the 'num_classes' argument.        
    Returns:
        model: keras deeplabv3plus segmentation model
    """

    # -------------------------------------------------------------------------#
    # Validate and preprocess arguments
    # -------------------------------------------------------------------------#
    # 1. num_classes - check deeplabv3plus_decoder functon
    # 2. encoder_type - check deeplabv3plus_encoder functon
    # 3. input_shape 
    if not isinstance(input_shape, tuple):
        raise ValueError("The `input_shape` argument should a tuple containing "
                         "the image width, height and channels respectively.")
    if not len(input_shape) == 3:
        warnings.warn("The `input_shape` argument should be a tuple containg three "
                      "integer values for each of the image width, height and "
                      "channels respectively.")
    
    # 4. model_weights
    if not (model_weights in {None} or file_io.file_exists_v2(model_weights)):
        warnings.warn('The `model_weights` argument should either be '
                      '`None` (random initialization), '
                      'or the path to the weights file to be loaded.')

    # 5. encoder_weights - check deeplabv3plus_encoder functon
    # 6. encoder_freeze - check deeplabv3plus_encoder functon
    # 7. num_dspp_filters - check deeplabv3plus_encoder functon
    # 8. dspp_dilation_rates - check deeplabv3plus_encoder functon
    # 9. decoder_type - check deeplabv3plus_decoder functon
    # 10. num_decoder_filters - check deeplabv3plus_decoder functon
    # 11. decoder_activation - check deeplabv3plus_decoder functon
    # 12. decoder_use_batchnorm - check deeplabv3plus_decoder functon
    # 13. decoder_dropout_rate - check deeplabv3plus_decoder functon
    # 14. output_activation - check deeplabv3plus_decoder functon

    # -------------------------------------------------------------------------#
    # Build Model
    # -------------------------------------------------------------------------#
    input_tensor = Input(shape=(input_shape))
    
    encoder_model, encoder_model_output_1, encoder_model_output_2 = deeplabV3plus_encoder(
        encoder_type=encoder_type, 
        input_tensor=input_tensor, 
        encoder_weights=encoder_weights, 
        encoder_freeze=encoder_freeze
    )
    
    
    model_output = deeplabV3plus_decoder(
        num_classes=num_classes,
        decoder_input_1=encoder_model_output_1,
        decoder_input_2=encoder_model_output_2,
        input_shape=input_shape,
        decoder_type=decoder_type,
        num_decoder_filters=num_decoder_filters,
        decoder_use_batchnorm=decoder_use_batchnorm,
        decoder_activation=decoder_activation,
        decoder_dropout_rate=decoder_dropout_rate,
        output_activation=output_activation)
    
    model = Model(inputs=input_tensor, outputs=model_output)

    return model
    

    
    
    
################################################################################
# Aliases
################################################################################
DeepLabV3Plus = deeplabV3plus    
    
    
    
    
    
    
