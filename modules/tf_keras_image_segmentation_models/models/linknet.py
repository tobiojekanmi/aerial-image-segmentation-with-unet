"""
####################################################################################
LinkNet Image Segmentation Model with Tensorflow and Keras.
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
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.python.lib.io import file_io
import warnings

from ..lib.encoders import encoder_models
from ..lib.utils import get_encoder_model
from ..lib.utils import get_encoder_model_output_layer
from ..lib.utils import get_skip_connection_layers
from ..lib.utils import get_skip_connection_layer_filters



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
# Default LinkNet Encoder
################################################################################
def linknet_encoder(encoder_type='Default',
                 input_tensor=None,
                 encoder_weights=None,
                 encoder_freeze=False,
                 num_blocks=5):
    """
    Instantiates  encoder architecture for the U-Net segmmantation model.

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
        num_blocks: number of blocks to use for each encoder. Default to 5.
        encoder_filters: a list or tuple containing the number of filters to 
                         use for each encoder block. Default to 
                         [32, 64, 128, 256, 512].
    Returns:
        encoder_model: keras model
        encoder_model_output: Encoder model output.
        skip_connection_layers: Skip connection layers/encoder block outputs 
                                before pooling.
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

    # 4. num_blocks
    if not isinstance(num_blocks, int):
        raise ValueError('The `num_blocks` argument should be integer')
    elif num_blocks <= 0:
        raise ValueError('The `num_blocks` argument cannot be less than or equal to '
                         'zero')
        
    #--------------------------------------------------------------------------#
    # Build the encoding blocks from the arguments specified
    #--------------------------------------------------------------------------#
    #--------------------------------------------------------------------------#
    # 1. Default encoding blocks
    #--------------------------------------------------------------------------#
    if encoder_type.lower() == 'default':
        encoder_filters = [2**(5+i) for i in range(num_blocks)]
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
    # 2. Pretrained model encoding blocks
    #--------------------------------------------------------------------------#
    else:
        encoder_filters = get_skip_connection_layer_filters(encoder_type)
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

    return encoder_model, encoder_model_output, encoder_filters, skip_connection_layers



################################################################################
# LinkNet Decoder
################################################################################
def linknet_decoder(num_classes,
                    decoder_input,
                    skip_connection_layers,
                    encoder_filters,
                    decoder_type='upsampling',
                    num_blocks=5,
                    num_decoder_block_conv_layers=1,
                    decoder_activation=None,
                    decoder_use_skip_connection=True,
                    decoder_use_batchnorm=True,
                    decoder_dropout_rate=0,
                    output_activation=None):
    
    """
    Instantiates decoder architecture for the LinkNet segmmantation model.
    Args:
        num_classes: number of the segmentation classes.
        decoder_input: decoder input or encoder model output.
        skip_connection_layers: encoder block outputs before pooling.
                                They will serve as skip connection inputs
                                for the decoder blocks.
        encoder_filters: (optional) a list containing filter sizes for each 
                          encoder block. Default to [32, 64, 128, 256].
        decoder_type: (optional) one of 'transpose' (to use Conv2DTanspose
                      operation for upsampling operation) or 'upsampling'
                      (to use UpSampling2D operation for upsampling operation).
                      Default to upsampling.
        num_blocks: (optional) number of encoder/decoder blocks to use. 
                    Default to 5.
        decoder_filters: (optional) a list containing filter sizes for each 
                          decoder block. Default to [512, 256, 128, 64, 32].
        num_decoder_block_conv_layers: (optional) number of convolution layers 
                                        for each decoder block (i.e. number of 
                                        Conv2D layers after upsampling layers). 
                                        Default to 1.
        decoder_activation: (optional) decoder activation name or function.
        decoder_use_skip_connection: (optional) one of True (to use skip
                                     connections) or False (not to use skip
                                     connections). Default to True.
        decoder_use_batchnorm: (optional) boolean to specify whether to
                                use BatchNormalization or not. Default to True.
        decoder_dropout_rate: (optional) dropout rate. Float between 0 and 1.
        output_activation: (optional) activation for output layer.
                            Default is either 'sigmoid' or 'softmax' based on
                            the value of the 'num_classes' argument.
    Returns:
        x: decoder output
    """
    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. number of classes specified
    if num_classes < 2:
        raise ValueError("The `num_classes` argument cannot be less than 2.")
    elif not (isinstance(num_classes, int)):
        raise ValueError("The `num_classes` argumentcan only be an integer.")

    # 2. num_blocks
    if not isinstance(num_blocks, int):
        raise ValueError('The `num_blocks` argument should be integer')
    elif num_blocks <= 0:
        raise ValueError('The `num_blocks` argument cannot be less '
                         'than or equal to zero')

    # 3. num_decoder_block_conv_layers
    if not isinstance(num_decoder_block_conv_layers, int):
        raise ValueError('The `num_decoder_block_conv_layers` argument should '
                         'be integer')
    elif num_decoder_block_conv_layers <= 0:
        raise ValueError('The `num_decoder_block_conv_layers` argument cannot '
                         'be less than or equal to zero')

    # 4. decoder_activation=None
    if decoder_activation == None:
        decoder_activation = 'relu'

    # 5. decoder_use_batchnorm=False,
    if not isinstance(decoder_use_batchnorm, bool):
        raise ValueError("The `decoder_use_batchnorm` argument should "
                         "either be True or False.")

    # 6. decoder_dropout_rate=0
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
    ## Decoder Blocks
    encoder_filters = encoder_filters[0:num_blocks]
    skip_connection_layers = skip_connection_layers[0:num_blocks]
    skip_connection_layers.reverse()
    encoder_filters.reverse()
    x = decoder_input
    if decoder_type.lower() == 'transpose':
        for decoder_block_id in range(num_blocks):
            r1 = skip_connection_layers[decoder_block_id]
            num_filters = encoder_filters[decoder_block_id]
            x = Conv2DTranspose(num_filters, 3, strides=(2, 2), 
                                padding="same")(x)
            if decoder_use_skip_connection:
                x = add([x, r1])
            for block_layer_id in range(num_decoder_block_conv_layers):
                x = convolution_block(x,
                                      num_filters=num_filters,
                                      kernel_size=(3, 3),
                                      dilation_rate=1,
                                      padding="same",
                                      use_bias=False,
                                      use_batchnorm=decoder_use_batchnorm,
                                      activation=decoder_activation)                         

    elif decoder_type.lower() == 'upsampling':
        for decoder_block_id in range(num_blocks):
            r1 = skip_connection_layers[decoder_block_id]
            num_filters = encoder_filters[decoder_block_id]
            num_filters = encoder_filters[decoder_block_id]
            x = UpSampling2D(size=(2, 2))(x)
            x = Conv2D(num_filters, (3, 3), padding='same')(x)
            x = Activation(decoder_activation)(x)
            if decoder_use_skip_connection:
                x = add([x, r1])
            for block_layer_id in range(num_decoder_block_conv_layers):
                x = convolution_block(x,
                                      num_filters=num_filters,
                                      kernel_size=(3, 3),
                                      dilation_rate=1,
                                      padding="same",
                                      use_bias=False,
                                      use_batchnorm=decoder_use_batchnorm,
                                      activation=decoder_activation)

    x = Conv2D(encoder_filters[-1], 3, activation='relu', padding='same',
               kernel_initializer='he_normal')(x)

    x = Dropout(decoder_dropout_rate)(x)

    x = Conv2D(filters=num_classes, kernel_size=(1, 1), 
               activation=output_activation, padding='same')(x)

    return x

                         
                         
#------------------------------------------------------------------------------#
# LinkNet Model
#------------------------------------------------------------------------------#
def linknet(num_classes,
         encoder_type='Default',
         input_shape=(224, 224, 3),
         model_weights=None,
         num_blocks=5,
         encoder_weights=None,
         encoder_freeze=False,
         decoder_type='upsampling',
         num_decoder_block_conv_layers=1,
         decoder_activation=None,
         decoder_use_skip_connection=True,
         decoder_use_batchnorm=True,
         decoder_dropout_rate=0,
         output_activation=None):
    """
    Merge the linknet_encoder and linknet_decoder to instantiate
    the linknet architecture for semantic segmantation tasks.

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
        num_blocks: (optional) number of encoder/decoder blocks.
        encoder_weights: (optional) pre-trained weights for encoder 
                         function. One of None (random initialization),
                         'imagenet' (pre-training on ImageNet),
                         or the path to the weights file to be loaded
        encoder_freeze: (optional) boolean to specify whether to train
                        encoder model parameters or not. Default is False.
        decoder_type: (optional) one of 'transpose' (to use Conv2DTanspose 
                       layer for deconvolution  operation) or 'upsampling' 
                       (to use UpSampling2D layer for deconvolution  operation). 
                       Default to upsampling.
        num_decoder_block_conv_layers: (optional) number of convolution layers  
                                        for each decoder block (i.e. number of 
                                        Conv2D layers after upsampling layers). 
                                        Default is 1.
        decoder_activation: (optional) decoder activation name or function.
        decoder_use_skip_connection: (optional) one of True (to use skip 
                                     connections) or False (not to use skip 
                                     connections). Default to True.
        decoder_use_batchnorm: (optional) boolean to specify whether decoder
                                layers should use BatchNormalization or not. 
                                Default is False.
        decoder_dropout_rate: (optional) dropout rate. Float between 0 and 1.
        output_activation: (optional) activation for output layer.
                            Default is either 'sigmoid' or 'softmax' based on
                            the value of the 'num_classes' argument.
    Returns:
        model: keras linknet segmentation model
    """
    # -------------------------------------------------------------------------#
    # Validate and preprocess arguments
    # -------------------------------------------------------------------------#
    # 1. num_classes - check linknet_decoder functon
    # 2. encoder_type - check linknet_encoder functon
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

    # 5. num_blocks - check linknet_encoder functon
    # 6. encoder_weights - check linknet_encoder functon
    # 7. encoder_freeze - check linknet_encoder functon
    # 8. encoder_filters - check linknet_encoder functon
    # 9. decoder_type - check linknet_decoder functon
    # 10. num_decoder_block_conv_layers - check linknet_decoder functon
    # 11. decoder_activation - check linknet_decoder functon
    # 12. decoder_use_skip_connection - check linknet_decoder functon
    # 13. decoder_use_batchnorm - check linknet_decoder functon
    # 14. decoder_dropout_rate - check linknet_decoder functon
    # 15. output_activation - check linknet_decoder functon

    # -------------------------------------------------------------------------#
    # Build Model
    # -------------------------------------------------------------------------#
    # 1. Get the encoder model, model output layer and skip connection layers
    input_tensor = Input(shape=(input_shape), name='input')    
    encoder_model, encoder_model_output, encoder_filters, skip_connection_layers = linknet_encoder(
        encoder_type=encoder_type,
        input_tensor=input_tensor,
        encoder_weights=encoder_weights,
        encoder_freeze=encoder_freeze, 
        num_blocks=num_blocks)

    # 3. Decoder blocks
    # Extend the model by adding the decoder blocks
    outputs = linknet_decoder(
        num_classes=num_classes,
        decoder_input = encoder_model_output,
        skip_connection_layers=skip_connection_layers,
        encoder_filters=encoder_filters,
        decoder_type=decoder_type,
        num_blocks=num_blocks,
        num_decoder_block_conv_layers=num_decoder_block_conv_layers,
        decoder_activation=decoder_activation,
        decoder_use_skip_connection=decoder_use_skip_connection,
        decoder_use_batchnorm=decoder_use_batchnorm,
        decoder_dropout_rate=decoder_dropout_rate,
        output_activation=output_activation)

    inputs = encoder_model.input

    ## Image Segmentation Model
    model = Model(inputs, outputs)

    return model