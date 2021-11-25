"""
Preprocessing Functions for Tensorflow Keras FCN Image Segmentation Models.
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt




def list_image_paths(image_directory, mask_directory):
    """
    Extracts image and their segmentation filenames from their
    directories.
    Args:
      image_directory: path to the image directory
      mask_directory:  path to the segmentation labels
                      directory
    Returns:
      image_paths: a list containing filepaths of the
                   images in the specified directory path
      mask_paths: a list containing filepaths of the image
                   segmentation labels in the specified
                   directory path
    """

    image_paths = []
    mask_paths = []
    image_filenames = os.listdir(image_directory)

    for image_filename in image_filenames:
        image_paths.append(image_directory + "/" + image_filename)
        mask_filename = image_filename.replace('.jpg', '.png')
        mask_paths.append(mask_directory + "/" + mask_filename)

    return image_paths, mask_paths



class Augment(tf.keras.layers.Layer):
    def __init__(self, augment_type="horizontal_and_vertical", seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(
        mode=augment_type, seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(
        mode=augment_type, seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels



def data_generator(image_paths, mask_paths, image_size=(224,224),
                   image_channels=3, batch_size=32, cache_dataset=True,
                   shuffle_dataset=True, buffer_size=512, augment=True,
                   augment_type="horizontal_and_vertical"):
    """
    Generates TF dataset containinng images and their segmentations.

    Arguments:
        image_paths: a list or tuple containing paths to the images.
        mask_paths: a list or tuple containing paths to the segmentations.
                    Note: The mask_paths list must follow the same format which
                    the image_paths is (i.e. The image path in index 0 in the
                    image_paths list must have it's segmentation/mask path to
                    be at 0 in the mask_paths list). Inability to ensure this
                    will lead to incorrect modelling.
        image_size: a tuple containing image height and image width respecively.
                    i.e. (image_height, image_width). Default to (224,224).
        image_channels: number of image channels.  Default is 3.
        batch_size: (optional) size of image/mask to load per batch.
                    Default to 32.
        cache_dataset: (optional) boolean to specify whether to store dataset
                       into memory after it has been loaded during the first
                       training epoch. Default is True.
        shuffle_dataset: (optional) boolean to specify whether to shuffle
                         dataset while loading or not. Default to True.
        buffer_size: (optional) buffer size. Default is 512.
        augment: (optional) boolean specifying whether to augment dataset or
                 not while loading the images and their segmentation labels.
                 Default to True.
        augment_type: (optional) a tensorflow ImageDataset data augmentation or
                      transformation type. Default to "horizontal_and_vertical".

    Returns:
        A tensorflow.data.Dataset pipeline to load images and their
        segmentation labels from the specified argument.
        """

    #--------------------------------------------------------------------------#
    # Validate and preprocess arguments
    #--------------------------------------------------------------------------#
    # 1. image size
    if len(image_size) != 2:
        raise ValueError('The `image_size` argument should be a tuple '
                         'with image height and width respectively '
                         'i.e. (256, 256).')
    elif not (isinstance(image_size[0], int) or isinstance(image_size[1], int)):
        raise ValueError('The `image_size` argument should be a tuple '
                         'containing only integer values.')
    # 2. batch size
    if not isinstance(batch_size, int):
        raise ValueError('The `batch_size` argument should be integer.')
    # 3. cache dataset
    if not (cache_dataset in {True, False}):
        raise ValueError("The `cache_dataset` argument should either be "
                         "True or False.")
    # 4. shuffle dataset and buffer size
    if not (shuffle_dataset in {True, False}):
        raise ValueError("The `shuffle_dataset` argument should either be "
                         "True or False.")

    #--------------------------------------------------------------------------#
    # Build function using the specified arguments
    #--------------------------------------------------------------------------#
    # Create a function to read images and masks into arrays.
    def read_image(image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=image_channels,
                                      expand_animations = False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_size, method='nearest')

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=image_channels,
                                     expand_animations = False)
        mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
        mask = tf.image.resize(mask, image_size, method='nearest')
        return image, mask

    image_list = tf.constant(image_paths)
    mask_list = tf.constant(mask_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(read_image)
    if augment:
        dataset = dataset.map(Augment(augment_type=augment_type))
    if cache_dataset:
        dataset = dataset.cache()
    if shuffle_dataset:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)


    return dataset


def display(display_list, figsize=(15, 15)):
    """
    Displays the image, its true segmentation label
    and predicted segmentation label
    Args:
        display_list: a list containing the image,
        its true segmentation label and predicted
        segmentation label repectively
        figsize: (optional). a tuple containing figure size.
                 Default to (15, 15).
    """
    plt.figure(figsize=figsize)

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
