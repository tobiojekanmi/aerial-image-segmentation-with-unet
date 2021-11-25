"""
Preprocessing Functions for Tensorflow Keras FCN Image Segmentation Models.
"""
import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import display


def predict_mask(dataset, model):

    """
    Predicts segementation labels/masks and returns both the true and
    predicted masks

    Args:
    dataset: tf.data Dataset containing both the images and true masks
    model: model to use for prediction
    Returns:
     true_masks, pred_masks
    """
    true_masks, pred_masks  = [], []
    for images, masks in dataset:
        y_pred = model.predict(images)
        y_pred2 = tf.expand_dims(tf.argmax(y_pred, axis=-1), axis=-1)
        pred_masks.extend(y_pred2)
        if masks.shape[-1] != 1:
            y_true = tf.expand_dims(tf.argmax(true_masks, axis=-1), axis=-1)
            true_masks.extend(y_true)
        else:
            true_masks.extend(masks)

    true_masks, pred_masks = np.array(true_masks), np.array(pred_masks)
    return true_masks, pred_masks



def show_predictions(dataset, model, num_batches, num_per_batch, random_state=True):
    """
    Displays the image, it's true segmentation label and
    predicted label.

    Args:
        dataset: tf.data Dataset containing both the images and true masks
        model: model to use for prediction
        num_batches: number of batches
        num_per_batch: number of images to display per batch
    """
    for images, true_masks in dataset.take(num_batches):
        y_pred = model.predict(images)
        pred_masks = tf.expand_dims(tf.argmax(y_pred, axis=-1), axis=-1)
        if true_masks.shape[-1] != 1:
            true_masks = tf.expand_dims(tf.argmax(true_masks, axis=-1), axis=-1)

        for iter_id in range(num_per_batch):
            if random_state:
                image_id = random.randint(0, num_per_batch - 1)
            else:
                image_id = iter_id
            image = images[image_id]
            true_mask = true_masks[image_id]
            pred_mask = pred_masks[image_id]
            display_list = [images[image_id], true_masks[image_id], pred_masks[image_id]]
            display([image, true_mask, pred_mask])
