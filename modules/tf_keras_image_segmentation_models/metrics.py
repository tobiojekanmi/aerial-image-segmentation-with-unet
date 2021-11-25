"""
Metrics for Tensorflow Keras Image Segmentation Models.
"""

import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K



#########-------- Evaluation Metrics -------- #########

class ConfusionMatrix(Metric):
    """
    Computes Multi-label/Multi-class confusion matrix.
    Returns either:
    1. Overall confusion matrix, or
    2. Class-wise confusion matrix.
    If multi-class input is provided, it will be treated
    as multilabel data.
    Consider classification problem with two classes
    (i.e num_classes=2).

    The resultant matrix `M`will be in the shape of:
    1. (2,2) for the overall confusion matrix would be (2,2)
    2. `(num_classes, 2, 2)` for class-wise confusion matrix.
        Every class `i` has a dedicated matrix of shape `(2, 2)` that contains:
        - true negatives for class `i` in `M(0,0)`
        - false positives for class `i` in `M(0,1)`
        - false negatives for class `i` in `M(1,0)`
        - true positives for class `i` in `M(1,1)`

    Arguments:
        num_classes: `int`, the number of labels/classes the prediction task can have.
        result_type: one of 'overall' or 'class-wise'
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:
        >>> # overall confusion matrix
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = ConfusionMatrix(num_classes=2, result_type='overall')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        array([[[2., 0.],
        [0., 2.]]], dtype=float32)
        >>> # class-wise confusion matrix
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = ConfusionMatrix(num_classes=2, result_type='class-wise')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        array([[[1., 0.],
        [0., 1.]],
        <BLANKLINE>
       [[1., 0.],
        [0., 1.]]], dtype=float32)
    """

    def __init__(self, num_classes, result_type='overall', name='confusion_matrix', dtype=None, **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, dtype=dtype, **kwargs)
        # Validate arguments
        if not isinstance(num_classes, int):
            raise ValueError("The `num_classes` argument should be an integer value. ")

        if not result_type.lower() in {'overall','class', 'class-wise'}:
            raise ValueError("The `result_type` argument should be either 'overall' or 'class-wise'. ")

        self.result_type = result_type.lower()
        self.num_classes = num_classes
        # Create variables to store confusion matrix outputs
        self.true_positives = self.add_weight(
            "true_positives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self._dtype,
        )
        self.false_positives = self.add_weight(
            "false_positives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self._dtype,
        )
        self.false_negatives = self.add_weight(
            "false_negatives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self._dtype,
        )
        self.true_negatives = self.add_weight(
            "true_negatives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self._dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Computes the true positives, true negatives,
        false positives and true negatives metric values.
            Args:
                y_true: The ground truth values.
                y_pred: The predicted values.
        """
        # Validate and preprocess arguments
        # 1. Flatten the input arrays if its rank > 1.
        y_true = tf.cast(y_true, self._dtype)
        if y_true.shape.ndims > 1:
            if y_true.shape[-1] == self.num_classes:
                y_true = tf.expand_dims(tf.argmax(y_true, axis=-1), axis=-1)
            y_true = tf.reshape(y_true, [-1])

        y_pred = tf.cast(y_pred, self._dtype)
        if y_pred.shape.ndims > 1:
            if y_pred.shape[-1] == self.num_classes:
                y_pred = tf.expand_dims(tf.argmax(y_pred, axis=-1), axis=-1)
            y_pred = tf.reshape(y_pred, [-1])

        # 2. Flatten the input arrays if its rank > 1.
        #if y_true.shape != y_pred.shape:
        #    raise ValueError("The `y_true` array size does not match with the `y_pred` array size. ")

        # 3. Enforce sample_weight to be none.
        if sample_weight is not None:
            raise ValueError("`sample_weight` is not None. "
                             "Be aware that the ConfusionMatrix "
                             "metric does not take `sample_weight` "
                             "into account when computing its value.")

        # Compute the confusion matrix values
        true_positive = [tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_id), tf.equal(y_pred, class_id)), self._dtype))
                         for class_id in range(self.num_classes)]
        true_negative = [tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_true, class_id), tf.not_equal(y_pred, class_id)), self._dtype))
                         for class_id in range(self.num_classes)]
        false_positive = [tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_true, class_id), tf.equal(y_pred, class_id)), self._dtype))
                          for class_id in range(self.num_classes)]
        false_negative = [tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, class_id), tf.not_equal(y_pred, class_id)), self._dtype))
                          for class_id in range(self.num_classes)]

        # Update confusion matrix variables
        self.true_positives.assign_add(tf.cast(true_positive, self.dtype))
        self.false_positives.assign_add(tf.cast(false_positive, self.dtype))
        self.false_negatives.assign_add(tf.cast(false_negative, self.dtype))
        self.true_negatives.assign_add(tf.cast(true_negative, self.dtype))

    def result(self):
        """
        Returns the confusion matrix values based on the specified result_type argumen.
        """

        if self.result_type in {'class', 'class-wise'}:
            flat_confusion_matrix = tf.convert_to_tensor(
                [
                    self.true_negatives,
                    self.false_positives,
                    self.false_negatives,
                    self.true_positives,
                ]
            )

        elif self.result_type == 'overall':
            flat_confusion_matrix = tf.convert_to_tensor(
                [
                    tf.reduce_sum(self.true_negatives),
                    tf.reduce_sum(self.false_positives),
                    tf.reduce_sum(self.false_negatives),
                    tf.reduce_sum(self.true_positives),
                ]
            )
        # Reshape into 2*2 matrix
        confusion_matrix = tf.reshape(tf.transpose(flat_confusion_matrix), [-1, 2, 2])

        return confusion_matrix


    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        reset_value = np.zeros(self.num_classes, dtype=np.int32)
        K.batch_set_value([(v, reset_value) for v in self.variables])

    def reset_states(self):
        return self.reset_state()




class IoU(ConfusionMatrix):
    """
    Computes the Intersection-over-Union (IoU) metric via the confusion matrix.
    IoU is defined as follows:
    IoU = true_positive / (true_positive + false_positive + false_negative).
    If multi-class input is provided, it will be treated as multilabel data.
    Also, based on the specified arguments, this metric returns either:
    1. Overall Intersection-over-Union, or
    2. Class-wise Intersection-over-Union, or
    3. Mean Intersection-over-Union.

    Consider classification problem with two classes
    (i.e num_classes=2).
    The output of this metric would be:
    1. a float value for both 'overall' and 'mean' iou, or
    2. (num_classes, 1) array for 'class-wise' iou. Each value of the array
        represents the iou of each class.

    Arguments:
        num_classes: `int`, the number of labels/classes the prediction task can have.
        result_type: one of 'overall', 'mean', or 'class-wise'
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:
        >>> # overall iou
        >>> from metrics import IoU
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = IoU(num_classes=2, result_type='overall')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        1.0
        >>> # mean iou
        >>> from metrics import IoU
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = IoU(num_classes=2, result_type='mean')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        1.0
        >>> # class-wise confusion matrix
        >>> from metrics import IoU
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = IoU(num_classes=2, result_type='class-wise')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        array([1., 1.], dtype=float32)

    Usage with `compile()` API:
        ```python
        from metrics import IoU
        iou = IoU(num_classes=3, result_type='mean', name='iou')
        model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[iou])
        """

    def __init__(self, num_classes, result_type='mean', name='iou', dtype=None, **kwargs):
        # Validate class arguments/attributes
        if not isinstance(num_classes, int):
            raise ValueError("The `num_classes` argument should be an integer.")

        if not result_type.lower() in {'overall','mean', 'class', 'class-wise'}:
            raise ValueError("The `result_type` argument should be either 'overall', 'mean', or 'class-wise'.")

        # Import the confusion matrix attributes
        self.num_classes = num_classes
        super().__init__(self.num_classes, result_type='class',  name=name, dtype=dtype, **kwargs)
        self.result_type = result_type.lower()


    def update_state(self, y_true, y_pred, sample_weight=None):
        """"
        Computes the confusion matrix for the arguments specified.
        """
        if sample_weight is not None:
            raise ValueError("`sample_weight` is not None. "
                             "Be aware that the IoU "
                             "metric does not take `sample_weight` "
                             "into account when computing its value.")

        # Compute the confusion matrix.
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        """
        Computes the IoU from the true positive, false postive, and false negative
        values from the confusion matrix computed by the update_state funtion.
        """

        if self.result_type in {'class', 'class-wise'}:
            iou = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.false_positives + self.false_negatives))
            iou = tf.cast(iou, self._dtype)

        elif self.result_type == 'overall':
            iou = tf.math.divide_no_nan(tf.reduce_sum(self.true_positives),
                                        tf.reduce_sum((self.true_positives + self.false_positives + self.false_negatives)))
            iou = tf.cast(iou, self._dtype)

        elif self.result_type == 'mean':
            iou = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.false_positives + self.false_negatives))
            iou = tf.reduce_mean(tf.cast(iou, self._dtype))

        return iou


    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        super().get_config()

    def reset_state(self):
        super().reset_state()

    def reset_states(self):
        super().reset_states()



class Precision(ConfusionMatrix):
    """
    Computes the Precision metric via the confusion matrix.
    Precision is defined as follows:
    Precision = true_positive / (true_positive + false_positive).
    If multi-class input is provided, it will be treated as multilabel data.
    Also, based on the specified arguments, this metric returns either:
    1. Overall Precision, or
    2. Class-wise Precision, or
    3. Mean Precision.

    Consider classification problem with two classes
    (i.e num_classes=2).
    The output of this metric would be:
    1. a float value for both 'overall' and 'mean' precision, or
    2. (num_classes, 1) array for 'class-wise' precision. Each value of
        the array represents the precision of each class.

    Arguments:
        num_classes: `int`, the number of labels/classes the prediction task can have.
        result_type: one of 'overall', 'mean', or 'class-wise'
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:
        >>> # overall precsion
        >>> from metrics import Precision
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = Precision(num_classes=2, result_type='overall')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        1.0
        >>> # mean precision
        >>> from metrics import Precision
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = Precision(num_classes=2, result_type='mean')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        1.0
        >>> # class-wise precision
        >>> from metrics import Precision
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = Precision(num_classes=2, result_type='class-wise')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        array([1., 1.], dtype=float32)

    Usage with `compile()` API:
        ```python
        from metrics import Precision
        precision = Precision(num_classes=3, result_type='mean', name='precision')
        model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[precision])
        """

    def __init__(self, num_classes, result_type='mean', name='precision', dtype=None, **kwargs):
        # Validate class arguments/attributes
        if not isinstance(num_classes, int):
            raise ValueError("The `num_classes` argument should be an integer.")

        if not result_type.lower() in {'overall', 'mean', 'class', 'class-wise'}:
            raise ValueError("The `result_type` argument should be either 'overall', 'mean', or 'class-wise'.")

        # Import the confusion matrix attributes
        self.num_classes = num_classes
        super().__init__(self.num_classes, result_type='class',  name=name, dtype=dtype, **kwargs)
        self.result_type = result_type.lower()


    def update_state(self, y_true, y_pred, sample_weight=None):
        """"
        Computes the confusion matrix for the arguments specified.
        """
        if sample_weight is not None:
            raise ValueError("`sample_weight` is not None. "
                             "Be aware that the Precision "
                             "metric does not take `sample_weight` "
                             "into account when computing its value.")

        # Compute the confusion matrix.
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        """
        Computes the precision metric from the true positives and false postives
        values from the confusion matrix computed by the update_state funtion.
        """

        if self.result_type in {'class', 'class-wise'}:
            precision = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.false_positives))
            precision = tf.cast(precision, self._dtype)

        elif self.result_type == 'overall':
            precision = tf.math.divide_no_nan(tf.reduce_sum(self.true_positives),
                                        tf.reduce_sum((self.true_positives + self.false_positives)))
            precision = tf.cast(precision, self._dtype)

        elif self.result_type == 'mean':
            precision = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.false_positives))
            precision = tf.reduce_mean(tf.cast(precision, self._dtype))

        return precision

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        super().get_config()

    def reset_state(self):
        super().reset_state()

    def reset_states(self):
        super().reset_states()



class Recall(ConfusionMatrix):
    """
    Computes the Recall metric via the confusion matrix.
    Recall is defined as follows:
    Recall = true_positive / (true_positive + false_positive + false_negative).
    If multi-class input is provided, it will be treated as multilabel data.
    Also, based on the specified arguments, this metric returns either:
    1. Overall Recall, or
    2. Class-wise Recall, or
    3. Mean Recall.

    Consider classification problem with two classes
    (i.e num_classes=2).
    The output of this metric would be:
    1. a float value for both 'overall' and 'mean' recall, or
    2. (num_classes, 1) array for 'class-wise' recall. Each value of
        the array represents the recall of each class.

    Arguments:
        num_classes: `int`, the number of labels/classes the prediction task can have.
        result_type: one of 'overall', 'mean', or 'class-wise'
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:
        >>> # overall recall
        >>> from metrics import Recall
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = Recall(num_classes=2, result_type='overall')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        1.0
        >>> # mean recall
        >>> from metrics import Recall
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = Recall(num_classes=2, result_type='mean')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        1.0
        >>> # class-wise recall
        >>> from metrics import Recall
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = Recall(num_classes=2, result_type='class-wise')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        array([1., 1.], dtype=float32)

    Usage with `compile()` API:
        ```python
        from metrics import Recall
        recall = Recall(num_classes=3, result_type='mean', name='recall')
        model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[recall])
        """

    def __init__(self, num_classes, result_type='mean', name='recall', dtype=None, **kwargs):
        # Validate class arguments/attributes
        if not isinstance(num_classes, int):
            raise ValueError("The `num_classes` argument should be an integer.")

        if not result_type.lower() in {'overall', 'mean', 'class', 'class-wise'}:
            raise ValueError("The `result_type` argument should be either 'overall', 'mean', or 'class-wise'.")

        # Import the confusion matrix attributes
        self.num_classes = num_classes
        super().__init__(self.num_classes, result_type='class',  name=name, dtype=dtype, **kwargs)
        self.result_type = result_type.lower()


    def update_state(self, y_true, y_pred, sample_weight=None):
        """"
        Computes the confusion matrix for the arguments specified.
        """
        if sample_weight is not None:
            raise ValueError("`sample_weight` is not None. "
                             "Be aware that the Recall "
                             "metric does not take `sample_weight` "
                             "into account when computing its value.")

        # Compute the confusion matrix.
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        """
        Computes the recall metric from the true positives and false negatives
        values from the confusion matrix computed by the update_state funtion.
        """

        if self.result_type in {'class', 'class-wise'}:
            recall = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.false_negatives))
            recall = tf.cast(recall, self._dtype)

        elif self.result_type == 'overall':
            recall = tf.math.divide_no_nan(tf.reduce_sum(self.true_positives),
                                        tf.reduce_sum((self.true_positives + self.false_negatives)))
            recall = tf.cast(recall, self._dtype)

        elif self.result_type == 'mean':
            recall = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.false_negatives))
            recall = tf.reduce_mean(tf.cast(recall, self._dtype))

        return recall

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        super().get_config()

    def reset_state(self):
        super().reset_state()

    def reset_states(self):
        super().reset_states()



class Specificity(ConfusionMatrix):
    """
    Computes the Specificity metric via the confusion matrix.
    Specificity is defined as follows:
    Specificity = true_negative / (true_negative + false_positive).
    If multi-class input is provided, it will be treated as multilabel data.
    Also, based on the specified arguments, this metric returns either:
    1. Overall specificity, or
    2. Class-wise specificity, or
    3. Mean specificity.

    Consider classification problem with two classes
    (i.e num_classes=2).
    The output of this metric would be:
    1. a float value for both 'overall' and 'mean' specificity, or
    2. (num_classes, 1) array for 'class-wise' specificity. Each value of
        the array represents the specificity of each class.

    Arguments:
        num_classes: `int`, the number of labels/classes the prediction task can have.
        result_type: one of 'overall', 'mean', or 'class-wise'
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:
        >>> # overall specificity
        >>> from metrics import Specificity
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = Specificity(num_classes=2, result_type='overall')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        1.0
        >>> # mean specificity
        >>> from metrics import Specificity
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = Specificity(num_classes=2, result_type='mean')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        1.0
        >>> # class-wise specificity
        >>> from metrics import Specificity
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = Specificity(num_classes=2, result_type='class-wise')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        array([1., 1.], dtype=float32)

    Usage with `compile()` API:
        ```python
        from metrics import Specificity
        specificity = Specificity(num_classes=3, result_type='mean', name='specificity')
        model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[specificity])
        """

    def __init__(self, num_classes, result_type='mean', name='specificity', dtype=None, **kwargs):
        # Validate class arguments/attributes
        if not isinstance(num_classes, int):
            raise ValueError("The `num_classes` argument should be an integer.")

        if not result_type.lower() in {'overall', 'mean', 'class', 'class-wise'}:
            raise ValueError("The `result_type` argument should be either 'overall', 'mean', or 'class-wise'.")

        # Import the confusion matrix attributes
        self.num_classes = num_classes
        super().__init__(self.num_classes, result_type='class',  name=name, dtype=dtype, **kwargs)
        self.result_type = result_type.lower()


    def update_state(self, y_true, y_pred, sample_weight=None):
        """"
        Computes the confusion matrix for the arguments specified.
        """
        if sample_weight is not None:
            raise ValueError("`sample_weight` is not None. "
                             "Be aware that the Specificity "
                             "metric does not take `sample_weight` "
                             "into account when computing its value.")

        # Compute the confusion matrix.
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        """
        Computes the specificity metric from the true negatives and false positives
        values from the confusion matrix computed by the update_state funtion.
        """

        if self.result_type in {'class', 'class-wise'}:
            specificity = tf.math.divide_no_nan(self.true_negatives,
                                        (self.true_negatives + self.false_positives))
            specificity = tf.cast(specificity, self._dtype)

        elif self.result_type == 'overall':
            specificity = tf.math.divide_no_nan(tf.reduce_sum(self.true_negatives),
                                        tf.reduce_sum((self.true_negatives + self.false_positives)))
            specificity = tf.cast(specificity, self._dtype)

        elif self.result_type == 'mean':
            specificity = tf.math.divide_no_nan(self.true_negatives,
                                        (self.true_negatives + self.false_positives))
            specificity = tf.reduce_mean(tf.cast(specificity, self._dtype))

        return specificity

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        super().get_config()

    def reset_state(self):
        super().reset_state()

    def reset_states(self):
        super().reset_states()



class Sensitivity(ConfusionMatrix):
    """
    Computes the Sensitivity metric via the confusion matrix.
    Sensitivity is defined as follows:
    Sensitivity = true_positive / (true_positive + true_negative).
    If multi-class input is provided, it will be treated as multilabel data.
    Also, based on the specified arguments, this metric returns either:
    1. Overall Sensitivity, or
    2. Class-wise Sensitivity, or
    3. Mean Sensitivity.

    Consider classification problem with two classes
    (i.e num_classes=2).
    The output of this metric would be:
    1. a float value for both 'overall' and 'mean' sensitivity, or
    2. (num_classes, 1) array for 'class-wise' sensitivity. Each value of
        the array represents the sensitivity of each class.

    Arguments:
        num_classes: `int`, the number of labels/classes the prediction task can have.
        result_type: one of 'overall', 'mean', or 'class-wise'
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:
        >>> # overall sensitivity
        >>> from metrics import Sensitivity
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = Sensitivity(num_classes=2, result_type='overall')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        0.5
        >>> # mean sensitivity
        >>> from metrics import Sensitivity
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = Sensitivity(num_classes=2, result_type='mean')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        0.5
        >>> # class-wise sensitivity
        >>> from metrics import Sensitivity
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = Sensitivity(num_classes=2, result_type='class-wise')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        array([0.5, 0.5], dtype=float32)

    Usage with `compile()` API:
        ```python
        from metrics import Sensitivity
        sensitivity = Sensitivity(num_classes=3, result_type='mean', name='sensitivity')
        model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[sensitivity])
        """

    def __init__(self, num_classes, result_type='mean', name='sensitivity', dtype=None, **kwargs):
        # Validate class arguments/attributes
        if not isinstance(num_classes, int):
            raise ValueError("The `num_classes` argument should be an integer.")

        if not result_type.lower() in {'overall', 'mean', 'class', 'class-wise'}:
            raise ValueError("The `result_type` argument should be either 'overall', 'mean', or 'class-wise'.")

        # Import the confusion matrix attributes
        self.num_classes = num_classes
        super().__init__(self.num_classes, result_type='class',  name=name, dtype=dtype, **kwargs)
        self.result_type = result_type.lower()


    def update_state(self, y_true, y_pred, sample_weight=None):
        """"
        Computes the confusion matrix for the arguments specified.
        """
        if sample_weight is not None:
            raise ValueError("`sample_weight` is not None. "
                             "Be aware that the Sensitivity "
                             "metric does not take `sample_weight` "
                             "into account when computing its value.")

        # Compute the confusion matrix.
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        """
        Computes the sensitivity metric from the true positives and true negatives
        values from the confusion matrix computed by the update_state funtion.
        """

        if self.result_type in {'class', 'class-wise'}:
            sensitivity = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.true_negatives))
            sensitivity = tf.cast(sensitivity, self._dtype)

        elif self.result_type == 'overall':
            sensitivity = tf.math.divide_no_nan(tf.reduce_sum(self.true_positives),
                                        tf.reduce_sum((self.true_positives + self.true_negatives)))
            sensitivity = tf.cast(sensitivity, self._dtype)

        elif self.result_type == 'mean':
            sensitivity = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.true_negatives))
            sensitivity = tf.reduce_mean(tf.cast(sensitivity, self._dtype))

        return sensitivity

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        super().get_config()

    def reset_state(self):
        super().reset_state()

    def reset_states(self):
        super().reset_states()



class TDR(ConfusionMatrix):
    """
    Computes the True-Detection-Rate (TDR) metric via the confusion matrix.
    TDR is defined as follows:
    TDR = 1 - (false_negative / (true_positive + false_negative)).
    If multi-class input is provided, it will be treated as multilabel data.
    Also, based on the specified arguments, this metric returns either:
    1. Overall TDR, or
    2. Class-wise TDR, or
    3. Mean TDR.

    Consider classification problem with two classes
    (i.e num_classes=2).
    The output of this metric would be:
    1. a float value for both 'overall' and 'mean' TDR, or
    2. (num_classes, 1) array for 'class-wise' TDR. Each value of
        the array represents the TDR of each class.

    Arguments:
        num_classes: `int`, the number of labels/classes the prediction task can have.
        result_type: one of 'overall', 'mean', or 'class-wise'
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:
        >>> # overall TDR
        >>> from metrics import TDR
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = TDR(num_classes=2, result_type='overall')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        0.5
        >>> # mean TDR
        >>> from metrics import TDR
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = TDR(num_classes=2, result_type='mean')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        0.5
        >>> # class-wise TDR
        >>> from metrics import TDR
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = TDR(num_classes=2, result_type='class-wise')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        array([0.5, 0.5], dtype=float32)

    Usage with `compile()` API:
        ```python
        from metrics import TDR
        tdr = TDR(num_classes=3, result_type='mean', name='tdr')
        model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tdr])
        """

    def __init__(self, num_classes, result_type='mean', name='tdr', dtype=None, **kwargs):
        # Validate class arguments/attributes
        if not isinstance(num_classes, int):
            raise ValueError("The `num_classes` argument should be an integer.")

        if not result_type.lower() in {'overall', 'mean', 'class', 'class-wise'}:
            raise ValueError("The `result_type` argument should be either 'overall', 'mean', or 'class-wise'.")

        # Import the confusion matrix attributes
        self.num_classes = num_classes
        super().__init__(self.num_classes, result_type='class',  name=name, dtype=dtype, **kwargs)
        self.result_type = result_type.lower()


    def update_state(self, y_true, y_pred, sample_weight=None):
        """"
        Computes the confusion matrix for the arguments specified.
        """
        if sample_weight is not None:
            raise ValueError("`sample_weight` is not None. "
                             "Be aware that the TDR "
                             "metric does not take `sample_weight` "
                             "into account when computing its value.")

        # Compute the confusion matrix.
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        """
        Computes the TDR metric from the true positives and false negatives
        values from the confusion matrix computed by the update_state funtion.
        """

        if self.result_type in {'class', 'class-wise'}:
            tdr = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.true_negatives))
            tdr = tf.cast(tdr, self._dtype)

        elif self.result_type == 'overall':
            tdr = tf.math.divide_no_nan(tf.reduce_sum(self.true_positives),
                                        tf.reduce_sum((self.true_positives + self.true_negatives)))
            tdr = tf.cast(tdr, self._dtype)

        elif self.result_type == 'mean':
            tdr = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.true_negatives))
            tdr = tf.reduce_mean(tf.cast(tdr, self._dtype))

        return tdr

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        super().get_config()

    def reset_state(self):
        super().reset_state()

    def reset_states(self):
        super().reset_states()


class F1Score(ConfusionMatrix):
    """
    Computes the F1Score metric via the confusion matrix.
    F1Score is defined as follows:
    F1-Score = (2 * precision * recall)/(precision + recall).
    Where:
        Recall = true_positive / (true_positive + false_negative).
        Precision = true_positive / (true_positive + false_positive).
    If multi-class input is provided, it will be treated as multilabel data.
    Also, based on the specified arguments, this metric returns either:
    1. Overall F1Score, or
    2. Class-wise F1Score, or
    3. Mean F1Score.

    Consider classification problem with two classes
    (i.e num_classes=2).
    The output of this metric would be:
    1. a float value for both 'overall' and 'mean' F1Score, or
    2. (num_classes, 1) array for 'class-wise' F1Score. Each value of
        the array represents the F1Score of each class.

    Arguments:
        num_classes: `int`, the number of labels/classes the prediction task can have.
        result_type: one of 'overall', 'mean', or 'class-wise'
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:
        >>> # overall F1Score
        >>> from metrics import F1Score
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = F1Score(num_classes=2, result_type='overall')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        1.0
        >>> # mean F1Score
        >>> from metrics import F1Score
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = F1Score(num_classes=2, result_type='mean')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        1.0
        >>> # class-wise F1Score
        >>> from metrics import F1Score
        >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
        >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
        >>> m = F1Score(num_classes=2, result_type='class-wise')
        >>> m.update_state(y_true, y_pred)
        >>> m.result().numpy()
        array([1., 1.], dtype=float32)

    Usage with `compile()` API:
        ```python
        from metrics import F1Score
        f1_score = F1Score(num_classes=3, result_type='mean', name='f1_score')
        model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[f1_score])
        """

    def __init__(self, num_classes, result_type='mean', name='f1_score', dtype=None, **kwargs):
        # Validate class arguments/attributes
        if not isinstance(num_classes, int):
            raise ValueError("The `num_classes` argument should be an integer.")

        if not result_type.lower() in {'overall', 'mean', 'class', 'class-wise'}:
            raise ValueError("The `result_type` argument should be either 'overall', 'mean', or 'class-wise'.")

        # Import the confusion matrix attributes
        self.num_classes = num_classes
        super().__init__(self.num_classes, result_type='class',  name=name, dtype=dtype, **kwargs)
        self.result_type = result_type.lower()


    def update_state(self, y_true, y_pred, sample_weight=None):
        """"
        Computes the confusion matrix for the arguments specified.
        """
        if sample_weight is not None:
            raise ValueError("`sample_weight` is not None. "
                             "Be aware that the F1Score "
                             "metric does not take `sample_weight` "
                             "into account when computing its value.")

        # Compute the confusion matrix.
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        """
        Computes the F1Score metric from the true positives, false positives
        and false negatives values from the confusion matrix computed
        by the update_state funtion.
        """

        if self.result_type in {'class', 'class-wise'}:
            recall = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.false_negatives))
            recall = tf.cast(recall, self._dtype)
            precision = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.false_positives))
            precision = tf.cast(precision, self._dtype)
            f1_score = tf.math.divide_no_nan((2 * precision * recall), (precision + recall))
            f1_score = tf.cast(f1_score, self._dtype)

        elif self.result_type == 'overall':
            recall = tf.math.divide_no_nan(tf.reduce_sum(self.true_positives),
                                        tf.reduce_sum((self.true_positives + self.false_negatives)))
            recall = tf.cast(recall, self._dtype)
            precision = tf.math.divide_no_nan(tf.reduce_sum(self.true_positives),
                                        tf.reduce_sum((self.true_positives + self.false_positives)))
            precision = tf.cast(precision, self._dtype)
            f1_score = tf.math.divide_no_nan((2 * precision * recall), (precision + recall))
            f1_score = tf.cast(f1_score, self._dtype)

        elif self.result_type == 'mean':
            recall = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.false_negatives))
            recall = tf.reduce_mean(tf.cast(recall, self._dtype))
            precision = tf.math.divide_no_nan(self.true_positives,
                                        (self.true_positives + self.false_positives))
            precision = tf.reduce_mean(tf.cast(precision, self._dtype))
            f1_score = tf.math.divide_no_nan((2 * precision * recall), (precision + recall))
            f1_score = tf.cast(f1_score, self._dtype)

        return f1_score

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        super().get_config()

    def reset_state(self):
        super().reset_state()

    def reset_states(self):
        super().reset_states()




#########-------- Classification Report --------#########
import numpy as np
import pandas as pd
from tabulate import tabulate

def classification_report(y_true,
                          y_pred,
                          num_classes,
                          metrics=["TP", "TN", "FP", "FN", "Recall", "Precision", "Specificity", "Sensitivity", "IoU", "TDR", "F1Score"],
                          target_names=None,
                          report_type='table',
                          table_type='pandas-dataframe',
                          show_mean_evaluations=True,
                          show_classwise_evaluations=True,
                          decimals = 2):

    """
    Evaluates image segmentation model and return classification report.

    Arguments:
        y_true: ground truth segmentations. 4D numpy array or tensorflow tensor.
        y_pred: predicted segmentations. 4D numpy array or tensorflow tensor.
        num_classes: number of segmentation classes.
        metrics: a list containing one or more of the following metrics:
                 'TP', 'TN', 'FP', 'FN', 'Recall', 'Precision', 'Specificity', 'Sensitivity'
                 'F1 Score', 'IoU', 'TDR'. Default list displays all of the aforementioned metrics.
        target_names: (optional) a list or tuple containing names of the segmentation classes.
        report_type: (optional) one of 'dictionary', 'table' or 'array'. Default is table.
        table_type: (optional) one of 'pandas-dataframe' (to display output as a pandas dataframe),
                             'or tabulate-table (to display output as a tabulate table).  Default
                             is pandas-dataframe.
        show_mean_evaluations: (optional) one of True (to display both mean evaluations),
                             'or False (to not display both mean evaluations). Default is True.
        show_classwise_evaluations: (optional) one of True (to display class-wise evaluations),
                             'or False (to display only the overall evaluations). Default is True.
        decimals: (optional) integer value. Number of decimal places to round to (Default is 2).
                 If decimals is negative, it specifies the number of positions to the left of the
                 decimal point. of how many decimal places the report values should have.

    Returns:
        A pandas dataframe or tabulate table containing output of the evaluation metrics specified.
        """

    #####----- Validate and preprocess arguments  ------#####
    # 1. y_true and y_pred
    # Predefined validation arguments in the ConfusionMatrix Class will
    # validate the y_true and y_pred arguments specified here.
    # Hence, we will only be converting both arguments to tensor as a preprocessing step.
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    # 2. number of classes specified
    if num_classes < 2:
        raise ValueError("The `num_classes` argument cannot be less than 2.")
    elif not(isinstance(num_classes, int)):
        raise ValueError("The `num_classes` argument can only be an integer.")

    # 3. metrics
    for metric_id in range(len(metrics)):
        metric = metrics[metric_id]
        if metric not in ["TP", "TN", "FP", "FN", "Recall", "Precision", "Specificity", "Sensitivity", "IoU", "TDR", "F1Score"]:
            raise ValueError("The `metrics` argument should be a list containing one or more of the following metrics: "
                             "'TP', 'TN', 'FP', 'FN', 'Recall', 'Precision', "
                             "'Specificity', 'Sensitivity', 'IoU', 'TDR', 'F1Score'")

    # 4. target_names
    if target_names == None:
        target_names = ["Class " + str(class_id+1) for class_id in range(num_classes)]
    else:
        if not(isinstance(target_names, list) or isinstance(target_names, tuple)):
            raise ValueError("The `target_names` argument should be None (display default class names) "
                             "or a list/tuple containing the class names.")
        elif not all(isinstance(name, str) for name in target_names):
            raise ValueError("The `target_names` argument list items should be string.")
        elif len(target_names) != num_classes:
            raise ValueError("The numer of class names specified in the `target_names` argument list/tuple "
                             "should match with the `num_classes` argument value.")

    # 5. report_type='table'
    if not report_type in {'array','dictionary','table'}:
        raise ValueError("The `report_type` argument should be one of  'array', 'dictionary', or 'table'.")

    # 6. table type
    if not table_type in {'pandas-dataframe', 'tabulate-table'}:
        raise ValueError("The `table_type` argument can only be either 'pandas-dataframe' or 'tabulate-table'.")

    # 7. show_mean_evaluations
    if not show_mean_evaluations in {True, False}:
        raise ValueError("The `show_mean_evaluations` argument can only be either True or False.")

    # 8. show_classwise_evaluations
    if not show_classwise_evaluations in {True, False}:
        raise ValueError("The `show_classwise_evaluations` argument can only be either True or False.")

    # 9. decimals
    if not(isinstance(decimals, int)):
        raise ValueError("The `decimals` argument can only be an integer.")

    #####----- Evaluate the model performance  ------#####
    evaluations = {}
    result_types = ['overall', 'mean', 'class-wise']
    ## Confusion Matrix (True Positive, True Negative, False Positive, False Negative)
    for result_type in result_types:
        if result_type == 'mean':
            true_negatives = '-'
            false_positives = '-'
            false_negatives = '-'
            true_positives = '-'
        else:
            matrix = ConfusionMatrix(num_classes=num_classes, result_type=result_type, dtype=tf.int32)
            matrix.update_state(y_true, y_pred)
            confusion_matrix = matrix.result().numpy()
            # Extract TP, TN, FP, and FN from the confusion matrix
            true_negatives = confusion_matrix[:,0,0]
            false_positives = confusion_matrix[:,0,1]
            false_negatives = confusion_matrix[:,1,0]
            true_positives = confusion_matrix[:,1,1]
        # Recall
        recall = Recall(num_classes=num_classes, result_type=result_type, dtype=tf.float32)
        recall.update_state(y_true, y_pred)
        recalls = np.around(recall.result().numpy(), decimals)
        # Precision
        precision = Precision(num_classes=num_classes, result_type=result_type, dtype=tf.float32)
        precision.update_state(y_true, y_pred)
        precisions = np.around(precision.result().numpy(), decimals)
        # Specificity
        specificity = Specificity(num_classes=num_classes, result_type=result_type, dtype=tf.float32)
        specificity.update_state(y_true, y_pred)
        specificities = np.around(specificity.result().numpy(), decimals)
        # Sensitivity
        sensitivity = Sensitivity(num_classes=num_classes, result_type=result_type, dtype=tf.float32)
        sensitivity.update_state(y_true, y_pred)
        sensitivities = np.around(sensitivity.result().numpy(), decimals)
        # IoU
        iou = IoU(num_classes=num_classes, result_type=result_type, dtype=tf.float32)
        iou.update_state(y_true, y_pred)
        ious = np.around(iou.result().numpy(), decimals)
        # TDR
        tdr = TDR(num_classes=num_classes, result_type=result_type, dtype=tf.float32)
        tdr.update_state(y_true, y_pred)
        tdrs = np.around(tdr.result().numpy(), decimals)
        # F1Score
        f1_score = F1Score(num_classes=num_classes, result_type=result_type, dtype=tf.float32)
        f1_score.update_state(y_true, y_pred)
        f1_scores = np.around(f1_score.result().numpy(), decimals)



        evaluations[result_type] = {"TP": true_positives, # True Positive Pixels
                                    "TN": true_negatives, # True Negative Pixels
                                    "FP": false_positives, # False Positive Pixels
                                    "FN": false_negatives, # False Negative Pixels
                                    "Recall": recalls,
                                    "Precision": precisions,
                                    "Specificity": specificities,
                                    "Sensitivity": sensitivities,
                                    "IoU": ious,
                                    "TDR": tdrs,
                                    "F1Score": f1_scores}

    # Store outputs
    if report_type == 'dictionary':
        report = {}
        report['overall'] = ["Overall"] + evaluations['overall']
        if show_mean_evaluations:
            report['mean'] = ["Mean"] + evaluations['mean']
        if show_classwise_evaluations:
            report['class-wise'] = [target_names] + evaluations['class-wise']

    else:
        overall_data = ["Overall"] + [evaluations['overall'][metrics[metric_id]][0]
                                      if isinstance(evaluations['overall'][metrics[metric_id]], np.ndarray)
                                      else evaluations['overall'][metrics[metric_id]]
                                      for metric_id in range(len(metrics))]
        mean_data = ["Mean"] + [evaluations['mean'][metrics[metric_id]]
                                if not isinstance(evaluations['mean'][metrics[metric_id]], np.ndarray)
                                else evaluations['mean'][metrics[metric_id]][0] for metric_id in range(len(metrics))]
        classwise_data = [target_names] + [evaluations['class-wise'][metrics[metric_id]] for metric_id in range(len(metrics))]
        overall_data = np.array(overall_data).reshape(1,-1)
        mean_data = np.array(mean_data).reshape(1,-1)
        classwise_data = np.array(classwise_data).transpose()

        if report_type=='array':

            report = overall_data
            if show_mean_evaluations:
                report = np.concatenate((report, mean_data), axis=0)
            if show_classwise_evaluations:
                report = np.concatenate((report, classwise_data), axis=0)

        else:
            report_data = overall_data
            if show_mean_evaluations:
                report_data = np.concatenate((report_data, mean_data), axis=0)
            if show_classwise_evaluations:
                report_data = np.concatenate((report_data, classwise_data), axis=0)

            report_header = np.array(["Evaluation"] + metrics)

            if table_type == 'pandas-dataframe':
                report = pd.DataFrame(report_data)
                report.columns = report_header
            else:
                report = print(tabulate(report_data, headers=report_header, tablefmt='fancy_grid'))

    return report
