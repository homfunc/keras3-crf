import tensorflow as tf


class MaskedTokenAccuracy(tf.keras.metrics.Metric):
    """Computes token-level accuracy with an optional mask.

    update_state(y_true, y_pred, sample_weight=None)
    - y_true: int tensor [batch, time]
    - y_pred: int tensor [batch, time] (decoded tags)
    - sample_weight: optional float/bool mask [batch, time]
    """

    def __init__(self, name="token_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        equal = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            # Broadcast if needed
            equal = equal * sw
            total = tf.reduce_sum(sw)
        else:
            total = tf.cast(tf.size(equal), tf.float32)
        correct = tf.reduce_sum(equal)
        self.correct.assign_add(correct)
        self.total.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_states(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)
