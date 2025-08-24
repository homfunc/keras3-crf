import keras
from keras import ops as K


class MaskedTokenAccuracy(keras.metrics.Metric):
    """Backend-agnostic token accuracy with optional mask using Keras Core.

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
        y_true = K.cast(y_true, "int32")
        y_pred = K.cast(y_pred, "int32")
        equal = K.cast(K.equal(y_true, y_pred), "float32")
        if sample_weight is not None:
            sw = K.cast(sample_weight, "float32")
            equal = equal * sw
            total = K.sum(sw)
        else:
            total = K.cast(K.size(equal), "float32")
        correct = K.sum(equal)
        # assign_add with backend scalars to support JAX and Torch tracing
        self.correct.assign_add(K.cast(correct, "float32"))
        self.total.assign_add(K.cast(total, "float32"))

    def result(self):
        # safe divide
        denom = self.total
        return self.correct / (denom + K.cast(1e-8, self.correct.dtype))

    def reset_states(self):
        self.correct.assign(K.zeros_like(self.correct))
        self.total.assign(K.zeros_like(self.total))
