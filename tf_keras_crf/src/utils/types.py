# Lightweight type aliases without hard dependency on TensorFlow
from typing import Union
import numpy as np
import tensorflow as tf

# Keras initializer can be specified as string, config dict, or initializer instance
Initializer = Union[str, dict, tf.keras.initializers.Initializer]

# Tensor-like values accepted by ops (backend-agnostic)
TensorLike = Union[np.ndarray, tf.Tensor]

