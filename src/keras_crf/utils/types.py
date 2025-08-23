# Lightweight type aliases for the standalone CRF package
# These avoid importing tensorflow_addons.utils.types

from typing import Union, Any
import numpy as np
import tensorflow as tf

# Keras initializer can be specified as string, config dict, or initializer instance
Initializer = Union[str, dict, tf.keras.initializers.Initializer]

# Tensor-like values accepted by ops
TensorLike = Union[tf.Tensor, tf.Variable, np.ndarray, Any]

