# Keras Core CRF on Torch or JAX

This short example shows how to run the CRF layer with Keras 3 universal ops on non-TensorFlow backends.

- Choose a backend before you import Keras:
  - bash: `export KERAS_BACKEND=torch` (or `jax`)
- Install backend packages if needed:
  - Torch: `pip install .[torch]`
  - JAX (CPU): `pip install .[jax]`

```python
# backend_snippet.py
import os
# Uncomment to select a backend from code (prefer environment variable in practice)
# os.environ["KERAS_BACKEND"] = "torch"  # or "jax"

import numpy as np
import keras
from keras import layers, ops as K
from keras_crf import CRF

# Simple model: Embedding -> BiLSTM -> CRF
vocab_size = 100
num_tags = 5
max_len = 10

inputs = keras.Input(shape=(max_len,), dtype="int32")
masking = layers.Embedding(vocab_size + 1, 16, mask_zero=True)(inputs)
seq = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(masking)
crf = CRF(units=num_tags)
decoded, potentials, seq_len, kernel = crf(seq)
model = keras.Model(inputs, [decoded, potentials, seq_len, kernel])

# Dummy data
X = np.random.randint(1, vocab_size, (4, max_len)).astype("int32")
Y = np.random.randint(0, num_tags, (4, max_len)).astype("int32")

# Forward pass
decoded_out, potentials_out, seq_len_out, kernel_out = model(X)
print("decoded shape:", K.shape(decoded_out))

# Compute a loss using backend-agnostic CRF ops
ll = crf.log_likelihood(potentials_out, Y, seq_len_out)
print("mean negative log-likelihood:", float(K.mean(-ll)))
```

