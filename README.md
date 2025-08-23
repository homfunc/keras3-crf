# Keras3-CRF: Standalone CRF layer and ops for Keras 3

This package provides a lightweight CRF layer and supporting ops extracted from TensorFlow Addons, updated for Keras 3 backend independence. The distribution name is keras3-crf on PyPI; the import path remains keras_crf.

Features
- Linear-chain CRF decoding (Viterbi)
- Log-likelihood and training via gradients
- Masking support (right-padding)
- Minimal API surface: `keras_crf.CRF` layer and `keras_crf.text` ops

Quickstart
```python
import numpy as np
import tensorflow as tf
from keras_crf import CRF

# logits: [batch, time, num_tags]
logits = np.random.randn(2, 5, 4).astype('float32')
tags = np.random.randint(0, 4, size=(2, 5)).astype('int32')

crf = CRF(units=4)
decoded, potentials, seq_len, kernel = crf(logits)

# loss
ll = crf.log_likelihood(potentials, tags, seq_len)
loss = -tf.reduce_mean(ll)

# train
opt = tf.keras.optimizers.Adam(1e-3)
with tf.GradientTape() as tape:
    _, potentials, seq_len, kernel = crf(logits)
    ll = crf.log_likelihood(potentials, tags, seq_len)
    loss = -tf.reduce_mean(ll)
opt.apply_gradients(zip(tape.gradient(loss, crf.trainable_variables), crf.trainable_variables))
```

Notes
- Designed for eager mode. Building Functional graphs with KerasTensors is not the target.
- Left-padding masks are not supported; right-padding masks are supported via sequence lengths.
- For single-timestep sequences (time == 1), decode and loss follow the simplified code path (argmax and unary log-sum-exp).

Backend independence
- Core ops and the CRF layer now use Keras 3 universal ops and are backend-agnostic (TensorFlow, PyTorch, JAX).
- Select backend by setting KERAS_BACKEND environment variable before import:
  - bash example: export KERAS_BACKEND=tensorflow (or torch, or jax)
- Install optional backend packages as needed:
  - TensorFlow: pip install .[tf]
  - PyTorch: pip install .[torch]
  - JAX (CPU): pip install .[jax]
- Legacy TF-only ops remain available under keras_crf.text for compatibility; the CRF layer does not depend on them.

Installation
- From PyPI: pip install keras3-crf
- From source (editable): pip install -e .

Testing
```bash
pytest -q tests
```

Backend snippets
- See `examples/torch_jax_snippet.md` for a minimal Torch/JAX example using the CRF layer with Keras 3 universal ops.

