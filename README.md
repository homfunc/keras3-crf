# Keras3-CRF: Standalone CRF layer and ops for Keras 3

This package provides a lightweight CRF layer and supporting ops extracted from TensorFlow Addons, updated for Keras 3 backend independence. The distribution name is keras3-crf on PyPI; the import path remains keras_crf.

Features
- Linear-chain CRF decoding (Viterbi)
- Log-likelihood and training via gradients
- Masking support (right-padding)
- Minimal API surface: `keras_crf.CRF` layer and `keras_crf.crf_ops` ops

Quickstart (simple TF loop)
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

High-level training helper (backend-agnostic)
Use make_crf_tagger to attach a CRF head that’s ready to train with Model.fit while keeping decoded output separate from the NLL loss head.

```python
import numpy as np
import keras
from keras import layers
from keras_crf.train_utils import make_crf_tagger, prepare_crf_targets

# Build encoder
vocab_size, num_tags = 1000, 5
tokens = keras.Input(shape=(None,), dtype='int32', name='tokens')
x = layers.Embedding(vocab_size + 1, 64, mask_zero=True)(tokens)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

# Build CRF tagger model
model = make_crf_tagger(tokens, x, num_tags)

# Fit
X = np.random.randint(1, vocab_size+1, size=(64, 20), dtype=np.int32)
Y = np.random.randint(0, num_tags, size=(64, 20), dtype=np.int32)
mask = (X != 0).astype('float32')
y_dict, sw_dict = prepare_crf_targets(Y, mask)
model.fit({'tokens': X, 'labels': Y}, y_dict, sample_weight=sw_dict, epochs=1)

# Predict decoded tags
decoded = model.get_layer('decoded_output').output  # or use a separate inference Model
```

Notes
- Backend-agnostic via Keras 3 universal ops: runs with TensorFlow, Torch, or JAX backends.
- Eager-first. Validated primarily in eager mode; works with Keras Model/fit across backends. TF graph mode (tf.function) is supported only for the legacy TF ops in `keras_crf.text`.
- Masking requires right-padding (ones for valid timesteps followed by zeros). Left-padding is not supported.
- Decode padding: positions beyond the true sequence length are deterministically filled by copying the last valid tag.
- Single-timestep sequences (time == 1): decode is argmax at t=0; log-likelihood reduces to the unary term minus logsumexp.

Backend independence
- Core ops and the CRF layer now use Keras 3 universal ops and are backend-agnostic (TensorFlow, PyTorch, JAX).
- Select backend by setting KERAS_BACKEND environment variable before import:
  - bash example: export KERAS_BACKEND=tensorflow (or torch, or jax)
- Install optional backend packages as needed:
  - TensorFlow: pip install .[tf]
  - PyTorch: pip install .[torch]
  - JAX (CPU): pip install .[jax]
- TensorFlow-specific ops and layer live in the separate tf_keras_crf package. Install tf-keras-crf and import `tf_keras_crf.text` and `tf_keras_crf.CRF` if you need TF-native behavior.

Automatic backend selection (optional)
You can auto-select the Keras backend based on what’s installed (priority: jax > tensorflow > torch) by creating a sitecustomize.py file on your PYTHONPATH (Python will import it automatically on startup). For example, place this file into your virtualenv’s site-packages or your project root and ensure it’s on sys.path:

```
# sitecustomize.py
import os, importlib.util

if "KERAS_BACKEND" not in os.environ:
    for mod in ("jax", "tensorflow", "torch"):
        if importlib.util.find_spec(mod) is not None:
            os.environ["KERAS_BACKEND"] = {"jax": "jax", "tensorflow": "tensorflow", "torch": "torch"}[mod]
            break
```

Notes:
- This runs before any user code and before `import keras`, so the backend is set in time.
- If KERAS_BACKEND is already set in the environment, this script will not override it.

Installation
- Python >= 3.10 supported.
- From PyPI: pip install keras3-crf
- From source (editable): pip install -e .

Testing
```bash
pytest -q tests
```

Backend snippets
- See `examples/torch_jax_snippet.md` for a minimal Torch/JAX example using the CRF layer with Keras 3 universal ops.

