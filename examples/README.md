# Keras-CRF Examples (Keras Core, backend-agnostic)

These examples run with any Keras 3 backend: TensorFlow, PyTorch, or JAX.

Setup
- Install the package from the repo root (dev tools include nox):
  - pip install -e .[dev]
- Install one backend of your choice (if running examples directly):
  - TensorFlow: pip install tensorflow
  - PyTorch: pip install torch
  - JAX (CPU): pip install "jax[cpu]"
- Or use nox to handle backend extras in an isolated session:
  - nox -s tests -- backend=tensorflow  # runs tests with TF
  - nox -s quickstarts -- backend=torch # runs the Torch quickstart
- Select the backend before importing Keras:
  - export KERAS_BACKEND=tensorflow   # or torch, or jax

BiLSTM-CRF trainer (synthetic dataset)
- python examples/bilstm_crf_train.py --dataset synthetic --epochs 3 --batch-size 64

CoNLL dataset
- Sample files are in examples/data.
- Example:
  - python examples/bilstm_crf_train.py \
      --dataset conll \
      --train examples/data/sample.conll \
      --val   examples/data/sample.conll \
      --test  examples/data/sample.conll \
      --token-col 0 --tag-col -1 --scheme BIO --epochs 3

Quickstarts
- Torch: python examples/quickstart_torch.py
- JAX (CPU): python examples/quickstart_jax.py
- TensorFlow: python examples/quickstart_tf.py

Helper API
- The quickstarts and trainer use keras_crf.train_utils.make_crf_tagger to attach a CRF training head with two outputs:
  - decoded_output: per-token predictions used for metrics
  - crf_log_likelihood_output: per-sample negative log-likelihood used for training loss
- Use keras_crf.train_utils.prepare_crf_targets(y_true, mask) to build the y and sample_weight dicts for Model.fit.

Alternative data loaders
- PyTorch DataLoader: python examples/data_pytorch_dataloader.py
- Grain (if installed): python examples/data_grain.py

Notes
- All training loops and metrics use Keras Core universal ops; no tf.data or TF-only APIs are required.
- Masking uses 0 as PAD in token IDs; metrics accept sample_weight masks produced from tokens != 0.
- For GPU support, follow official backend installation docs to choose the correct wheels.
