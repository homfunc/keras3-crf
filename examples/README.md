# Keras-CRF Examples (Keras Core, backend-agnostic)

These examples run with any Keras 3 backend: TensorFlow, PyTorch, or JAX.

Setup
- Install the package from the repo root:
  - pip install -e .
- Install one backend of your choice:
  - TensorFlow: pip install tensorflow
  - PyTorch: pip install torch
  - JAX (CPU): pip install "jax[cpu]"
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

Notes
- All training loops and metrics use Keras Core universal ops; no tf.data or TF-only APIs are required.
- Masking uses 0 as PAD in token IDs; metrics accept sample_weight masks produced from tokens != 0.
- For GPU support, follow official backend installation docs to choose the correct wheels.
