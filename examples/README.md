# Keras-CRF Examples

This directory contains example notebooks and scripts for training a BiLSTM-CRF tagger using the standalone `keras_crf` package.

Backend independence
- Notebooks and examples prefer Keras 3 universal ops. You can choose a backend by setting KERAS_BACKEND to tensorflow, torch, or jax before running Python.
- The BiLSTM_CRF_Core.ipynb notebook demonstrates backend-agnostic training loops.
- The bilstm_crf_train.py CLI currently uses a TensorFlow-based custom training step.

## Setup

Install examples dependencies:

```
pip install -r examples/requirements-examples.txt
```

Install the package (from repo root) and optional backend extras if needed:

```
# Core package
pip install -e .

# Optional: install a backend implementation (choose as needed)
# TensorFlow
pip install .[tf]
# PyTorch
pip install .[torch]
# JAX (CPU)
pip install .[jax]
```

## Notebooks

- `BiLSTM_CRF_Core.ipynb`: backend-agnostic example using Keras 3 universal ops (recommended).
- `BiLSTM_CRF.ipynb`: simple fixed-length synthetic dataset training & evaluation (TensorFlow).
- `BiLSTM_CRF_Advanced.ipynb`: variable-length sequences (masking) and CoNLL-style loaders (TensorFlow).

## CLI Script

`examples/bilstm_crf_train.py` trains a BiLSTM-CRF model on synthetic or CoNLL data (TensorFlow-specific training loop).

### Synthetic

```
python examples/bilstm_crf_train.py --dataset synthetic --epochs 3 --batch-size 64
```

Arguments:
- `--synthetic-max-len`, `--synthetic-vocab`, `--synthetic-tags`, `--synthetic-samples`

### CoNLL

Sample files:
- `examples/data/sample.conll` (BIO-style)
- `examples/data/sample_bilou.conll` (BILOU-style)

```
python examples/bilstm_crf_train.py \
  --dataset conll \
  --train /path/train.conll \
  --val   /path/valid.conll \
  --test  /path/test.conll \
  --token-col 0 --tag-col -1 --scheme BIO --epochs 5
```

- CoNLL format: whitespace-separated columns per token line. Blank line between sentences. `--token-col` and `--tag-col` select columns (default: token=0, tag=last). `-DOCSTART-` lines are skipped.
- Tag scheme supported for entity F1: `BIO` (default) or `BILOU`.

## Metrics

- Token accuracy (masked): counts only tokens where input != 0.
- Entity F1 (BIO/BILOU): span-level F1 over predicted vs. gold entities.

## Torch/JAX environment notes

- Set the backend before importing Keras:
  - bash: `export KERAS_BACKEND=torch` or `export KERAS_BACKEND=jax`
- Installation guidance (GPU optional):
  - Torch: follow https://pytorch.org/get-started/locally/ for the correct wheel (CPU or CUDA). Our optional extra `. [torch]` installs a CPU build by default.
  - JAX: see https://jax.readthedocs.io/en/latest/installation.html. Our optional extra `. [jax]` installs a CPU build by default. For GPU, use the official instructions to pick the right CUDA wheel.
- Known constraints:
  - The CLI (bilstm_crf_train.py) uses TensorFlow-specific input pipelines (tf.data) and TFRecord utilities; use the numpy-based code paths if running under Torch/JAX.
  - Legacy tests under `tests/test_text.py` are TF-only and unrelated to the backend-agnostic CRF layer.

## Tips

- Use `--run-eagerly` for debugging train/test steps.
- For large datasets, consider using tf.data to stream batches.
- If using pre-trained embeddings, replace the Embedding layer accordingly.
