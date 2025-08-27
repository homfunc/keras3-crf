# Keras-CRF Examples (Keras Core, backend-agnostic)

These examples run with any Keras 3 backend: TensorFlow, PyTorch, or JAX.

Setup
- Install the package from the repo root (dev tools include nox):
  - pip install -e .[dev]
- Install one backend of your choice (if running examples directly):
  - TensorFlow: pip install tensorflow
  - PyTorch: pip install torch
  - JAX: pip install "jax"
- Or install extras for notebooks: pip install -e .[examples]
- Or use nox to handle backend extras in an isolated session:
  - nox -s tests -- backend=tensorflow  # runs tests with TF
  - nox -s quickstarts -- backend=torch # runs the Torch quickstart
- Select the backend before importing Keras:
  - export KERAS_BACKEND=tensorflow   # or torch, or jax

BiLSTM-CRF trainer (synthetic dataset)
- python examples/bilstm_crf_train.py --dataset synthetic --epochs 3 --batch-size 64
- You can choose the training loss via --loss:
  - --loss nll         # CRF negative log-likelihood (default)
  - --loss dice        # token-level Dice loss using CRF marginals (forward-backward)
  - --loss dice+nll    # joint: alpha*NLL + (1-alpha)*Dice (use --joint-nll-weight to set alpha)

CoNLL/CleanCoNLL and MultiCoNER datasets
- Sample toy files are in examples/data (tiny; for smoke tests only).
- CleanCoNLL (recommended for a better demo):
  - Option A: use our helper to fetch/prepare splits:
    - python examples/data/cleanconll_prepare.py --output-dir examples/data/cleanconll
  - Option B: clone manually and point to the cleaned splits.
  - Then run the trainer on the produced files:
    - python examples/bilstm_crf_train.py \
        --dataset conll \
        --train examples/data/cleanconll/train.txt \
        --val   examples/data/cleanconll/valid.txt \
        --test  examples/data/cleanconll/test.txt \
        --token-col 0 --tag-col -1 --scheme BIO --epochs 5 --loss nll
- MultiCoNER (EN subset):
  - Option A: if you placed a zip (e.g., EN-English.zip) under examples/data, extract it:
    - python examples/data/multiconer_prepare.py --zip examples/data/EN-English.zip --output-dir examples/data/multiconer/EN-English
  - Option B: copy from your local directory (e.g., ../multiconer2023/EN-English):
    - python examples/data/multiconer_prepare.py --source-dir ../multiconer2023/EN-English --output-dir examples/data/multiconer/EN-English
  - Then run the trainer:
    - python examples/bilstm_crf_train.py --dataset multiconer --mc-dir examples/data/multiconer/EN-English --epochs 5 --loss dice+nll

Manual, no-helper example (showing custom loss/metrics wiring)
- python examples/bilstm_crf_manual.py --dataset synthetic --loss dice+nll --joint-nll-weight 0.2
- This example builds the CRF head and loss layer(s) directly without using train_utils,
  so you can see where to plug in your own loss and metric functions.

Quickstarts
- Torch: python examples/quickstart_torch.py
- JAX: python examples/quickstart_jax.py
- TensorFlow: python examples/quickstart_tf.py

Helper API
- The quickstarts and trainer use keras_crf.train_utils.make_crf_tagger to attach a CRF training head with two outputs:
  - decoded_output: per-token predictions used for metrics
  - crf_log_likelihood_output: per-sample training loss (named for back-compat; may represent NLL, CRF-marginal Dice, or joint)
- Use keras_crf.train_utils.prepare_crf_targets(y_true, mask) to build the y and sample_weight dicts for Model.fit.

Alternative data loaders
- PyTorch DataLoader: python examples/data_pytorch_dataloader.py
- Grain (if installed): python examples/data_grain.py

Notes
- All training loops and metrics use Keras Core universal ops; no tf.data or TF-only APIs are required.
- Masking uses 0 as PAD in token IDs; metrics accept sample_weight masks produced from tokens != 0.
- For GPU support, follow official backend installation docs to choose the correct wheels.
- Existing notebooks (BiLSTM_CRF*.ipynb, Torch_JAX_KerasCore_CRF.ipynb) are legacy and may contain older APIs. Prefer the Python scripts here (bilstm_crf_train.py, bilstm_crf_manual.py) for current usage.
