import numpy as np
import keras
from keras import ops as K

from keras_crf import CRF
from keras_crf.losses import nll_loss, dice_loss, joint_dice_nll_loss
from keras_crf.losses import CRFNLLLoss, CRFDiceLoss, CRFNLLHead, CRFDiceHead


def _rand_potentials(B=3, T=5, C=4, seed=0):
    rng = np.random.default_rng(seed)
    return K.convert_to_tensor(rng.normal(size=(B, T, C)).astype("float32"))


def _rand_transitions(C=4, seed=1):
    rng = np.random.default_rng(seed)
    return K.convert_to_tensor(rng.normal(size=(C, C)).astype("float32"))


def _full_lengths(B, T):
    return K.full((B,), T, dtype="int32")


def test_public_losses_shapes_and_dtypes():
    B, T, C = 2, 6, 5
    pot = _rand_potentials(B, T, C)
    trans = _rand_transitions(C)
    y = K.convert_to_tensor(np.random.randint(0, C, size=(B, T)).astype("int32"))
    lens = _full_lengths(B, T)

    # Mean reduction -> scalar
    nll = nll_loss(pot, lens, trans, y, reduction="mean")
    dce = dice_loss(pot, lens, trans, y, reduction="mean")
    jnt = joint_dice_nll_loss(pot, lens, trans, y, alpha=0.3, reduction="mean")
    assert K.ndim(nll) == 0
    assert K.ndim(dce) == 0
    assert K.ndim(jnt) == 0

    # None reduction -> [B]
    nll_v = nll_loss(pot, lens, trans, y, reduction="none")
    dce_v = dice_loss(pot, lens, trans, y, reduction="none")
    assert K.shape(nll_v)[0] == B
    assert K.shape(dce_v)[0] == B


def _build_simple_model_heads(T=6, V=20, C=5, H=16, use_mask=False):
    tokens = keras.Input(shape=(T,), dtype="int32", name="tokens")
    y_true = keras.Input(shape=(T,), dtype="int32", name="labels")
    x = keras.layers.Embedding(V + 1, 16, mask_zero=use_mask)(tokens)
    x = keras.layers.Bidirectional(keras.layers.LSTM(H, return_sequences=True))(x)
    crf = CRF(C)
    decoded, potentials, lens, trans = crf(x)
    nll_vec = CRFNLLHead(name="nll_head")([potentials, y_true, lens, trans])
    dice_vec = CRFDiceHead(name="dice_head", smooth=1.0)([potentials, y_true, lens, trans])
    model_nll = keras.Model({"tokens": tokens, "labels": y_true}, nll_vec)
    model_dice = keras.Model({"tokens": tokens, "labels": y_true}, dice_vec)
    return model_nll, model_dice, crf


def test_loss_class_trains_and_updates_transitions_nll():
    B, T, V, C = 8, 6, 30, 5
    model_nll, _model_dice, crf = _build_simple_model_heads(T=T, V=V, C=C, use_mask=False)
    # Loss reducer expects per-sample vector from the model
    loss_inst = CRFNLLLoss()
    model_nll.compile(optimizer=keras.optimizers.Adam(5e-3), loss=loss_inst, run_eagerly=True)

    rng = np.random.default_rng(42)
    X = rng.integers(1, V + 1, size=(B, T), dtype=np.int32)
    Y = rng.integers(0, C, size=(B, T), dtype=np.int32)

    before = [w.copy() for w in crf.get_weights()]
    model_nll.fit({"tokens": X, "labels": Y}, Y, epochs=1, batch_size=4, verbose=0)
    after = crf.get_weights()
    assert any(np.any(a != b) for a, b in zip(after, before))


def test_loss_class_trains_and_updates_transitions_dice():
    B, T, V, C = 8, 6, 30, 5
    _model_nll, model_dice, crf = _build_simple_model_heads(T=T, V=V, C=C, use_mask=False)
    loss_inst = CRFDiceLoss(smooth=1.0)
    model_dice.compile(optimizer=keras.optimizers.Adam(5e-3), loss=loss_inst, run_eagerly=True)

    rng = np.random.default_rng(7)
    X = rng.integers(1, V + 1, size=(B, T), dtype=np.int32)
    # Construct labels mostly matching a simple rule to help the optimizer
    Y = (X % C).astype(np.int32)

    before = [w.copy() for w in crf.get_weights()]
    model_dice.fit({"tokens": X, "labels": Y}, Y, epochs=1, batch_size=4, verbose=0)
    after = crf.get_weights()
    assert any(np.any(a != b) for a, b in zip(after, before))

