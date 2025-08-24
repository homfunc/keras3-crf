import os
import time
import numpy as np
import keras
from keras import ops as K

from keras_crf.crf_ops import crf_decode, crf_log_likelihood


def bench_decode(B=16, T=20, N=8, iters=200, seed=0):
    rng = np.random.default_rng(seed)
    potentials = K.convert_to_tensor(rng.normal(size=(B, T, N)).astype("float32"))
    lens = K.convert_to_tensor(np.full((B,), T, dtype=np.int32))
    trans = K.convert_to_tensor(rng.normal(size=(N, N)).astype("float32"))

    # Warmup
    _ = crf_decode(potentials, lens, trans)

    t0 = time.perf_counter()
    for _ in range(iters):
        tags, score = crf_decode(potentials, lens, trans)
        # Force materialization
        _ = K.convert_to_numpy(score)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def bench_ll(B=16, T=20, N=8, iters=200, seed=1):
    rng = np.random.default_rng(seed)
    potentials = K.convert_to_tensor(rng.normal(size=(B, T, N)).astype("float32"))
    lens = K.convert_to_tensor(np.full((B,), T, dtype=np.int32))
    trans = K.convert_to_tensor(rng.normal(size=(N, N)).astype("float32"))
    tags = K.convert_to_tensor(rng.integers(0, N, size=(B, T), dtype=np.int32))

    # Warmup
    _ = crf_log_likelihood(potentials, tags, lens, trans)

    t0 = time.perf_counter()
    for _ in range(iters):
        ll = crf_log_likelihood(potentials, tags, lens, trans)
        _ = K.convert_to_numpy(ll)
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main():
    backend = os.environ.get("KERAS_BACKEND", "tensorflow")
    print(f"Backend: {backend}")

    # T=1 case
    t1_decode = bench_decode(T=1)
    t1_ll = bench_ll(T=1)

    # Multi-step case
    tm_decode = bench_decode(T=20)
    tm_ll = bench_ll(T=20)

    print(f"Decode avg time: T=1 -> {t1_decode*1e3:.3f} ms, T=20 -> {tm_decode*1e3:.3f} ms")
    print(f"Log-likelihood avg time: T=1 -> {t1_ll*1e3:.3f} ms, T=20 -> {tm_ll*1e3:.3f} ms")


if __name__ == "__main__":
    main()

