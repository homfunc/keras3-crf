"""
Nox test session for keras-crf with automatic Keras backend selection.

Usage:
  - Default (auto: jax > tensorflow > torch):
      nox -s tests
  - Force a backend via CLI:
      nox -s tests -- backend=jax | backend=tensorflow | backend=torch
  - Or via environment variable before running nox:
      KERAS_BACKEND=jax nox -s tests

This session installs:
  - Project in editable mode (-e .)
  - Test dependencies (.[test])
  - A backend extra (.[jax] / .[tensorflow] / .[torch]) chosen automatically
"""
from __future__ import annotations

import os
import sys
import nox

BACKEND_EXTRAS = {
    "jax": "jax",
    "tensorflow": "tensorflow",
    "torch": "torch",
}

BACKEND_PREFERENCE = ["jax", "tensorflow", "torch"]


def _parse_backend_posarg(posargs: list[str]) -> tuple[str | None, list[str]]:
    requested = None
    passthrough: list[str] = []
    for arg in posargs:
        if arg.startswith("backend="):
            requested = arg.split("=", 1)[1].strip()
        else:
            passthrough.append(arg)
    return requested, passthrough


def _install_backend(session: nox.Session, backend: str) -> bool:
    extra = BACKEND_EXTRAS.get(backend)
    if not extra:
        session.log(f"Unknown backend '{backend}'. Valid: {list(BACKEND_EXTRAS)}")
        return False
    try:
        session.log(f"Installing backend extra: .[{extra}]")
        session.install(f".[{extra}]")
        # Ensure jaxlib is present for JAX CPU wheels on CI
        if backend == "jax":
            session.install("jaxlib>=0.4.28")
        session.env["KERAS_BACKEND"] = backend
        session.log(f"KERAS_BACKEND set to '{backend}'")
        return True
    except Exception as e:  # nox.command.CommandFailed in practice
        session.log(f"Failed to install backend '{backend}': {e}")
        return False


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def tests(session: nox.Session) -> None:
    requested, passthrough = _parse_backend_posarg(session.posargs)

    # Always install package and test deps first.
    session.install("-e", ".")
    session.install(".[test]")

    # Determine backend choice: CLI > env var > auto
    backend = requested or os.environ.get("KERAS_BACKEND")

    if backend:
        if not _install_backend(session, backend):
            session.error(
                f"Requested/backend from env '{backend}' could not be installed. "
                f"Try a different backend (backend=jax|tensorflow|torch)."
            )
    else:
        chosen = None
        for b in BACKEND_PREFERENCE:
            if _install_backend(session, b):
                chosen = b
                break
        if chosen is None:
            session.warn(
                "No backends could be installed; tests will likely fail at import time."
            )

    # Run the tests (pass through any extra pytest args from CLI)
    session.run("pytest", "-q", *passthrough)


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def quickstarts(session: nox.Session) -> None:
    """Run the backend-specific quickstart scripts in examples/ as a smoke test."""
    requested, _ = _parse_backend_posarg(session.posargs)

    # Install the package (no test deps needed) and backend
    session.install("-e", ".")

    backend = requested or os.environ.get("KERAS_BACKEND")
    if backend:
        if not _install_backend(session, backend):
            session.error(
                f"Requested/backend from env '{backend}' could not be installed. "
                f"Try a different backend (backend=jax|tensorflow|torch)."
            )
    else:
        chosen = None
        for b in BACKEND_PREFERENCE:
            if _install_backend(session, b):
                chosen = b
                break
        if chosen is None:
            session.error("No backends could be installed; aborting quickstarts session.")
        backend = chosen

    # Run the appropriate quickstart
    if backend == "torch":
        session.run("python", "examples/quickstart_torch.py")
    elif backend == "jax":
        session.run("python", "examples/quickstart_jax.py")
    elif backend == "tensorflow":
        session.run("python", "examples/quickstart_tf.py")
    else:
        session.error(f"Unsupported backend '{backend}' for quickstarts.")


@nox.session(name="clean")
def clean(session: nox.Session) -> None:
    """Remove local build/test artifacts and nox virtualenvs.

    Usage:
        nox -s clean
    """
    import shutil
    import glob
    import os

    # Paths/directories to remove
    paths = [
        ".nox",
        ".pytest_cache",
        "build",
        "dist",
    ]

    for p in paths:
        session.log(f"Removing {p} ...")
        shutil.rmtree(p, ignore_errors=True)

    # Remove any egg-info directories
    for egg in glob.glob("*.egg-info"):
        session.log(f"Removing {egg} ...")
        shutil.rmtree(egg, ignore_errors=True)

