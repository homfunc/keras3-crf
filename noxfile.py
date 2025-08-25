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


@nox.session(python=["3.9", "3.10", "3.11", "3.12"])
def tests(session: nox.Session) -> None:
    requested, passthrough = _parse_backend_posarg(session.posargs)

    # Always install package and test deps first.
    session.install("-e", ".")
    session.install(".[test]")

    # Determine backend choice: CLI > env var > auto
    backend = requested or os.environ.get("KERAS_BACKEND")

    def install_backend(b: str) -> bool:
        extra = BACKEND_EXTRAS.get(b)
        if not extra:
            session.log(f"Unknown backend '{b}'. Valid: {list(BACKEND_EXTRAS)}")
            return False
        try:
            session.log(f"Installing backend extra: .[{extra}]")
            session.install(f".[{extra}]")
            session.env["KERAS_BACKEND"] = b
            session.log(f"KERAS_BACKEND set to '{b}'")
            return True
        except Exception as e:  # nox.command.CommandFailed in practice
            session.log(f"Failed to install backend '{b}': {e}")
            return False

    if backend:
        if not install_backend(backend):
            session.error(
                f"Requested/backend from env '{backend}' could not be installed. "
                f"Try a different backend (backend=jax|tensorflow|torch)."
            )
    else:
        chosen = None
        for b in BACKEND_PREFERENCE:
            if install_backend(b):
                chosen = b
                break
        if chosen is None:
            session.warn(
                "No backends could be installed; tests will likely fail at import time."
            )

    # Run the tests (pass through any extra pytest args from CLI)
    session.run("pytest", "-q", *passthrough)

