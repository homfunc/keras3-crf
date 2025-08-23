import os
import pytest

# Define a custom marker for tests that require TensorFlow-based ops
# Usage: pytestmark = pytest.mark.tf_only (at module level) or @pytest.mark.tf_only on tests

def pytest_configure(config):
    config.addinivalue_line("markers", "tf_only: marks tests that require TensorFlow backend/ops")


def pytest_collection_modifyitems(config, items):
    backend = os.environ.get("KERAS_BACKEND", "tensorflow").lower()
    skip_tf_only = pytest.mark.skip(reason="Skipping TF-only tests on non-TensorFlow backend")
    if backend != "tensorflow":
        for item in items:
            if any(mark.name == "tf_only" for mark in item.iter_markers()):
                item.add_marker(skip_tf_only)

