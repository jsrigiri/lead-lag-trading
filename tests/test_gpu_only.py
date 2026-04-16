import os
import pytest

RUN_GPU_TESTS = os.getenv("RUN_GPU_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_GPU_TESTS,
    reason="GPU-only tests disabled; set RUN_GPU_TESTS=1 to enable"
)