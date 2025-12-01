import os
import subprocess
import sys
import pytest


def run_script(path):
    """Run a python script with EXAMPLES_MAX_ITERS=1 and assert it exits cleanly."""
    env = os.environ.copy()
    env["EXAMPLES_MAX_ITERS"] = "1"
    # Enable fast mode to avoid slow training in CI/tests
    env["EXAMPLES_FAST_MODE"] = "1"
    proc = subprocess.run([sys.executable, path], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    # For debugging on failure include stdout/stderr
    assert proc.returncode == 0, f"Script {path} failed:\nSTDOUT:\n{proc.stdout.decode()}\nSTDERR:\n{proc.stderr.decode()}"


def test_classification_examples_runs():
    run_script(os.path.join("examples", "classification_examples.py"))


def test_regression_examples_runs():
    run_script(os.path.join("examples", "regression_examples.py"))
