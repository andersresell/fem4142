"""Shared setup for the test suite. pytest loads this file automatically.

Run all tests from the project root with:

    python -m pytest
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")  # Draw into memory only, so no windows pop up while testing

# Make the project root importable, so that "from fem... import ..." works from the tests folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
