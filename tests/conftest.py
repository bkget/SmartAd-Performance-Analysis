"""
conftest.py
===========
Pytest configuration and shared fixtures for SmartAd test suite.
"""

import os
import sys

# Ensure script/ directory is always on the path for all tests
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "script"))
