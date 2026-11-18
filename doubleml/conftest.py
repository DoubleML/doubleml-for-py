"""Pytest configuration shared by all tests and doctests in the doubleml package."""

import matplotlib

# Use the non-interactive backend so plotting tests do not depend on a GUI toolkit (e.g. Tk) being available.
matplotlib.use("Agg")
