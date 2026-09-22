"""Compatibility helpers for the range of NumPy versions pyproject.toml
allows (NumPy >= 1.24).

`numpy.trapezoid` exists only from NumPy 2.0; NumPy 1.x has the same
function under the name `numpy.trapz` (removed in NumPy 2.4). The name
is looked up at call time, so the helper works on either side.
"""
import numpy as np


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Trapezoidal-rule integral; `numpy.trapezoid` (NumPy >= 2.0) or
    `numpy.trapz` (NumPy 1.x), same arguments and result."""
    fn = getattr(np, "trapezoid", None)
    if fn is None:
        fn = np.trapz
    return fn(y, x=x, dx=dx, axis=axis)
