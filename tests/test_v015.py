"""0.15.0: the exact critical current is the default calibration.

Up to 0.14.0 `JunctionModel.calibrate` and `junction_properties` took
the critical current from `Ic`, the largest value of the level-based
current on its phase grid. For MoRe that value lies 1.7 % above the
exact maximum of the same model (`Ic_matsubara`), so a model calibrated
by default had an exact critical current at 20 mK 1.7 % below the
measured Ic20 it was calibrated to. Since 0.15.0 both use the exact
maximum by default; method="levels" reproduces the old numbers.
"""
import dataclasses

import numpy as np
import pytest

from absnoise import RECIPES, JunctionModel, junction_properties


def _recipe(name_start):
    return next(r for r in RECIPES if r.name.startswith(name_start))


def test_default_calibration_reproduces_the_measured_critical_current():
    """MoRe: the default calibration makes the exact critical current at
    20 mK equal Ic20. The 0.14.0 default (method="levels") left it
    1.66 % low (scale 0.492841 instead of 0.501142)."""
    r = _recipe("MoRe")
    m = JunctionModel(r)
    s_old = m.calibrate(method="levels")
    exact_old = m.Ic_matsubara(0.02)
    s_new = m.calibrate()
    exact_new = m.Ic_matsubara(0.02)
    assert exact_new == pytest.approx(r.Ic20, rel=1e-9)
    assert exact_old / r.Ic20 == pytest.approx(0.98344, abs=2e-4)
    assert s_new / s_old == pytest.approx(1.01684, abs=2e-4)


def test_default_junction_table_is_the_exact_one():
    r = dataclasses.replace(RECIPES[1], W=0.3e-6)  # Ti/Al/Au, narrow
    T = np.array([0.225])
    default = junction_properties(r, T)
    exact = junction_properties(r, T, method="matsubara")
    for k in ("Ic", "dIdphi0", "a3", "scale"):
        assert np.array_equal(np.asarray(default[k]), np.asarray(exact[k]))
    m = JunctionModel(r)                           # scale 1
    ic_exact = m.Ic_matsubara(0.02)
    assert m.calibrate() == pytest.approx(r.Ic20 / ic_exact, rel=1e-12)
