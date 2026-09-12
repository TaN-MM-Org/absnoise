"""Bring your own measurement data: documented file contracts (new in
v0.7).

Two plain-text CSV contracts, enforced instead of guessed at, with
exact save/load round trips asserted in the tests:

Critical-current curve (`load_ic_csv` / `save_ic_csv`), the input of
`fit_ic_curve`:

    T_K,Ic_A            (or T_K,Ic_A,sigma_Ic_A)

one row per temperature point, T strictly increasing, SI units.

Sampled readout trace (`load_trace_csv` / `save_trace_csv`), the input
of `fit_hmm`, `psd_single_sided` and `allan_variance`:

    t_s,y

one row per sample on a uniform time grid (checked; the decoder and
the spectral estimators all assume uniform sampling).

`validate_ic_data` separates two kinds of trouble the same way the
rest of this organization's packages do: structural problems
(non-finite values, non-positive temperatures, negative currents)
RAISE, while unit-plausibility findings are RETURNED as flags for the
user to judge -- a temperature above 30 K in a superconducting-
junction dataset usually means millikelvin values entered as kelvin,
and a critical current above 1 mA usually means microamps entered as
amps. Flags, not silent edits: this package never rescales a
measurement on its own.
"""
from __future__ import annotations

import csv

import numpy as np

__all__ = ["load_ic_csv", "save_ic_csv", "load_trace_csv",
           "save_trace_csv", "validate_ic_data"]

_IC2 = ("T_K", "Ic_A")
_IC3 = ("T_K", "Ic_A", "sigma_Ic_A")
_TR = ("t_s", "y")


def save_ic_csv(path, T_K, Ic_A, sigma_Ic_A=None):
    """Write an Ic(T) curve in the documented contract."""
    T = np.asarray(T_K, dtype=float).ravel()
    Ic = np.asarray(Ic_A, dtype=float).ravel()
    if T.size != Ic.size or T.size < 2:
        raise ValueError("T_K and Ic_A must be equal-length arrays "
                         "with at least 2 points")
    sig = None
    if sigma_Ic_A is not None:
        sig = np.asarray(sigma_Ic_A, dtype=float).ravel()
        if sig.size != T.size:
            raise ValueError("sigma_Ic_A must match T_K")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_IC3 if sig is not None else _IC2)
        for i in range(T.size):
            row = [repr(float(T[i])), repr(float(Ic[i]))]
            if sig is not None:
                row.append(repr(float(sig[i])))
            w.writerow(row)


def load_ic_csv(path):
    """Read an Ic(T) curve; returns (T_K, Ic_A, sigma_Ic_A or None).

    Refusals instead of guesses: wrong header, wrong field count, or
    a temperature axis that is not strictly increasing.
    """
    rows = []
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        try:
            header = tuple(h.strip() for h in next(r))
        except StopIteration:
            raise ValueError("empty Ic file") from None
        if header not in (_IC2, _IC3):
            raise ValueError(
                f"Ic file header must be exactly {_IC2} or {_IC3}; "
                f"got {header}")
        ncol = len(header)
        for line, rec in enumerate(r, start=2):
            if not rec:
                continue
            if len(rec) != ncol:
                raise ValueError(f"line {line}: expected {ncol} fields")
            rows.append([float(v) for v in rec])
    if len(rows) < 2:
        raise ValueError("Ic file contains fewer than 2 data rows")
    arr = np.asarray(rows, dtype=float)
    T = arr[:, 0]
    if np.any(np.diff(T) <= 0.0):
        raise ValueError("T_K must be strictly increasing")
    sig = arr[:, 2] if ncol == 3 else None
    return T, arr[:, 1], sig


def validate_ic_data(T_K, Ic_A, max_T_K=30.0, max_Ic_A=1e-3):
    """Structural checks raise; unit-plausibility findings are
    returned as flags for the user to judge.

    Returns dict(n_points, flags) where each flag is
    (index, kind, value): kind "T_large" for T above `max_T_K`
    (millikelvin entered as kelvin?), "Ic_large" for Ic above
    `max_Ic_A` (microamps entered as amps?). The thresholds are
    explicit keyword parameters, not hidden constants.
    """
    T = np.asarray(T_K, dtype=float).ravel()
    Ic = np.asarray(Ic_A, dtype=float).ravel()
    if T.shape != Ic.shape:
        raise ValueError("T_K and Ic_A must have the same shape")
    if not (np.all(np.isfinite(T)) and np.all(np.isfinite(Ic))):
        raise ValueError("Ic data contain non-finite values")
    if np.any(T <= 0.0):
        raise ValueError("temperatures must be positive kelvin")
    if np.any(Ic < 0.0):
        raise ValueError("critical currents must be non-negative")
    flags = []
    for i in np.nonzero(T > max_T_K)[0]:
        flags.append((int(i), "T_large", float(T[i])))
    for i in np.nonzero(Ic > max_Ic_A)[0]:
        flags.append((int(i), "Ic_large", float(Ic[i])))
    return dict(n_points=int(T.size), flags=flags)


def save_trace_csv(path, dt_s, y):
    """Write a uniformly sampled trace in the documented contract."""
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 2:
        raise ValueError("need at least 2 samples")
    dt = float(dt_s)
    if not (dt > 0.0 and np.isfinite(dt)):
        raise ValueError("dt_s must be finite and positive")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_TR)
        for i in range(y.size):
            w.writerow([repr(i * dt), repr(float(y[i]))])


def load_trace_csv(path, rtol=1e-6):
    """Read a sampled trace; returns (dt_s, y).

    The time grid must be uniform to relative tolerance `rtol` (the
    decoder and the spectral estimators assume uniform sampling); a
    non-uniform grid is refused, not resampled.
    """
    ts, ys = [], []
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        try:
            header = tuple(h.strip() for h in next(r))
        except StopIteration:
            raise ValueError("empty trace file") from None
        if header != _TR:
            raise ValueError(f"trace file header must be exactly "
                             f"{_TR}; got {header}")
        for line, rec in enumerate(r, start=2):
            if not rec:
                continue
            if len(rec) != 2:
                raise ValueError(f"line {line}: expected 2 fields")
            ts.append(float(rec[0]))
            ys.append(float(rec[1]))
    t = np.asarray(ts, dtype=float)
    y = np.asarray(ys, dtype=float)
    if t.size < 2:
        raise ValueError("trace file contains fewer than 2 samples")
    d = np.diff(t)
    dt = float(np.median(d))
    if dt <= 0.0 or np.any(np.abs(d - dt) > rtol * dt):
        raise ValueError(
            "time grid is not uniform (or not increasing); the "
            "decoder and spectral estimators assume uniform sampling, "
            "so resample deliberately rather than have it guessed")
    return dt, y
