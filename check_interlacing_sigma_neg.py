#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:51:20 2026

@author: venugopal
"""

import numpy as np

def check_interlacing_sigma_neg(x, y, tol=1e-12):
    """
    Check the interlacing property for sigma < 0

        y1 < x1 < y2 < x2 < ... < yn < xn

    Parameters
    ----------
    x : ndarray
        Sorted poles.
    y : ndarray
        Sorted eigenvalues (roots of secular equation).
    tol : float
        Numerical tolerance.

    Returns
    -------
    ok : bool
        True if all roots satisfy interlacing.
    num_failed : int
        Number of roots violating interlacing.
    failed_indices : ndarray
        Indices of the failed roots.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    n = len(x)

    if len(y) != n:
        raise ValueError("x and y must have the same length.")

    failed_indices = []

    # First root: y1 < x1
    if not (y[0] < x[0] - tol):
        failed_indices.append(0)

    # Remaining roots: x[i-1] < y[i] < x[i]
    for i in range(1, n):

        if not (x[i-1] + tol < y[i] < x[i] - tol):
            failed_indices.append(i)

    failed_indices = np.array(failed_indices, dtype=int)

    num_failed = len(failed_indices)

    ok = (num_failed == 0)

    # Print summary
    if ok:
        print("Interlacing property satisfied.")
    else:
        print(f"Interlacing failed for {num_failed} root(s).")
        print("Failed indices:", failed_indices)

    return ok, num_failed, failed_indices