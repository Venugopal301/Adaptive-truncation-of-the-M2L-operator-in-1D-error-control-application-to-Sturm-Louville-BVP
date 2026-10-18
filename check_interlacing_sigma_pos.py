#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:00:18 2026

@author: venugopal
"""

import numpy as np

def check_interlacing_sigma_pos(x, y, tol=1e-12):
    """
    Check the interlacing property

        x1 < y1 < x2 < y2 < ... < xn < yn

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

    # Check first n-1 roots
    for i in range(n - 1):

        if not (x[i] + tol < y[i] < x[i + 1] - tol):
            failed_indices.append(i)

    # Check last root
    if not (y[-1] > x[-1] + tol):
        failed_indices.append(n - 1)

    failed_indices = np.array(failed_indices, dtype=int)
    num_failed = len(failed_indices)
    ok = (num_failed == 0)

    # Print summary
    if ok:
        print("✓ Interlacing property satisfied.")
    else:
        print(f"Interlacing failed for {num_failed} root(s).")
        # print("Failed indices:", failed_indices)

    return ok, num_failed, failed_indices
