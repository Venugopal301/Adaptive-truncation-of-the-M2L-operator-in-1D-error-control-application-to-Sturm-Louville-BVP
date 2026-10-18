#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:59:31 2026

@author: venugopal

Test data from
Asymptotic and Numerical Methods in Estimating Eigenvalues


"""

import numpy as np
from scipy.sparse import diags
from scipy.linalg import eigh_tridiagonal
from scipy.linalg import block_diag

# ============================================================
# Parameters
# ============================================================

N = 2000
nu = 5
k = 1

h = 1.0 / (N + 1)
x = np.arange(1, N + 1) * h

# Singular point
t_k = k / nu

# ============================================================
# Potential
# q_k(x) = 1 / |x - t_k|^(1/2)
# ============================================================

q = 1.0 / np.sqrt(np.abs(x - t_k))

# ============================================================
# Tridiagonal matrix
# ============================================================

main_diag = 2.0 / h**2 + q
off_diag = -np.ones(N - 1) / h**2

T = diags(
    [off_diag, main_diag, off_diag],
    offsets=[-1, 0, 1],
    format="csr"
)

print("Matrix size:", T.shape)
print("h =", h)
print("Singular point t_k =", t_k)
print("Minimum distance from grid to singular point:",
      np.min(np.abs(x - t_k)))


alpha = T.diagonal()
beta = T.diagonal(k=1)

# # Main diagonal
# alpha = np.diag(T)

# # Superdiagonal
# beta = np.diag(T, k=1)

n = len(alpha)
m = n // 2

#------------------ deriving the secular equation parameters

alpha1 = alpha[:m].copy()
beta1  = beta[:m-1].copy()

bm = beta[m-1]

alpha1[-1] -= bm

alpha2 = alpha[m:].copy()
beta2  = beta[m:].copy()

alpha2[0] -= bm

Lambda1, Q1 = eigh_tridiagonal(alpha1, beta1)

Lambda2, Q2 = eigh_tridiagonal(alpha2, beta2)

v = np.zeros(len(alpha))

v[m-1] = 1.0
v[m]   = 1.0

Q = block_diag(Q1,Q2)

u = Q.T @ v

Lambda = np.concatenate((Lambda1,Lambda2))


""" 
#verify the T constructed
M = np.diag(Lambda) + bm*np.outer(u,u)
T_reconstructed = Q @ M @ Q.T
err = np.linalg.norm(G - T_reconstructed)

print(err)

err_abs = np.linalg.norm(T_dense - T_reconstructed, ord='fro')

err_rel = err_abs / np.linalg.norm(T_dense, ord='fro')

print("Absolute error :", err_abs)
print("Relative error :", err_rel)
"""

perm = np.argsort(Lambda)
Lambda_sorted = Lambda[perm]
u_sorted = u[perm]

x = Lambda_sorted.copy()
q = u_sorted
sigma = bm

#eig_T = eigh_tridiagonal(alpha, beta, eigvals_only=True)

# selecting the midpoints of the sources as the initial guess
y_mid = 0.5 * (x[:-1] + x[1:])

# One additional point to the left of the first source since \sigma < 0
h = x[1] - x[0]
y_first = x[0] - 0.5 * h

# Put the leftmost point first
y = np.concatenate(([y_first], y_mid))
y = y.reshape(-1, 1)

# y_mid = 0.5 * (x[:-1] + x[1:])
# h = x[-1] - x[-2]
# y_last = x[-1] + 0.5 * h
# y = np.concatenate((y_mid, [y_last]))
# y = y.reshape(-1, 1)

#----------------- save the variables
np.savez(
    "SL2_N_2000.npz",
    x=x,
    y = y,
    q=q,
    sigma=sigma
)

print("Data saved successfully.")

