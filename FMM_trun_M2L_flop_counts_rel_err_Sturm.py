# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:51:28 2026

@author: se22pmat003
"""

import numpy as np
# from scipy.io import loadmat
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, "utils")

sys.path.insert(0, utils_dir)
sys.path.insert(0, current_dir)


from FMM_trun_M2L_expansion_flops import FMM_trun_M2L_expansion_flops
from particle_to_local import particle_to_local
from flops import flops

scaling = 0

# data = loadmat("test_data_N_512.mat")
# data2 = data["data_start"]

# x = np.squeeze(data["x"])
# y = np.squeeze(data["y"])
# q = np.squeeze(data["q"])


# Load PDE data
# data = np.load("test_data_trun_M2L/Gauss_Laguerre_2000.npz")
data = np.load("test_data_trun_M2L/SL2_N_2000.npz")

x = np.squeeze(data["x"])
y = np.squeeze(data["y"])
q = np.squeeze(data["q"])
sigma = np.squeeze(data["sigma"])

q = q**2

results_adaptive = []
flops_adaptive_all = []
flops_fixed_all = []
r_max = []



# for k in range(5,6):
for k in range(5,13):
    eps = 10.0**(-k)
    print("epsilon = ", eps)

    u1, u2, post_ord, ch, Px, Py, I, nflops, neighbor, r , flops_adaptive \
            = FMM_trun_M2L_expansion_flops(eps, x, y, q, scaling
            )

    z1 = np.zeros(len(q))
    z2 = np.zeros(len(q))
    
    # ==========================================
    # Near-field interaction & final evaluation
    # ==========================================
    
    for i in post_ord:
        idx = i - 1
    
        # leaf node ?
        if len(ch[idx]) == 0:
            py = Py[idx]
            if len(py) > 0:
                yj = y[py]
                if u1[idx] is not None:
                    Ui = particle_to_local(yj, r, I[idx, 0], 2 * I[idx, 1], scaling)
                    z1[py] = Ui @ u1[idx]
                    z2[py] = Ui @ u2[idx]
                    nflops += flops("prod", Ui, "n", u1[idx], "n")
                    flops_adaptive["cauchy"]["L2P"] += flops("prod", Ui, "n", u1[idx], "n")
                    flops_adaptive["inverse_square"]["L2P"] += flops("prod", Ui, "n", u1[idx], "n")
                    
                    near_nodes = [i] + neighbor[idx]
    
                for j in np.unique(near_nodes):
                    jdx = j - 1
                    px = Px[jdx]
                    if len(px) > 0:
                        xi = x[px]
                        D1 = 1.0 / (xi[np.newaxis, :] - yj[:, np.newaxis])
                        D2 = 1.0 / (xi[np.newaxis, :] - yj[:, np.newaxis]) ** 2
                        D1[np.isinf(D1)] = 0.0
                        D2[np.isinf(D2)] = 0.0
                        z1[py] += D1 @ q[px]
                        z2[py] += D2 @ q[px]
                        nflops += D1.shape[0] * (2 * D1.shape[1] - 1) + len(py)
    
                        flops_adaptive["cauchy"]["near_field"] += D1.shape[0] * (2 * D1.shape[1] - 1) + len(py)                         
                        flops_adaptive["inverse_square"]["near_field"] += D2.shape[0] * (2 * D2.shape[1] - 1) + len(py)                         
                        
    flops_adaptive["cauchy"]["total"] = sum( flops_adaptive["cauchy"][key]
        for key in ["P2M", "M2M", "M2L", "L2L", "L2P", "near_field"]
            )
    flops_adaptive["inverse_square"]["total"] = sum( flops_adaptive["inverse_square"][key]
        for key in ["P2M", "M2M", "M2L", "L2L", "L2P", "near_field"]
            )
   
    
    flops_adaptive_all.append({
    "epsilon": eps,
    "r": r,
    **flops_adaptive
        })
    
    
    z1 = 1 + sigma*z1
    z2 = sigma*z2
    
    ###### Direct evaluation ##############
    
    Ker1 = np.zeros(len(y), dtype=np.float64)
    Ker2 = np.zeros(len(y), dtype=np.float64)
    
    for i in range(len(y)):
        diff = x - y[i]
    
        Ker1[i] = np.sum(q / diff)
        Ker2[i] = np.sum(q / diff**2)  
    
    exact_sum1 = 1 + sigma*Ker1
    exact_sum2 = sigma*Ker2
    
    err1 = np.linalg.norm(exact_sum1 - z1) / np.linalg.norm(exact_sum1)
    err2 = np.linalg.norm(exact_sum2 - z2) / np.linalg.norm(exact_sum2)
    
    print("expansion order = ", r)
    # print("Relative error (function)   =", err1, "flop count", nflops)
    # print("Relative error (derivative) =", err2)
   
    r_max.append(r)
    
    results_adaptive.append([
        eps,
        r,
        err1,
        err2,
        ])

        
# INITIALIZE DATA FOR FIXED ORDER 
    
r = 0
err1 = 0
err2 = 0
nflops = 0

results_fixed = []
y = np.squeeze(data["y"])


#=========================================================================
#                  FIXED EXPANSION ORDER
#=========================================================================

from FMM_fixed_expansion import FMM_fixed_expansion

for r in r_max:     

    u1, u2, post_ord, ch, Px, Py, I, nflops, neighbor, flops_fixed \
        = FMM_fixed_expansion(r, x, y, q, scaling)

    z1 = np.zeros(len(q))
    z2 = np.zeros(len(q))
    
    # ==========================================
    # Near-field interaction & final evaluation
    # ==========================================
    
    for i in post_ord:
        idx = i - 1
    
        # leaf node ?
        if len(ch[idx]) == 0:
            py = Py[idx]
            if len(py) > 0:
                yj = y[py]
                if u1[idx] is not None:
                    Ui = particle_to_local(yj, r, I[idx, 0], 2 * I[idx, 1], scaling)
                    z1[py] = Ui @ u1[idx]
                    z2[py] = Ui @ u2[idx]
                    nflops += flops("prod", Ui, "n", u1[idx], "n")
                    flops_fixed["cauchy"]["L2P"] += flops("prod", Ui, "n", u1[idx], "n")
                    flops_fixed["inverse_square"]["L2P"] += flops("prod", Ui, "n", u1[idx], "n")
                   
                    near_nodes = [i] + neighbor[idx]
    
                for j in np.unique(near_nodes):
                    jdx = j - 1
                    px = Px[jdx]
                    if len(px) > 0:
                        xi = x[px]
                        D1 = 1.0 / (xi[np.newaxis, :] - yj[:, np.newaxis])
                        D2 = 1.0 / (xi[np.newaxis, :] - yj[:, np.newaxis]) ** 2
                        D1[np.isinf(D1)] = 0.0
                        D2[np.isinf(D2)] = 0.0
                        z1[py] += D1 @ q[px]
                        z2[py] += D2 @ q[px]
                        nflops += D1.shape[0] * (2 * D1.shape[1] - 1) + len(py)
    
                        flops_fixed["cauchy"]["near_field"] += D1.shape[0] * (2 * D1.shape[1] - 1) + len(py)
                        flops_fixed["inverse_square"]["near_field"] += D2.shape[0] * (2 * D2.shape[1] - 1) + len(py)
                        
    flops_fixed["cauchy"]["total"] = sum( flops_fixed["cauchy"][key]
        for key in ["P2M", "M2M", "M2L", "L2L", "L2P", "near_field"]
            )
    flops_fixed["inverse_square"]["total"] = sum( flops_fixed["inverse_square"][key]
        for key in ["P2M", "M2M", "M2L", "L2L", "L2P", "near_field"]
            )
   
    
    
    flops_fixed_all.append({
    "epsilon": eps,
    "r": r,
    **flops_fixed
        })
    
    z1 = 1 + sigma*z1
    z2 = sigma*z2
    
    err1 = np.linalg.norm(exact_sum1 - z1) / np.linalg.norm(exact_sum1)
    err2 = np.linalg.norm(exact_sum2 - z2) / np.linalg.norm(exact_sum2)
    
    results_fixed.append([
        eps,
        r,
        err1,
        err2
    ])
    
    
#-----------------------------------------------------------------
#       ANALYSE RESULTS
#-----------------------------------------------------------------

import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Convert results to NumPy arrays
# ---------------------------------------------------------
adaptive = np.array(results_adaptive, dtype=float)
fixed = np.array(results_fixed, dtype=float)

# Adaptive:
# [epsilon, r, function error, derivative error, FLOPs]
eps_adaptive = adaptive[:,0]
r_adaptive = adaptive[:, 1]
err_fun_adaptive = adaptive[:, 2]
err_der_adaptive = adaptive[:, 3]

# Non-adaptive:
# [r, function error, derivative error, FLOPs]
eps_fixed = fixed[:,0]
r_fixed = fixed[:, 1]
err_fun_fixed = fixed[:, 2]
err_der_fixed = fixed[:, 3]


# =========================================================
# Figure 1: TOTAL FLOPS VERSUS EXPANSION ORDER 
# =========================================================

#==================== CAUCHY KERNEL ======================

plt.figure(figsize=(7, 5))

cauchy_adaptive_total = [
    item["cauchy"]["total"] for item in flops_adaptive_all
]

cauchy_fixed_total = [
    item["cauchy"]["total"] for item in flops_fixed_all
]

plt.plot(
    r_adaptive,
    cauchy_adaptive_total,
    'o-',
    label='Adaptive order'
)

plt.plot(
    r_fixed,
    cauchy_fixed_total,
    's-',
    label='Fixed-order'
)

plt.xlabel('Expansion order $r$')
plt.ylabel('Total FLOP counts')
plt.title('Total FLOPs versus Expansion Order for Cauchy kernel')

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

#============ INVERSE SQUARE KERNEL ========================

plt.figure(figsize=(7, 5))

inverse_square_adaptive_total = [
    item["inverse_square"]["total"] for item in flops_adaptive_all
]

inverse_square_fixed_total = [
    item["inverse_square"]["total"] for item in flops_fixed_all
]


plt.plot(
    r_adaptive,
    inverse_square_adaptive_total,
    'o-',
    label='Adaptive order'
)

plt.plot(
    r_fixed,
    inverse_square_fixed_total,
    's-',
    label='Fixed-order'
)

plt.xlabel('Expansion order $r$')
plt.ylabel('Total FLOP counts')
plt.title('Total FLOPs versus Expansion Order for inverse square kernel')

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()


# =========================================================
# Figure 2: RELATIVE ERROR CAUCHY KERNEL
# =========================================================

plt.figure(figsize=(7, 5))

plt.semilogy(
    r_adaptive,
    err_fun_adaptive,
    'o-',
    label='Adaptive order'
)

plt.semilogy(
    r_fixed,
    err_fun_fixed,
    's-',
    label='Fixed-order'
)

plt.xlabel('Expansion order $r$')
plt.ylabel('Relative error (Cauchy kernel)')
plt.title('Relative Error versus Expansion Order - Cauchy kernel')

plt.grid(True, which='both')
plt.legend()
plt.tight_layout()

plt.show()


# =========================================================
# Figure 3: RELATIVE ERROR INVERSE SQUARE KERNEL
# =========================================================

plt.figure(figsize=(7, 5))

plt.semilogy(
    r_adaptive,
    err_der_adaptive,
    'o-',
    label='Adaptive order'
)

plt.semilogy(
    r_fixed,
    err_der_fixed,
    's-',
    label='Fixed-order'
)

plt.xlabel('Expansion order $r$')
plt.ylabel('Relative error (inverse square kernel)')
plt.title('Relative Error versus Expansion Order- inverse square kernel')

plt.grid(True, which='both')
plt.legend()
plt.tight_layout()

plt.show()



# =======================================================================
# Table: Percentage reduction in M2L and Total FLOP counts - Cauchy kernel
# =======================================================================

import pandas as pd

# ---------------------------------------------------------
# Extract adaptive data
# ---------------------------------------------------------
eps_adaptive = np.array(
    [d["epsilon"] for d in flops_adaptive_all],
    dtype=float
)

# Extract expansion order
r_used = [
    d["r"] for d in flops_adaptive_all
]

# M2L FLOPs for each epsilon
m2l_adaptive = np.array(
    [d["cauchy"]["M2L"] for d in flops_adaptive_all],
    dtype=float
)

# Total FLOPs for each epsilon
total_adaptive = np.array(
    [d["cauchy"]["total"] for d in flops_adaptive_all],
    dtype=float
)


# Extract fixed-order FLOP counts for each expansion order
m2l_fixed = np.array(
    [d["cauchy"]["M2L"] for d in flops_fixed_all],
    dtype=float
)

total_fixed = np.array(
    [d["cauchy"]["total"] for d in flops_fixed_all],
    dtype=float
)

# ---------------------------------------------------------
# Percentage reductions
# ---------------------------------------------------------
m2l_reduction = (
    (m2l_fixed - m2l_adaptive) / m2l_fixed
) * 100

total_reduction = (
    (total_fixed - total_adaptive) / total_fixed
) * 100

# ---------------------------------------------------------
# Create results table
# ---------------------------------------------------------
results = pd.DataFrame({
    "epsilon": eps_adaptive,
    "r":       r_used, 
    "Fixed M2L FLOPs": m2l_fixed,
    "Adaptive M2L FLOPs": m2l_adaptive,
    "M2L Reduction (%)": m2l_reduction,
    "Fixed Total FLOPs": total_fixed,
    "Adaptive Total FLOPs": total_adaptive,
    "Total Reduction (%)": total_reduction
})

# ---------------------------------------------------------
# Print table
# ---------------------------------------------------------
print("\nM2L and Total FLOP Comparison - Cauchy Kernel")
print("=" * 120)

print(results.to_string(
    index=False,
    formatters={
        "epsilon": lambda x: f"{x:.1e}",
        "r": lambda x: f"{x:.0f}",
        "Fixed M2L FLOPs": lambda x: f"{x:.0f}",
        "Adaptive M2L FLOPs": lambda x: f"{x:.0f}",
        "M2L Reduction (%)": lambda x: f"{x:.2f}",
        "Fixed Total FLOPs": lambda x: f"{x:.0f}",
        "Adaptive Total FLOPs": lambda x: f"{x:.0f}",
        "Total Reduction (%)": lambda x: f"{x:.2f}"
    }
))

print("=" * 120)


# =======================================================================
# Table: Percentage reduction in M2L and Total FLOP counts - Inverse-square kernel
# =======================================================================

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Extract adaptive data
# ---------------------------------------------------------
eps_adaptive = np.array(
    [d["epsilon"] for d in flops_adaptive_all],
    dtype=float
)

# Extract expansion order
r_used = [
    d["r"] for d in flops_adaptive_all
]
# M2L FLOPs
m2l_adaptive = np.array(
    [d["inverse_square"]["M2L"] for d in flops_adaptive_all],
    dtype=float
)

# Total FLOPs
total_adaptive = np.array(
    [d["inverse_square"]["total"] for d in flops_adaptive_all],
    dtype=float
)


# Extract fixed-order FLOP counts for each expansion order
m2l_fixed = np.array(
    [d["inverse_square"]["M2L"] for d in flops_fixed_all],
    dtype=float
)

total_fixed = np.array(
    [d["inverse_square"]["total"] for d in flops_fixed_all],
    dtype=float
)

# ---------------------------------------------------------
# Percentage reduction in M2L FLOPs
# ---------------------------------------------------------
m2l_reduction = (
    (m2l_fixed - m2l_adaptive) / m2l_fixed
) * 100

# ---------------------------------------------------------
# Percentage reduction in Total FLOPs
# ---------------------------------------------------------
total_reduction = (
    (total_fixed - total_adaptive) / total_fixed
) * 100

# ---------------------------------------------------------
# Create results table
# ---------------------------------------------------------
results_inverse_square = pd.DataFrame({
    "epsilon": eps_adaptive,
    "r": r_used,
    "Fixed M2L FLOPs": m2l_fixed,
    "Adaptive M2L FLOPs": m2l_adaptive,
    "M2L Reduction (%)": m2l_reduction,
    "Fixed Total FLOPs": total_fixed,
    "Adaptive Total FLOPs": total_adaptive,
    "Total Reduction (%)": total_reduction
})

# ---------------------------------------------------------
# Print table
# ---------------------------------------------------------
print("\nM2L and Total FLOP Comparison - Inverse-Square Kernel")
print("=" * 120)

print(results_inverse_square.to_string(
    index=False,
    formatters={
        "epsilon": lambda x: f"{x:.1e}",
        "r": lambda x: f"{x:.0f}",
        "Fixed M2L FLOPs": lambda x: f"{x:.0f}",
        "Adaptive M2L FLOPs": lambda x: f"{x:.0f}",
        "M2L Reduction (%)": lambda x: f"{x:.2f}",
        "Fixed Total FLOPs": lambda x: f"{x:.0f}",
        "Adaptive Total FLOPs": lambda x: f"{x:.0f}",
        "Total Reduction (%)": lambda x: f"{x:.2f}"
    }
))

print("=" * 120)


# =========================================================
# Figure 4: Pie chart adaptive expansion order cauchy kernel
# =========================================================

# Extract total FLOPs
total_flops = flops_adaptive["cauchy"]["total"]

# Compute percentage contribution of each operation
flop_percentages = {
    "P2M": 100 * flops_adaptive["cauchy"]["P2M"] / total_flops,
    "M2M": 100 * flops_adaptive["cauchy"]["M2M"] / total_flops,
    "M2L": 100 * flops_adaptive["cauchy"]["M2L"] / total_flops,
    "L2L": 100 * flops_adaptive["cauchy"]["L2L"] / total_flops,
    "L2P": 100 * flops_adaptive["cauchy"]["L2P"] / total_flops,
    "near_field": 100* flops_adaptive["cauchy"]["near_field"]/total_flops
}

print("FLOP percentages adaptive expansion order (cauchy kernel) :")
for key, value in flop_percentages.items():
    print(f"{key}: {value:.2f}%")

labels = ["P2M", "M2M", "M2L", "L2L", "L2P", "near_field"]
values = [
    flop_percentages["P2M"],
    flop_percentages["M2M"],
    flop_percentages["M2L"],
    flop_percentages["L2L"],
    flop_percentages["L2P"],
    flop_percentages["near_field"]
]

plt.figure(figsize=(7, 7))

plt.pie(
    values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(f"FLOP distribution for adaptive expansion order Cauchy kernel (r_max = {r})")
plt.tight_layout()
plt.show()

# =========================================================
# Figure 5: Pie chart fixed expansion order cauchy kernel
# =========================================================

# Extract total FLOPs
total_flops_fixed = flops_fixed["cauchy"]["total"]

# Compute percentage contribution of each operation
flop_percentages_fixed = {
    "P2M": 100 * flops_fixed["cauchy"]["P2M"] / total_flops_fixed,
    "M2M": 100 * flops_fixed["cauchy"]["M2M"] / total_flops_fixed,
    "M2L": 100 * flops_fixed["cauchy"]["M2L"] / total_flops_fixed,
    "L2L": 100 * flops_fixed["cauchy"]["L2L"] / total_flops_fixed,
    "L2P": 100 * flops_fixed["cauchy"]["L2P"] / total_flops_fixed,
    "near_field": 100* flops_adaptive["cauchy"]["near_field"]/total_flops
}

print("FLOP percentages fixed expansion order (cauchy kernel):")
for key, value in flop_percentages_fixed.items():
    print(f"{key}: {value:.2f}%")

labels = ["P2M", "M2M", "M2L", "L2L", "L2P", "near_field"]
values = [
    flop_percentages_fixed["P2M"],
    flop_percentages_fixed["M2M"],
    flop_percentages_fixed["M2L"],
    flop_percentages_fixed["L2L"],
    flop_percentages_fixed["L2P"],
    flop_percentages_fixed["near_field"]
]

plt.figure(figsize=(7, 7))

plt.pie(
    values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(f"FLOP distribution for fixed expansion order Cauchy kernel (r = {r})")
plt.tight_layout()
plt.show()



# =========================================================
# Figure 6: Pie chart adaptive expansion order inverse square kernel
# =========================================================

# Extract total FLOPs
total_flops = flops_adaptive["inverse_square"]["total"]

# Compute percentage contribution of each operation
flop_percentages = {
    "P2M": 100 * flops_adaptive["inverse_square"]["P2M"] / total_flops,
    "M2M": 100 * flops_adaptive["inverse_square"]["M2M"] / total_flops,
    "M2L": 100 * flops_adaptive["inverse_square"]["M2L"] / total_flops,
    "L2L": 100 * flops_adaptive["inverse_square"]["L2L"] / total_flops,
    "L2P": 100 * flops_adaptive["inverse_square"]["L2P"] / total_flops,
    "near_field": 100* flops_adaptive["inverse_square"]["near_field"]/total_flops
}

print("FLOP percentages adaptive expansion order (inverse_square kernel):")
for key, value in flop_percentages.items():
    print(f"{key}: {value:.2f}%")

labels = ["P2M", "M2M", "M2L", "L2L", "L2P", "near_field"]
values = [
    flop_percentages["P2M"],
    flop_percentages["M2M"],
    flop_percentages["M2L"],
    flop_percentages["L2L"],
    flop_percentages["L2P"],
    flop_percentages["near_field"]
]

plt.figure(figsize=(7, 7))

plt.pie(
    values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(f"FLOP distribution for adaptive expansion order inverse_square kernel (r_max = {r})")
plt.tight_layout()
plt.show()

# =========================================================
# Figure 7: Pie chart fixed expansion order inverse square kernel
# =========================================================

# Extract total FLOPs
total_flops_fixed = flops_fixed["inverse_square"]["total"]

# Compute percentage contribution of each operation
flop_percentages_fixed = {
    "P2M": 100 * flops_fixed["inverse_square"]["P2M"] / total_flops_fixed,
    "M2M": 100 * flops_fixed["inverse_square"]["M2M"] / total_flops_fixed,
    "M2L": 100 * flops_fixed["inverse_square"]["M2L"] / total_flops_fixed,
    "L2L": 100 * flops_fixed["inverse_square"]["L2L"] / total_flops_fixed,
    "L2P": 100 * flops_fixed["inverse_square"]["L2P"] / total_flops_fixed,
    "near_field": 100* flops_adaptive["inverse_square"]["near_field"]/total_flops
}

print("FLOP percentages fixed expansion order (inverse_square kernel):")
for key, value in flop_percentages_fixed.items():
    print(f"{key}: {value:.2f}%")

labels = ["P2M", "M2M", "M2L", "L2L", "L2P", "near_field"]
values = [
    flop_percentages_fixed["P2M"],
    flop_percentages_fixed["M2M"],
    flop_percentages_fixed["M2L"],
    flop_percentages_fixed["L2L"],
    flop_percentages_fixed["L2P"],
    flop_percentages_fixed["near_field"]
]

plt.figure(figsize=(7, 7))

plt.pie(
    values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(f"FLOP distribution for fixed expansion order inverse_square kernel (r = {r})")
plt.tight_layout()
plt.show()

