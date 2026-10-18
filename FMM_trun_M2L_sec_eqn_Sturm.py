# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:51:28 2026

@author: se22pmat003
"""

import numpy as np
# from scipy.io import loadmat
import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, "utils")

sys.path.insert(0, utils_dir)
sys.path.insert(0, current_dir)

from FMM_trun_M2L_expansion_sec_eqn import FMM_trun_M2L_expansion_sec_eqn
from particle_to_local import particle_to_local
from flops import flops
# from check_interlacing_sigma_pos import check_interlacing_sigma_pos
from check_interlacing_sigma_neg import check_interlacing_sigma_neg
from FMM_fixed_expansion import FMM_fixed_expansion


# Parameters
scaling = 0
max_iter = 10  # Maximum Newton iterations
# tol = 1e-6

# Preallocate containers
setup_time = np.zeros((10, 1))
fmm_timer = np.zeros(max_iter)
fmm_timer_fixed = np.zeros(max_iter)
exact_timer = np.zeros(max_iter)

fmm_cauchy_adaptive = [None for _ in range(10)]
fmm_inv_square_adaptive = [None for _ in range(10)]

fmm_y = [None for _ in range(10)]
flop_counts = [None for _ in range(10)]

fmm_cauchy_fixed = [None for _ in range(10)]
fmm_inv_square_fixed = [None for _ in range(10)]
fmm_y_fixed = [None for _ in range(10)]

exact_cauchy = [None for _ in range(10)]
exact_inv_square = [None for _ in range(10)]

exact_y = [None for _ in range(10)]

abs_error_cauchy = [None for _ in range(10)]
abs_error_inv_square = [None for _ in range(10)]

rel_error = [None for _ in range(10)]
#==================================================
#        LOAD DATA
#==================================================
# data = loadmat("test_data_N_512.mat")
# data2 = data["data_start"]

# x = np.squeeze(data["x"])
# y = np.squeeze(data["y"])
# q = np.squeeze(data["q"])
# sigma = 1
 

# # Load PDE data
# data = np.load("test_data_trun_M2L/Strum_Louville_2000.npz")
# data = np.load("test_data_trun_M2L/Gauss_Laguerre_2000.npz")

data = np.load("test_data_trun_M2L/SL2_N_2000.npz")

x = np.squeeze(data["x"])
y = np.squeeze(data["y"])
q = np.squeeze(data["q"])
sigma = np.squeeze(data["sigma"])

q = q**2

    
for k in range(12,13):
    eps = 10.0**(-k)
    print("epsilon = ", eps)
        
    start_time = time.perf_counter()

    u1, u2, post_ord, ch, Px, Py, I, nflops, neighbor, r \
        = FMM_trun_M2L_expansion_sec_eqn(eps, x, y, q, scaling )
    
    setup_time = time.perf_counter() - start_time

    for iter in range(max_iter):
        
        fmm_start_time = time.perf_counter()    
        
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
        
        z1 = 1 + sigma*z1
        z2 = sigma*z2
        ynew = y - (z1 / z2)
        # print("||y - ynew|| =", np.linalg.norm(y - ynew))
        y = ynew
        
        fmm_cauchy_adaptive[iter] = z1.copy()  # store function values
        fmm_inv_square_adaptive[iter] = z2.copy()
        
        fmm_y[iter] = ynew.copy()  # store updated targets
        
        fmm_timer[iter] = time.perf_counter() - fmm_start_time + setup_time
        
        flop_counts = nflops

       
ok_fmm, num_failed_fmm, failed_indices_fmm = check_interlacing_sigma_neg(x, fmm_y[iter])

r_fixed = r
#===================================================================================
#   FIXED EXPANSION ORDER
#===================================================================================

y = np.squeeze(data["y"])  # initializing targets

for r in [r_fixed]:
    
    start_time = time.perf_counter()
    
    u1, u2, post_ord, ch, Px, Py, I, nflops, neighbor, flop_counts_fixed \
    = FMM_fixed_expansion(r, x, y, q, scaling)
    # print("r = ", r)
    
    setup_time_fixed = time.perf_counter() - start_time 
    
    for iter in range(max_iter):
    
        fmm_start_time_fixed = time.perf_counter() 
        
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


        z1 = 1 + sigma*z1
        z2 = sigma*z2
        ynew = y - (z1 / z2)
        # print("||y - ynew|| =", np.linalg.norm(y - ynew))
        y = ynew
        
        fmm_cauchy_fixed[iter] = z1.copy()  # store function values
        fmm_inv_square_fixed[iter] = z2.copy()
        
        fmm_y_fixed[iter] = ynew.copy()  # store updated targets
        
        fmm_timer_fixed[iter] = time.perf_counter() - fmm_start_time_fixed + setup_time_fixed
        
        # flop_counts = nflops

           
ok_fmm_fixed, num_failed_fmm_fixed, failed_indices_fmm_fixed = check_interlacing_sigma_neg(x, fmm_y_fixed[iter])


###### Direct evaluation ##############
y = np.squeeze(data["y"])  # initialize 

for iter in range(max_iter):
    
    dtt = time.perf_counter()
    
    Ker1 = np.zeros(len(y), dtype=np.float64)
    Ker2 = np.zeros(len(y), dtype=np.float64)
    
    for i in range(len(y)):
        diff = x - y[i]
    
        Ker1[i] = np.sum(q / diff)
        Ker2[i] = np.sum(q / diff**2)  
    
    exact_sum1 = 1 + sigma*Ker1
    exact_sum2 = sigma*Ker2
    
    ynew = y - (exact_sum1 / exact_sum2)
    # print("||y - ynew|| exact =", np.linalg.norm(y - ynew))
    y = ynew
    exact_cauchy[iter] = exact_sum1
    exact_inv_square[iter] = exact_sum2
    exact_y[iter] = ynew
    exact_timer[iter] = time.perf_counter() - dtt
    
    abs_error_cauchy[iter] = np.abs(fmm_cauchy_adaptive[iter] - exact_cauchy[iter])
    abs_error_inv_square[iter] = np.abs(fmm_inv_square_adaptive[iter] - exact_inv_square[iter])
    
    rel_error[iter] = ( np.linalg.norm(fmm_cauchy_adaptive[iter] - exact_cauchy[iter])
                        / np.linalg.norm(exact_cauchy[iter])
                    )

ok_exact, num_failed_exact, failed_indices_exact = check_interlacing_sigma_neg(x, exact_y[iter])



#==========================================================================
#                       ERROR PLOTS
#==========================================================================

#=================== ABSOLUTE ERROR PLOT ==================================
#==================== CAUCHY KERNEL =======================================


import matplotlib.pyplot as plt


for i in range(0, 1): # plotting for the first Newton iterate
    plt.figure(figsize=(8, 5))

    indices = np.arange(len(abs_error_cauchy[i]))
    errors = np.asarray(abs_error_cauchy[i])

    # Exclude zero errors for logarithmic plotting
    mask = errors > 0

    plt.scatter(indices[mask], errors[mask], s=5)
    plt.yscale("log")

    plt.xlabel("Index $i$")
    plt.ylabel("Absolute error")
    plt.title("Absolute Error for the Cauchy Kernel")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

#===================== INVERSE SQUARE KERNEL ====================

for i in range(0, 1):
    plt.figure(figsize=(8, 5))

    indices = np.arange(len(abs_error_inv_square[i]))
    errors = np.asarray(abs_error_inv_square[i])

    # Exclude zero errors for logarithmic plotting
    mask = errors > 0

    plt.scatter(indices[mask], errors[mask], s=5)
    plt.yscale("log")

    plt.xlabel("Index $i$")
    plt.ylabel("Absolute error")
    plt.title("Absolute Error for the inverse square Kernel")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


#======================================================================
# PLOTTING THE TIMGING GRAPHS
#======================================================================


iterations = np.arange(1, len(fmm_timer) + 1)

plt.figure(figsize=(8, 5))

plt.plot(iterations, fmm_timer,
         marker='o', linewidth=1.5, label='Adaptive FMM')

plt.plot(iterations, fmm_timer_fixed,
         marker='^', linewidth=1.5, label='Fixed-order FMM')

plt.plot(iterations, exact_timer,
         marker='s', linewidth=1.5, label='Direct')

plt.xlabel('Newton iteration')
plt.ylabel('Time (s)')
plt.title('FMM and Direct Evaluation Time per Iteration')
plt.xticks(iterations)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ==================================================
# HISTOGRAM PLOTS
# ==================================================

bins = np.concatenate((10.0**np.arange(-13, 0), [np.inf]))

labels = [
    r"$10^{-13}$-$10^{-12}$",
    r"$10^{-12}$-$10^{-11}$",
    r"$10^{-11}$-$10^{-10}$",
    r"$10^{-10}$-$10^{-9}$",
    r"$10^{-9}$-$10^{-8}$",
    r"$10^{-8}$-$10^{-7}$",
    r"$10^{-7}$-$10^{-6}$",
    r"$10^{-6}$-$10^{-5}$",
    r"$10^{-5}$-$10^{-4}$",
    r"$10^{-4}$-$10^{-3}$",
    r"$10^{-3}$-$10^{-2}$",
    r"$10^{-2}$-$10^{-1}$",
    r"$>10^{-1}$"
]


# ==================================================
# Function to plot histogram
# ==================================================

def plot_residual_histogram(
        fmm_fun_val_adaptive,
        fmm_fun_val_fixed,
        exact_fun_val,
        index,
        n):

    # --------------------------------------------------
    # Absolute function values
    # --------------------------------------------------

    fmm_vals = np.abs(fmm_fun_val_adaptive[index])
    fmm_vals_fixed = np.abs(fmm_fun_val_fixed[index])
    exact_vals = np.abs(exact_fun_val[index])

    # --------------------------------------------------
    # Histogram counts
    # --------------------------------------------------

    fmm_counts, _ = np.histogram(fmm_vals, bins=bins)
    fmm_counts_fixed, _ = np.histogram(fmm_vals_fixed, bins=bins)
    exact_counts, _ = np.histogram(exact_vals, bins=bins)

    # --------------------------------------------------
    # Print counts
    # --------------------------------------------------

    print(f"\nAbsolute function value distribution for n = {n}")

    print(
        "Magnitude Bin".ljust(25),
        "FMM".rjust(8),
        "FMM_fixed".rjust(12),
        "Exact".rjust(8)
    )

    print("-" * 55)

    for lab, fc, fcf, ec in zip(
            labels,
            fmm_counts,
            fmm_counts_fixed,
            exact_counts):

        print(
            f"{lab:<25}"
            f"{fc:>8}"
            f"{fcf:>12}"
            f"{ec:>8}"
        )

    # ==================================================
    # Grouped bar chart
    # ==================================================

    x = np.arange(len(labels))

    width = 0.25

    plt.figure(figsize=(12, 6))

    # Adaptive FMM
    bars1 = plt.bar(
        x - width,
        fmm_counts,
        width,
        label="FMM_adaptive"
    )

    # Fixed-order FMM
    bars2 = plt.bar(
        x,
        fmm_counts_fixed,
        width,
        label="FMM_fixed"
    )

    # Exact
    bars3 = plt.bar(
        x + width,
        exact_counts,
        width,
        label="Exact"
    )

    # --------------------------------------------------
    # Counts above bars
    # --------------------------------------------------

    for bar in bars1:
        h = bar.get_height()

        if h > 0:
            plt.text(
                bar.get_x() + bar.get_width()/2,
                h,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    for bar in bars2:
        h = bar.get_height()

        if h > 0:
            plt.text(
                bar.get_x() + bar.get_width()/2,
                h,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    for bar in bars3:
        h = bar.get_height()

        if h > 0:
            plt.text(
                bar.get_x() + bar.get_width()/2,
                h,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    # --------------------------------------------------
    # Labels and formatting
    # --------------------------------------------------

    plt.xticks(
        x,
        labels,
        rotation=45,
        ha="right"
    )

    plt.xlabel(r"Absolute function value $|f(y)|$")
    plt.ylabel("Number of initial approximations")

    plt.title(
        rf"Absolute function value distribution for tolerance $\epsilon={eps}$"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ==================================================
# Call function
# ==================================================

plot_residual_histogram(
    fmm_cauchy_adaptive,
    fmm_cauchy_fixed,
    exact_cauchy,
    index=9,
    n=2000
)

#=======================================================================
#          PLOTTING SOURCE LOCATIONS AND SOURCE STRENGTHS
#=======================================================================

# q1 = np.sqrt(q) # original source strengths
q1 = q
plt.figure(figsize=(10, 5))

sc = plt.scatter(x, q1, c=q1, s=8, cmap='viridis')

plt.xscale('log')
plt.xlabel('Source location, $x_j$')
plt.ylabel('Source strength, $q_j^2$')
plt.title('Distribution of Source Particles')
plt.colorbar(sc, label='Source strength')
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


#======= SOURCE LOCATION PLOT ===================

plt.figure(figsize=(10, 3))

plt.scatter(x, np.zeros_like(x), c=q1, s=8, cmap='viridis')

plt.xscale('log')
plt.xlabel('Source location, $x_j$')
plt.yticks([])
plt.title('Spatial Distribution of Source Particles')
plt.colorbar(label='Source strength')
plt.grid(True, axis='x', which='both', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()                
        
#======== SOURCE POTENTIAL PLOT ====================

plt.figure(figsize=(8, 5))

plt.hist(q1, bins=50, edgecolor='black')
plt.xlabel('Source strength, $q_j^2$')
plt.ylabel('Frequency')
plt.title('Distribution of Source Strengths $q_j^2$')
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()   


        





