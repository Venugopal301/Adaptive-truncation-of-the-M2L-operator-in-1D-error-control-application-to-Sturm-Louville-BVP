# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:51:28 2026

@author: se22pmat003
"""

import numpy as np
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, "utils")

sys.path.insert(0, utils_dir)
sys.path.insert(0, current_dir)

from particle_to_local import particle_to_local
from flops import flops
# from check_interlacing_sigma_pos import check_interlacing_sigma_pos
# from check_interlacing_sigma_neg import check_interlacing_sigma_neg
from FMM_fixed_expansion import FMM_fixed_expansion
from FMM_trun_M2L_expansion_flops import FMM_trun_M2L_expansion_flops


# Preallocate containers
# setup_time = np.zeros((10, 1))
elapsed_time = [[None for _ in range(2)] for _ in range(10)]
fmm_fun_val = [None for _ in range(10)]
fmm_y = [None for _ in range(10)]
flop_counts = [None for _ in range(10)]


# Parameters
scaling = 0
max_iter = 10  # Maximum Newton iterations
# tol = 1e-6


n_values = np.arange(5000, 50001, 5000)

rng = np.random.default_rng(12345)
sigma = 1

flops_adaptive_all = []
flops_fixed_all = []



for jj, n in enumerate(n_values):
    
    print(f"jj = {jj}, n = {n}")

    # Generate 2n random points
    z = rng.uniform(0.0, 1.0, 2 * n)
    z.sort()

    # Interlaced x and y
    x = z[0::2]
    y = z[1::2]
    q = rng.uniform(0.5, 1.0, n)
    q = q**2  # elementwise square
    
    r_max = []    

    for k in range(10,11):
        eps = 10.0**(-k)
        print("epsilon = ", eps)
        
        start_time = time.perf_counter()
    
        u1, u2, post_ord, ch, Px, Py, I, nflops, neighbor, r , flops_adaptive \
                = FMM_trun_M2L_expansion_flops(eps, x, y, q, scaling
                )
        setup_time = time.perf_counter() - start_time
        
        fmm_time_begin = time.perf_counter()        
                
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
       
                    
        fmm_time = time.perf_counter() - fmm_time_begin
        
        r_max.append(r)
        
            ###### Direct evaluation ##############
                
        exact_sum1 = np.zeros(len(y), dtype=np.float64)
        exact_sum2 = np.zeros(len(y), dtype=np.float64)
        
        for i in range(len(y)):
            diff = x - y[i]
        
            exact_sum1[i] = np.sum(q / diff)
            exact_sum2[i] = np.sum(q / diff**2)        
        
        err1 = np.linalg.norm(exact_sum1 - z1) / np.linalg.norm(exact_sum1)
        err2 = np.linalg.norm(exact_sum2 - z2) / np.linalg.norm(exact_sum2)
                               
        flops_adaptive_all.append({
                "n": n,
                "r": r,
                "epsilon": eps,
                "rel_err_fn": err1,
                "rel_err_der": err2,
                **flops_adaptive.copy()
            })
        
            
    # INITIALIZE DATA FOR FIXED EXPANSION ORDER 
    
    y = z[1::2]    
    nflops = 0
    err1 = 0
    err2 = 0
        
    #-------------------------------------------------------------------------
    #                  FIXED EXPANSION ORDER
    #-------------------------------------------------------------------------
    
    
    for r in r_max:     
        u1, u2, post_ord, ch, Px, Py, I, nflops, neighbor, flops_fixed \
            = FMM_fixed_expansion(r, x, y, q, scaling)
        # print("r = ", r)
    
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
                            flops_fixed["inverse_square"]["near_field"] += D1.shape[0] * (2 * D1.shape[1] - 1) + len(py)
        
        flops_fixed["cauchy"]["total"] = sum( flops_fixed["cauchy"][key]
            for key in ["P2M", "M2M", "M2L", "L2L", "L2P", "near_field"]
                )
        flops_fixed["inverse_square"]["total"] = sum( flops_fixed["inverse_square"][key]
            for key in ["P2M", "M2M", "M2L", "L2L", "L2P", "near_field"]
                )

        
        err1 = np.linalg.norm(exact_sum1 - z1) / np.linalg.norm(exact_sum1)
        err2 = np.linalg.norm(exact_sum2 - z2) / np.linalg.norm(exact_sum2)
        
        
        flops_fixed_all.append({
                "n": n,
                "r": r,
                "epsilon": eps,
                "rel_err_fn": err1,
                "rel_err_der": err2,
                **flops_fixed.copy()
            })
        
        
        
#-----------------------------------------------------------------
#       ANALYSE RESULTS
#-----------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Choose kernel: "cauchy" or "inverse_square"
kernel = "cauchy"
# kernel = "inverse_square"

# ---------------------------------------------------------
# 1. Convert saved lists into DataFrames
# ---------------------------------------------------------
df_adaptive = pd.DataFrame(flops_adaptive_all)
df_fixed = pd.DataFrame(flops_fixed_all)

for data in (df_adaptive, df_fixed):
    data["M2L"] = data[kernel].apply(lambda d: d["M2L"])
    data["total"] = data[kernel].apply(lambda d: d["total"])

adaptive = df_adaptive[
    ["n", "epsilon", "r", "rel_err_fn", "rel_err_der", "M2L", "total"]
].rename(columns={
    "r": "r_adaptive",
    "rel_err_fn": "err_fn_adaptive",
    "rel_err_der": "err_der_adaptive",
    "M2L": "M2L_adaptive",
    "total": "total_adaptive"
})

fixed = df_fixed[
    ["n", "epsilon", "r", "rel_err_fn", "rel_err_der", "M2L", "total"]
].rename(columns={
    "r": "r_fixed",
    "rel_err_fn": "err_fn_fixed",
    "rel_err_der": "err_der_fixed",
    "M2L": "M2L_fixed",
    "total": "total_fixed"
})

df = pd.merge(
    adaptive, fixed,
    on=["n", "epsilon"],
    how="inner",
    validate="one_to_one"
).sort_values(["n", "epsilon"])

# ---------------------------------------------------------
# 2. Calculate FLOP reductions
# ---------------------------------------------------------
df["M2L_reduction_%"] = np.where(
    df["M2L_fixed"] > 0,
    100 * (df["M2L_fixed"] - df["M2L_adaptive"])
    / df["M2L_fixed"],
    np.nan
)

df["total_reduction_%"] = np.where(
    df["total_fixed"] > 0,
    100 * (df["total_fixed"] - df["total_adaptive"])
    / df["total_fixed"],
    np.nan
)

# ---------------------------------------------------------
# 3. Print one table with headings 
# ---------------------------------------------------------
columns = [
    "epsilon", "n", "r_adaptive",
    "M2L_fixed", "M2L_adaptive", "M2L_reduction_%",
    "total_fixed", "total_adaptive", "total_reduction_%"
]

table = df[columns].rename(columns={
    "r_adaptive": "r",
    "M2L_fixed": "M2L (Fixed)",
    "M2L_adaptive": "M2L (Adaptive)",
    "M2L_reduction_%": "M2L Reduction (%)",
    "total_fixed": "Total (Fixed)",
    "total_adaptive": "Total (Adaptive)",
    "total_reduction_%": "Total Reduction (%)"
})

print("\nFLOP Comparison:", kernel)
print(table.to_string(
    index=False,
    formatters={
        "epsilon": lambda x: f"{x:.0e}",
        "M2L Reduction (%)": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A",
        "Total Reduction (%)": lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    }
))

# # Optional: save the table
# table.to_csv(f"flop_comparison_{kernel}.csv", index=False)

# ---------------------------------------------------------
# 4. Separate plot: Function relative error
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

for eps, group in df.groupby("epsilon", sort=True):
    group = group.sort_values("n")

    plt.plot(
        group["n"], group["err_fn_adaptive"],
        marker="o", linestyle="-",
        label=f"Adaptive, epsilon={eps:.0e}"
    )
    plt.plot(
        group["n"], group["err_fn_fixed"],
        marker="s", linestyle="--",
        label=f"Fixed, epsilon={eps:.0e}"
    )

plt.xlabel("$n$")
plt.ylabel("Relative error")
plt.title("Relative Error in the Cauchy kernel")
plt.yscale("log")
plt.grid(True, which="both", linestyle=":")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5. Separate plot: Derivative relative error
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

for eps, group in df.groupby("epsilon", sort=True):
    group = group.sort_values("n")

    plt.plot(
        group["n"], group["err_der_adaptive"],
        marker="o", linestyle="-",
        label=f"Adaptive, epsilon={eps:.0e}"
    )
    plt.plot(
        group["n"], group["err_der_fixed"],
        marker="s", linestyle="--",
        label=f"Fixed, epsilon={eps:.0e}"
    )

plt.xlabel("$n$")
plt.ylabel("Relative error")
plt.title("Relative Error in the Inverse Square kernel")
plt.yscale("log")
plt.grid(True, which="both", linestyle=":")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()














