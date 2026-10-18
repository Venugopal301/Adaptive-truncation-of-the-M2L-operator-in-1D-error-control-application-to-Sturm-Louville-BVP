#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:04:41 2026

@author: venugopal
"""

import numpy as np
# from scipy.io import loadmat
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, "utils")

sys.path.insert(0, utils_dir)
sys.path.insert(0, current_dir)

from FMM_trun_M2L_expansion_ratio_eps_tau import FMM_trun_M2L_expansion_ratio_eps_tau
# from particle_to_local import particle_to_local
# from flops import flops

scaling = 0

# data = loadmat("test_data_N_512.mat")
# data2 = data["data_start"]
# sigma = -1

# Load PDE data
# data = np.load("test_data_trun_M2L/Gauss_Laguerre_2000.npz")

data = np.load("test_data_trun_M2L/SL2_N_2000.npz")

x = np.squeeze(data["x"])
y = np.squeeze(data["y"])
q = np.squeeze(data["q"])
sigma = np.squeeze(data["sigma"])

q = q**2

results_adaptive = []
flop_counts_adaptive = []

tau = 0.6


# Store the results
ratio_results = []
all_ratios = {}
all_r_values = {}

for k in range(7, 13):
    eps = 10.0**(-k)
    print(f"\nEpsilon = {eps:.0e}")

    ratios, r_values = FMM_trun_M2L_expansion_ratio_eps_tau(
        eps, x, y, q, scaling, tau
    )

    # Convert to NumPy arrays
    ratios = np.asarray(ratios, dtype=float).ravel()
    r_values = np.asarray(r_values).ravel()

    # Exclude non-finite ratios from the statistics
    valid_ratios = ratios[np.isfinite(ratios)]

    if valid_ratios.size == 0:
        print("No finite ratios available.")
        continue

    # Store individual results for later analysis
    all_ratios[eps] = ratios.copy()
    all_r_values[eps] = r_values.copy()

    # Compute statistics
    ratio_results.append({
        "epsilon": eps,
        "Number of ratios": len(ratios),
        "Number finite": len(valid_ratios),
        "Minimum": np.min(valid_ratios),
        "Maximum": np.max(valid_ratios),
        "Mean": np.mean(valid_ratios),
        "Median": np.median(valid_ratios),
        "Standard deviation": np.std(valid_ratios)
    })



import pandas as pd
import matplotlib.pyplot as plt


# Create summary table
df_ratios = pd.DataFrame(ratio_results)

print("\nSummary statistics of ratios for varying epsilon:")
print(df_ratios.to_string(index=False, float_format=lambda v: f"{v:.6e}"))

# # Save results
# df_ratios.to_csv("ratio_statistics_vs_epsilon.csv", index=False)


# Sort by decreasing epsilon: 1e-5, 1e-6, ..., 1e-12
df_plot = df_ratios.sort_values("epsilon", ascending=False)

plt.figure(figsize=(8, 5))

plt.plot(df_plot["epsilon"], df_plot["Minimum"],
         marker="o", linewidth=1.6, label="Minimum")

plt.plot(df_plot["epsilon"], df_plot["Maximum"],
         marker="s", linewidth=1.6, label="Maximum")

plt.plot(df_plot["epsilon"], df_plot["Mean"],
         marker="^", linewidth=1.6, label="Mean")

plt.plot(df_plot["epsilon"], df_plot["Median"],
         marker="D", linewidth=1.6, label="Median")

plt.xscale("log")
plt.gca().invert_xaxis()

plt.xlabel(r"Truncation tolerance $\epsilon$")
plt.ylabel("Ratio")
plt.title(r"Statistics of the Ratio Bound for Varying $\epsilon$")
plt.xticks(
    df_plot["epsilon"],
    [rf"$10^{{{int(np.log10(eps))}}}$"
     for eps in df_plot["epsilon"]]
)

plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()