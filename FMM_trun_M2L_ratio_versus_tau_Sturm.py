#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:22:46 2026

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


import pandas as pd
import matplotlib.pyplot as plt

# Fixed truncation tolerance
eps = 1e-12

# Vary the separation parameter
tau_values = np.round(np.arange(0.1, 1.0, 0.1), 1)

# Store results
ratio_results = []
all_ratios = {}
all_r_values = {}

for tau in tau_values:

    print(f"\nTau = {tau:.1f}, epsilon = {eps:.0e}")

    ratios, r_values = FMM_trun_M2L_expansion_ratio_eps_tau(
        eps, x, y, q, scaling, tau
    )

    # Convert to NumPy arrays
    ratios = np.asarray(ratios, dtype=float).ravel()
    r_values = np.asarray(r_values).ravel()

    # Keep finite ratios for statistical analysis
    valid_ratios = ratios[np.isfinite(ratios)]

    if valid_ratios.size == 0:
        print("No finite ratios available.")
        continue

    # Save individual values for subsequent analysis
    all_ratios[tau] = ratios.copy()
    all_r_values[tau] = r_values.copy()

    # Calculate summary statistics
    ratio_results.append({
    "tau": tau,
    "Number of ratios": len(ratios),
    "Number finite": len(valid_ratios),
    "Minimum": np.min(valid_ratios),
    "Maximum": np.max(valid_ratios),
    "Mean": np.mean(valid_ratios),
    "Median": np.median(valid_ratios),
    "Standard deviation": np.std(valid_ratios)
        })

# Create summary table
df_ratios = pd.DataFrame(ratio_results)

print("\nSummary statistics of ratios for varying tau:")
print(
    df_ratios.to_string(
        index=False,
        float_format=lambda v: f"{v:.6e}"
    )
)

# Optional: save the table
# df_ratios.to_csv("ratio_statistics_vs_tau_eps_1e-12.csv", index=False)

# Plot ratio statistics against tau
if not df_ratios.empty:

    df_plot = df_ratios.sort_values("tau")

    plt.figure(figsize=(8, 5))

    plt.plot(
        df_plot["tau"], df_plot["Minimum"],
        marker="o", linewidth=1.6, label="Minimum"
    )
    plt.plot(
        df_plot["tau"], df_plot["Maximum"],
        marker="s", linewidth=1.6, label="Maximum"
    )
    plt.plot(
        df_plot["tau"], df_plot["Mean"],
        marker="^", linewidth=1.6, label="Mean"
    )
    plt.plot(
        df_plot["tau"], df_plot["Median"],
        marker="D", linewidth=1.6, label="Median"
    )

    plt.xlabel(r"Separation parameter $\tau$")
    plt.ylabel("Ratio")
    plt.title(
        r"Ratio Statistics for Varying $\tau$ "
        r"($\epsilon=10^{-12}$)"
    )

    plt.xticks(tau_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
