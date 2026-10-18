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

from FMM_trun_M2L_expansion_ratio_eps_tau import FMM_trun_M2L_expansion_ratio_eps_tau
# from particle_to_local import particle_to_local
# from flops import flops

scaling = 0

# data = loadmat("test_data_N_512.mat")
# data2 = data["data_start"]
# sigma = -1


data = np.load("test_data_trun_M2L/SL2_N_2000.npz")

x = np.squeeze(data["x"])
y = np.squeeze(data["y"])
q = np.squeeze(data["q"])
sigma = np.squeeze(data["sigma"])

q = q**2

results_adaptive = []
flop_counts_adaptive = []

import pandas as pd

# Store expansion orders for each epsilon
r_values_all = {}

tau = 0.6

for k in range(5,13):
    eps = 10.0**(-k)
    print("epsilon = ", eps)

    ratios, r_values    = FMM_trun_M2L_expansion_ratio_eps_tau(eps, x, y, q, scaling, tau)

    # Save the expansion orders for this epsilon
    r_values_all[eps] = np.asarray(r_values).ravel().copy()

# --------------------------------------------------
# Summary statistics for each epsilon
# --------------------------------------------------
summary = []

for eps, values in r_values_all.items():
    # Exclude non-finite values, if any
    values = values[np.isfinite(values)]

    if len(values) == 0:
        summary.append({
            "epsilon": eps,
            "Number of interactions": 0,
            "Minimum": np.nan,
            "Maximum": np.nan,
            "Mean": np.nan,
            "Median": np.nan
        })
        continue

    summary.append({
        "epsilon": eps,
        "Number of interactions": len(values),
        "Minimum": np.min(values),
        "Maximum": np.max(values),
        "Mean": np.mean(values),
        "Median": np.median(values)
    })

summary_df = pd.DataFrame(summary)

# Display the table
print("\nSummary Statistics of Selected M2L Expansion Orders with varying epsilon (ε)")
print(summary_df.to_string(
    index=False,
    formatters={
        "epsilon": lambda value: f"{value:.0e}",
        "Mean": lambda value: f"{value:.2f}",
        "Median": lambda value: f"{value:.2f}"
    }
))

# # Optional: save the summary table
# summary_df.to_csv("M2L_expansion_order_summary.csv", index=False)   
