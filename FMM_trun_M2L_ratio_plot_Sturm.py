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

for k in range(10,11):
    eps = 10.0**(-k)
    print("epsilon = ", eps)

    ratios, r_values    = FMM_trun_M2L_expansion_ratio_eps_tau(eps, x, y, q, scaling, tau )

#================================================================
#  PLOT THE RATIO
#================================================================     

import matplotlib.pyplot as plt

interaction_indices = np.arange(1, len(ratios) + 1)

plt.figure(figsize=(8, 5))
plt.semilogy(
    interaction_indices,
    ratios,
    marker='o',
    markersize=4,
    linewidth=2
)

plt.axhline(1.0, linestyle='--', linewidth=2)

plt.xlabel("Interaction index")
plt.ylabel(r"$\mathrm{Actual\ error}/\mathrm{Theoretical\ bound}$")
plt.title(f"M2L truncation error ratio for epsilon {eps}")
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# print("Number of interactions =", len(ratios))
# print("Maximum ratio =", np.max(ratios))
# print("Minimum ratio =", np.min(ratios))
# print("Mean ratio =", np.mean(ratios))                    

#================================================================
#  PLOTTING NUMBER OF CALLS TO EACH EXPANSION ORDER 
#================================================================     

from collections import Counter


# Count occurrences of each expansion order
r_counts = Counter(r_values)

# Sort by expansion order
r_sorted = sorted(r_counts)

counts = [r_counts[r] for r in r_sorted]

# Plot
plt.figure(figsize=(8, 5))
plt.bar(r_sorted, counts)

plt.xlabel("Expansion order $r$")
plt.ylabel("Number of M2L calls at leaf level")
plt.title("Distribution of Adaptive Expansion Orders")
plt.xticks(r_sorted)
plt.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.show()
    
