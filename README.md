# Adaptive Truncation of the M2L Operator for the One-Dimensional Cauchy Kernel

This repository contains Python implementations and numerical experiments for an adaptive truncation strategy for the multipole-to-local (M2L) operator in the one-dimensional fast multipole method (FMM).

The method selects an expansion order for each source–target interaction using a truncation-error criterion. The numerical experiments compare adaptive M2L truncation with fixed-order M2L expansions and, where indicated, direct kernel evaluation.

## Contents

The scripts are grouped by the main numerical investigation. 

### 1. M2L bound ratios versus tolerance and separation ratio


| Script | Purpose |
|---|---|
| `FMM_trun_M2L_ratio_versus_tau_Sturm.py` | Computes summary statistics of bound ratios while varying the separation ratio, `tau`. |
| `FMM_trun_M2L_ratio_versus_eps_Sturm.py` | Computes summary statistics of bound ratios while varying the error tolerance, `epsilon`. |
| `FMM_trun_M2L_ratio_plot_Sturm.py` | Plots the ratio of bounds and the number of M2L calls assigned to each expansion order. |

The summary statistics include the number of ratios, number of finite values, minimum, maximum, mean, median, and standard deviation, as applicable.

### 2. Selected M2L expansion order versus tolerance and separation ratio


| Script | Purpose |
|---|---|
| `FMM_trun_M2L_r_versus_eps_Sturm.py` | Studies the selected M2L expansion orders as the tolerance, `epsilon`, varies. The reported summary uses `tau = 0.6`. |
| `FMM_trun_M2L_r_versus_tau_Sturm.py` | Studies the selected M2L expansion orders as the separation ratio, `tau`, varies. |

The scripts report summary statistics such as the number of interactions, minimum and maximum selected orders, mean, and median.

### 3. FLOP counts and relative errors


| Script | Purpose |
|---|---|
| `FMM_trun_M2L_flop_counts_rel_err_Sturm.py` | Examines total FLOP counts versus expansion order, relative errors for the Cauchy and inverse-square kernels, FLOP reductions, and the distribution of adaptive and fixed expansion orders. |
| `FMM_trun_M2L_flop_counts_rel_err_rand_data.py` | Compares fixed and adaptive FLOP counts for randomly generated data over multiple problem sizes and plots function and derivative relative errors. |

The Sturm–Liouville experiments produce:
- Total FLOPs versus expansion order for the Cauchy and inverse-square kernels.
- Relative-error plots for both kernels.
- Tables of percentage reductions in M2L FLOPs and total FLOPs.
- Pie charts showing the distribution of adaptive and fixed expansion orders for the Cauchy kernel.

The random-data experiments report FLOP comparisons for problem sizes from `n = 5000` to `n = 50000`, at `epsilon = 1e-10`, and provide separate plots for function and derivative relative errors.

#### Reported FLOP reductions for randomly generated data

The following results are reported by `FMM_trun_M2L_flop_counts_rel_err_rand_data.py`.

| `n` | Selected order `r` | M2L reduction (%) | Total FLOP reduction (%) |
|---:|---:|---:|---:|
| 5,000 | 52 | 40.16 | 21.41 |
| 10,000 | 60 | 52.62 | 29.53 |
| 15,000 | 65 | 55.01 | 29.64 |
| 20,000 | 64 | 57.00 | 32.78 |
| 25,000 | 66 | 57.82 | 31.36 |
| 30,000 | 66 | 53.12 | 29.33 |
| 35,000 | 66 | 57.62 | 33.86 |
| 40,000 | 56 | 41.21 | 22.88 |
| 45,000 | 49 | 22.87 | 11.58 |
| 50,000 | 67 | 57.52 | 31.39 |

Here, the reduction is calculated relative to the fixed-order implementation:

\[
\text{Reduction (\%)} =
\frac{\text{Fixed FLOPs} - \text{Adaptive FLOPs}}
     {\text{Fixed FLOPs}} \times 100.
\]

For the reported random-data tests, the M2L FLOP reduction ranges from **22.87% to 57.82%**, while the total FLOP reduction ranges from **11.58% to 33.86%**.

### 4. Secular-equation evaluation and timing


| Script | Purpose |
|---|---|
| `FMM_trun_M2L_sec_eqn_Sturm.py` | Plots absolute errors of Cauchy- and inverse-square-kernel evaluations relative to direct evaluation at the tenth iteration; plots histograms of absolute function values, execution-time comparisons, and source locations and strengths. |
| `FMM_trun_M2L_rand_data_timing_comparison.py` | Compares the time required for a single evaluation of the Cauchy and inverse-square kernels for randomly generated data. |

These experiments assess numerical accuracy and computational performance for secular-equation evaluation and related kernel calculations.

## Method overview

The experiments investigate an interaction-dependent M2L expansion order rather than applying one uniform order to every M2L interaction.

The principal comparisons are:
- **Adaptive M2L:** the expansion order is selected separately for each source–target interaction according to the implemented truncation criterion.
- **Fixed-order M2L:** the prescribed expansion order is used for all M2L interactions.
- **Direct evaluation:** used as a reference in the kernel-error and timing experiments where indicated.

The numerical assessment includes approximation errors, floating-point operation (FLOP) counts, and execution time. The local truncation-error criterion is used to select interaction-dependent orders; the global errors of the complete evaluation are assessed numerically.

## Requirements

The scripts are written in Python. The required packages depend on the imports used in each script. Common scientific-computing dependencies may include:

- Python 3
- NumPy
- SciPy
- Matplotlib

