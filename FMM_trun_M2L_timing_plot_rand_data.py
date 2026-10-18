
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:51:28 2026

@author: se22pmat003
"""

import numpy as np
import matplotlib.pyplot as plt
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
from check_interlacing_sigma_pos import check_interlacing_sigma_pos
# from check_interlacing_sigma_neg import check_interlacing_sigma_neg
from FMM_fixed_expansion import FMM_fixed_expansion


# Parameters
scaling = 0
max_iter = 1  # Maximum Newton iterations
# tol = 1e-6


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
 

# # Load data

# data = np.load("test_data_trun_M2L/SL2_N_2000.npz")

# x = np.squeeze(data["x"])
# y = np.squeeze(data["y"])
# q = np.squeeze(data["q"])
# sigma = np.squeeze(data["sigma"])

# q = q**2


n_values = np.arange(5000, 50001, 5000)

rng = np.random.default_rng(12345)
sigma = 1

# Store the first-iteration timing for every problem size
time_adaptive_all = np.zeros(len(n_values))
time_fixed_all = np.zeros(len(n_values))
time_direct_all = np.zeros(len(n_values))

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
    
    # Preallocate containers
    # setup_time = np.zeros((max_iter, 1))
    fmm_timer_adaptive = np.zeros(max_iter)
    fmm_timer_fixed = np.zeros(max_iter)
    exact_timer = np.zeros(max_iter)

    
    for k in range(10,11):
        eps = 10.0**(-k)
        print("epsilon = ", eps)
        
    #========================================================
    #           ADAPTIVE EXPANSION ORDER 
    #========================================================    
    
            
        start_time_adaptive = time.perf_counter()
    
        u1, u2, post_ord, ch, Px, Py, I, nflops, neighbor, r \
            = FMM_trun_M2L_expansion_sec_eqn(eps, x, y, q, scaling )
        
        setup_time_adaptive = time.perf_counter() - start_time_adaptive
    
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
            
            fmm_timer_adaptive[iter] = time.perf_counter() - fmm_start_time + setup_time_adaptive
            
             
            flop_counts = nflops
    
        time_adaptive_all[jj] = fmm_timer_adaptive[iter]
       
        print('FMM adaptive interlacing failure')           
        ok_fmm, num_failed_fmm, failed_indices_fmm = check_interlacing_sigma_pos(x, fmm_y[iter])
        
    r_fixed = r
    
    #===================================================================================
    #   FIXED EXPANSION ORDER
    #===================================================================================
    
    y = z[1::2] # intialize the target vector
    
    for r in [r_fixed]:
        
        start_time_fixed = time.perf_counter()
        
        u1, u2, post_ord, ch, Px, Py, I, nflops, neighbor, flop_counts_fixed \
        = FMM_fixed_expansion(r, x, y, q, scaling)
        # print("r = ", r)
        
        setup_time_fixed = time.perf_counter() - start_time_fixed
        
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
            
        time_fixed_all[jj] = fmm_timer_fixed[iter]        
                # flop_counts = nflops
        
        print('FMM fixed interlacing failure')               
        ok_fmm_fixed, num_failed_fmm_fixed, failed_indices_fmm_fixed = check_interlacing_sigma_pos(x, fmm_y_fixed[iter])
        

    #========================================================
    #           DIRECT EVALUATION
    #========================================================    
    
    y = z[1::2] # intialize the target vector

    for iter in range(max_iter):
        
        exact_timer_begin = time.perf_counter()
        
        Ker1 = np.zeros(len(y), dtype=np.float64)
        Ker2 = np.zeros(len(y), dtype=np.float64)
        
        for i in range(len(y)):
            diff = x - y[i]
        
            Ker1[i] = np.sum(q / diff)
            Ker2[i] = np.sum(q / diff**2)        

        Ker1  = 1 + sigma*Ker1
        Ker2 = sigma*Ker2
    
        ynew = y - (Ker1 / Ker2)
        # print("||y - ynew|| exact =", np.linalg.norm(y - ynew))
        y = ynew
        exact_cauchy[iter] = Ker1
        exact_inv_square[iter] = Ker2
        exact_y[iter] = ynew

        exact_timer[iter] = time.perf_counter() - exact_timer_begin
        
        # abs_error_cauchy[iter] = np.abs(fmm_cauchy_adaptive[iter] - exact_cauchy[iter])
        # abs_error_inv_square[iter] = np.abs(fmm_inv_square_adaptive[iter] - exact_inv_square[iter])
        
        # rel_error[iter] = ( np.linalg.norm(fmm_cauchy_adaptive[iter] - exact_cauchy[iter])
        #                     / np.linalg.norm(exact_cauchy[iter])
        #                 )
   
    time_direct_all[jj] = exact_timer[0]    
   
    print('exact interlacing failure')
    ok_exact, num_failed_exact, failed_indices_exact = check_interlacing_sigma_pos(x, exact_y[iter])


    #======================================================================
    # PLOTTING THE TIMGING GRAPHS
    #======================================================================
    

    # #--------------- ALL TEN NEWTON ITERATIONS -----------    

    # iterations = np.arange(1, len(fmm_timer_adaptive) + 1)
    
    # plt.figure(figsize=(8, 5))
    
    # plt.plot(iterations, fmm_timer_adaptive,
    #          marker='o', linewidth=1.5, label='Adaptive FMM')
    
    # plt.plot(iterations, fmm_timer_fixed,
    #          marker='^', linewidth=1.5, label='Fixed-order FMM')
    
    # plt.plot(iterations, exact_timer,
    #          marker='s', linewidth=1.5, label='Direct')
    
    # plt.xlabel('Newton iteration')
    # plt.ylabel('Time (s)')
    # plt.title(f'FMM and Direct Evaluation Time per Iteration (n = {n})')
    # plt.xticks(iterations)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.show()



#------------ NEWTON FIRST ITERATE ONLY----------------    
plt.figure(figsize=(8, 5))

plt.plot(
    n_values, time_adaptive_all,
    marker='o', linewidth=1.8,
    label='Adaptive FMM'
)

plt.plot(
    n_values, time_fixed_all,
    marker='^', linewidth=1.8,
    label='Fixed-order FMM'
)

plt.plot(
    n_values, time_direct_all,
    marker='s', linewidth=1.8,
    label='Direct evaluation'
)

plt.xlabel('Problem size $n$')
plt.ylabel('Execution time (s)')
plt.title('Timing Comparison ')

plt.xticks(n_values, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

        
    
        
        
        
    
    

        





