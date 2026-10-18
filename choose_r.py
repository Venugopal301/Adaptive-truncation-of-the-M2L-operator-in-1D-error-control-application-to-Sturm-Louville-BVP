#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 09:02:55 2026

@author: venugopal
"""
import numpy as np

def choose_r(a, b, dx, dy, Q, eps, rmin=4, rmax=100):
    
    r_flops = 0
    D = abs(a - b)
    
    r_flops += 1
    Rs = dy / 2.0
    Rt = dx / 2.0
    r_flops += 2
    
    theta = (Rs + Rt) / D
    r_flops += 2
    
    # Empty source box
    if Q <= 0.0:
        return rmin, rmax, True

    # Non-convergent geometry
    if theta >= 1.0:
        return rmax, rmax, False

    rhs = eps * D * (1.0 - theta) / Q
    r_flops += 4

    if rhs >= 1.0:
        r_required = 0
    else:
        r_required = int(
            np.ceil(np.log(rhs) / np.log(theta))
        )
    r_flops += 1
    
    # r_required = int(np.ceil(0.5 * r_required))  
    r_flops += 1
    
    r = max(rmin, r_required)
    tolerance_met = (r <= rmax)
    # print("rmin =", rmin,
    #       "r_required = ", r_required, "tolerance_met = ", tolerance_met)
    

    r = min(r, rmax)
    
    return r, rmax, tolerance_met, r_flops
