# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 04:57:48 2026

@author: John Arimboor
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Constants & Parameters
N = 100000            # Increased samples for better accuracy
deltax = 0.5          # Max step size (nm)
kb = 1.380649e-23     # Boltzmann Constant (J/K)
T = 300.0             # Temperature (K)
kT = kb * T
kf = 10.0             # Force constant (N/m)
nm = 1e-9             # Nanometer conversion

def calculate_energy(x_nm):
    # V = 0.5 * kf * x^2
    return 0.5 * kf * (x_nm * nm)**2

# 2. Initialization
x1 = 0.0
E1 = calculate_energy(x1)
Etot = 0.0
E2tot = 0.0
x_history = []

# 3. Metropolis Loop
for i in range(N):
    # Corrected: Symmetric random step [-deltax, deltax]
    x2 = x1 + (np.random.rand() * 2 - 1) * deltax
    E2 = calculate_energy(x2)
    
    delta_E = E2 - E1
    
    # Metropolis Acceptance Criterion
    if delta_E <= 0 or np.exp(-delta_E / kT) > np.random.rand():
        x1 = x2
        E1 = E2
    
    # Always accumulate current state
    Etot += E1
    E2tot += E1**2
    x_history.append(x1)

# 4. Data Analysis
E_avg = Etot / N
E2_avg = E2tot / N
Cv = (E2_avg - E_avg**2) / (kb * T**2)

print(f"Average Potential Energy: {E_avg:.4e} J")
print(f"Theoretical Energy (0.5kT): {0.5 * kT:.4e} J")
print(f"Heat Capacity (Cv): {Cv:.4e} J/K")