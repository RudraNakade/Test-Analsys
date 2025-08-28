import numpy as np

inj_A = 60 * 0.25 * np.pi * (1.5e-3)**2

# Experimental
mdot = 2.33
rho = 906
dP = 4.87e5

CdA_exp = mdot / np.sqrt(2 * rho * dP)

Cd_exp = CdA_exp / inj_A

print(f"Injector Area: {inj_A:.2e} m^2")
print(f"CdA Experimental: {CdA_exp:.2e} m^2")
print(f"Experimental Cd: {Cd_exp:.4f}")