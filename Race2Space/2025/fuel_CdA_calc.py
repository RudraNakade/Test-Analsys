import numpy as np

# Expected
pintle_Cd = 0.75
film_Cd = 0.65

pintle_A = 21.921e-6
film_A = 5.404e-6

pintle_CdA = pintle_Cd * pintle_A
film_CdA = film_Cd * film_A

CdA_design = pintle_CdA + film_CdA

# Experimental
mdot = 0.761
rho = 790
dP = 12.2e5

CdA_exp = mdot / np.sqrt(2 * rho * dP)

overall_Cd = CdA_exp / (pintle_A + film_A)

pintle_CdA_exp = CdA_exp - film_CdA
pintle_Cd_exp = pintle_CdA_exp / pintle_A

print(f"CdA Design: {CdA_design:.2e} m^2")
print(f"CdA Experimental: {CdA_exp:.2e} m^2")
print(f"Overall Experimental Cd: {overall_Cd:.4f}")