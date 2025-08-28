import numpy as np
from scipy.constants import g
from pyfluids import Fluid, FluidsList, Input
from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.blends import makeCardForNewTemperature

# dt = 47.02e-3
dt = 42.41e-3
At = 0.25 * np.pi * dt**2
eps = 3.7594

# thrust = 7000
# pc = 28.66e5
# mdot_o = 2.768
# mdot_f = 0.794

# thrust = 5925
# pc = 31.8e5
# mdot_o = 2.330
# mdot_f = 0.760

# thrust = 5628.6
# pc = 29.95e5
# mdot_o = 2.226
# mdot_f = 0.826

thrust = 5724
pc = 30.45e5
mdot_o = 2.227
mdot_f = 0.762

mdot = mdot_o + mdot_f
OF = mdot_o / mdot_f
print(f"OF Ratio: {OF:.3f}")

cf = thrust / (pc * At)
print(f"CF: {cf:.4f}")

cea_n2o_temp = 25 # deg C
exp_n2o_temp = 0 # deg C

N2O_rt = Fluid(FluidsList.NitrousOxide)
N2O_sc = Fluid(FluidsList.NitrousOxide)
N2O_rt.update(Input.quality(0), Input.temperature(cea_n2o_temp))
N2O_sc.update(Input.quality(0), Input.temperature(exp_n2o_temp))
dT = (N2O_rt.temperature - N2O_sc.temperature) * 9/5  # Convert to degR
dH = (N2O_rt.enthalpy - N2O_sc.enthalpy) * 0.0004299226 # Convert to BTU / lbm
cp_avg = dH / dT
N2O_subcooled = makeCardForNewTemperature( ceaName='N2O', newTdegR=((9/5)*(N2O_sc.temperature + 273.15)), CpAve=cp_avg, MolWt=44.013)
cea = CEA_Obj(
    oxName = 'N2O',
    fuelName = 'Isopropanol',
    isp_units= 'sec',
    cstar_units = 'm/s',
    pressure_units='Bar',
    temperature_units='K',
    sonic_velocity_units='m/s',
    enthalpy_units='J/kg',
    density_units='kg/m^3',
    specific_heat_units='J/kg-K',
    viscosity_units='centipoise', # stored value in pa-s
    thermal_cond_units='W/cm-degC', # stored value in W/m-K
    fac_CR = 4.7059,
    make_debug_prints=False)

rt_cstar_ideal = cea.get_Cstar(pc/1e5, OF)
rt_isp_ideal = cea.estimate_Ambient_Isp(pc/1e5, OF, eps, 1.01325)[0]

cea = CEA_Obj(
    oxName = N2O_subcooled,
    fuelName = 'Isopropanol',
    isp_units='sec',
    cstar_units = 'm/s',
    pressure_units='Bar',
    temperature_units='K',
    sonic_velocity_units='m/s',
    enthalpy_units='J/kg',
    density_units='kg/m^3',
    specific_heat_units='J/kg-K',
    viscosity_units='centipoise', # stored value in pa-s
    thermal_cond_units='W/cm-degC', # stored value in W/m-K
    fac_CR = 4.7059,
    make_debug_prints=False)

sc_cstar_ideal = cea.get_Cstar(pc/1e5, OF)
sc_isp_ideal = cea.estimate_Ambient_Isp(pc/1e5, OF, eps, 1.01325)[0]

isp_exp = thrust / (mdot * g)
cstar_exp = pc * At / mdot

print(f"ISP (measured): {isp_exp:.2f} s")
print(f"ISP (ideal) - room temp: {rt_isp_ideal:.2f} s")
print(f"ISP (ideal) - subcooled: {sc_isp_ideal:.2f} s")
print(f"c* (measured): {cstar_exp:.2f} m/s")
print(f"c* (ideal) - room temp: {rt_cstar_ideal:.2f} m/s")
print(f"c* (ideal) - subcooled: {sc_cstar_ideal:.2f} m/s")
print(f"η_c* - room temp: {cstar_exp/rt_cstar_ideal:.3f}")
print(f"η_c* - subcooled: {cstar_exp/sc_cstar_ideal:.3f}")