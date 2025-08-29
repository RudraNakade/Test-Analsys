from test_analysis import *
from rocketcea.cea_obj_w_units import CEA_Obj
from os import system
from pyfluids import Fluid, FluidsList, Input
system('cls')

def spi_mdot(CdA, rho, dP):
    """Calculate mass flow rate through a single-phase injector."""
    # if np.any(dP < 0):
    #     raise ValueError("Pressure drop (dP) must be non-negative.")
    # dP = np.maximum(dP, 1e-6)  # Prevent division by zero or negative sqrt
    # return CdA * np.sqrt(2 * rho * dP)
    return np.multiply(CdA * np.ones_like(dP), np.sqrt(2e5 * rho * dP), out=np.zeros_like(dP), where=(dP>0))

def spi_dP(mdot, CdA, rho):
    """Calculate pressure drop across a single-phase injector."""
    return 1e5 * (mdot / CdA)**2 / (2 * rho)

def spi_CdA(mdot, dP, rho):
    """Calculate CdA of a single-phase injector."""
    # if np.any(dP < 0):
    #     raise ValueError("Pressure drop (dP) must be non-negative.")
    # dP = np.maximum(dP, 1e-6)  # Prevent division by zero or negative sqrt
    # return mdot / np.sqrt(2e5 * rho * dP)
    return np.divide(mdot, np.sqrt(2e5 * rho * dP), out=np.zeros_like(mdot), where=(dP>0))

# Import data and make channels
base_data_folder = "C:\\Users\\Rudra\\Desktop\\Propulsion\\Test Data\\2025\\23_02_25_HOPPER_ENGINE"
test_folder = "hotfire_2"
data_folder = base_data_folder + "\\" + test_folder + "\\"

## Board sensor outputs:
# Kermit: ch0sens, ch1sens, ch2sens, ch3sens, temp0, temp1
# Stark: servo_voltage, ch0sens, ch1sens, ch2sens, ch3sens, ch4sens, ch5sens, flowmeter, oxAngle, fuelAngle
# Greg: Feedforward, FuelTankP, regAngle, Proportional_Term, Kp

## File names
kermit_tank_data_file = "kermit_telem_trim.csv"
# kermit_fill_data_file = ""
# kermit_hose_data_file = ""
# greg_ereg_data_file   = ""
stark_data_file       = "stark_telem_trim.csv"

## Datasets
kermit_tank_data = Dataset(data_folder + kermit_tank_data_file) # N2, Ox, Fuel tank PTs
# kermit_fill_data = Dataset(data_folder + kermit_fill_data_file) # Filling panel PTs, Tank weight LC
# kermit_hose_data = Dataset(data_folder + kermit_hose_data_file) # Hoses PTs
# greg_ereg_data   = Dataset(data_folder + greg_ereg_data_file)   # Ereg PTs
stark_data       = Dataset(data_folder + stark_data_file)       # Fuel + Ox Inj, Chamber PTs, Fuel Inj TC

## Reference time
# base_time = np.array(kermit_tank_data.backend_time)
base_time = np.array(stark_data.backend_time)

## Make channels
n2_tank_p = kermit_tank_data.convert_to_channel('ch1sens', 'N2 Tank Pressure')
ox_tank_p = kermit_tank_data.convert_to_channel('ch0sens', 'Ox Tank Pressure')
fuel_tank_p = stark_data.convert_to_channel('ch4sens', 'Fuel Tank Pressure')

fuel_inj_p = stark_data.convert_to_channel('ch1sens', 'Fuel Injector Pressure', kermit_tank_data)
fuel_inj_t = kermit_tank_data.convert_to_channel('temp1', 'Fuel Injector Temperature')
ox_inj_p = stark_data.convert_to_channel('ch0sens', 'Ox Injector Pressure', kermit_tank_data)
chamber_p = stark_data.convert_to_channel('ch3sens', 'Chamber Pressure', kermit_tank_data)

ox_valve_angle = stark_data.convert_to_channel('oxAngle', 'Ox Valve Angle', kermit_tank_data)
fuel_valve_angle = stark_data.convert_to_channel('fuelAngle', 'Fuel Valve Angle', kermit_tank_data)

fuel_vdot = stark_data.convert_to_channel('flowmeter', 'Fuel vdot', kermit_tank_data, gain=1e-3) # converted to m^3/s

# rocket_weight = kermit_fill_data.convert_to_channel('', 'Tank Weight')

# # Ereg channels
# ereg_fuel_tank_p = greg_ereg_data.convert_to_channel('FuelTankP', 'Ereg Fuel Tank Pressure', kermit_tank_data)
# ereg_ff = greg_ereg_data.convert_to_channel('Feedforward', 'Ereg Feedforward', kermit_tank_data)
# ereg_angle = greg_ereg_data.convert_to_channel('regAngle', 'Ereg Valve Angle', kermit_tank_data)
# ereg_p_term = greg_ereg_data.convert_to_channel('Proportional_Term', 'Ereg Proportional Term', kermit_tank_data)
# ereg_kp = greg_ereg_data.convert_to_channel('Kp', 'Ereg Kp', kermit_tank_data)


## Propellant densities
pdms_frac = 0.03
rho_fuel_const = np.ones_like(base_time) * (786 * (1 - pdms_frac) + 965 * pdms_frac)

# n2o_vapor_pressure = 28e5
# tank_p = 50e5
# nitrous = Fluid(FluidsList.NitrousOxide).with_state(Input.pressure(n2o_vapor_pressure), Input.quality(0))
# rho_ox_const = np.ones_like(base_time) * nitrous.density
rho_ox_const = np.ones_like(base_time) * 900

## Engine parameters
eta_cstar_default = 0.95
eta_cf_default = 0.90

# Injector
fuel_annulus_id = 5.569e-3
fuel_annulus_od = 6e-3
fuel_annulus_A = 0.25 * np.pi * (fuel_annulus_od**2 - fuel_annulus_id**2)
fuel_film_A = 42 * 0.25 * np.pi * (0.2e-3)**2
fuel_inj_A = fuel_annulus_A + fuel_film_A
fuel_inj_Cd = 0.75

fuel_inj_CdA = fuel_inj_Cd * fuel_inj_A
# fuel_inj_CdA = 25e-6

ox_inj_CdA = 1.4e-6


# Chamber
dc = 38.80e-3
dt = 13.542e-3
# de = 82.23e-3

Ac = 0.25 * np.pi * dc**2
At = 0.25 * np.pi * dt**2
# Ae = 0.25 * np.pi * de**2

cr = Ac/At
# eps = Ae/At

CEA = CEA_Obj(
    oxName = 'N2O',
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
    fac_CR=cr,
    make_debug_prints=False)

# Derived parameter calcs
rho_fuel_chosen = rho_fuel_const
rho_ox_chosen = rho_ox_const

fuel_inj_dP = Channel((fuel_inj_p.data - chamber_p.data), base_time, 'Fuel Inj dP')
fuel_system_dP = Channel((fuel_tank_p.data - fuel_inj_p.data), base_time, 'Fuel System dP')
fuel_mdot_flowmeter = Channel(rho_fuel_chosen * fuel_vdot.data, base_time, 'Fuel mdot (from flowmeter)')
fuel_mdot_dP = Channel(spi_mdot(fuel_inj_CdA, rho_fuel_chosen, fuel_inj_dP.data), base_time, 'Fuel mdot (from dP)')

ox_inj_dP = Channel((ox_inj_p.data - chamber_p.data), base_time, 'Ox Inj dP')
ox_system_dP = Channel((ox_tank_p.data - ox_inj_p.data), base_time, 'Ox System dP')
ox_mdot_dP = Channel(spi_mdot(ox_inj_CdA, rho_ox_chosen, ox_inj_dP.data), base_time, 'Ox mdot (from dP)')

engine_running_mask = np.where((chamber_p.data > 5)&(fuel_mdot_flowmeter.data > 0)&(ox_mdot_dP.data > 0))[0]
cold_flow = len(engine_running_mask) == 0

from scipy.optimize import root_scalar

def cstar_residual_eq_fuel_based(OF, pc, At, fuel_mdot, eta_cstar, cea_instance: CEA_Obj):
    cstar = cea_instance.get_Cstar(pc, OF)
    ox_mdot = OF * fuel_mdot
    total_mdot = fuel_mdot + ox_mdot
    return At * pc * 1e5 / (cstar * eta_cstar) - total_mdot

def cstar_residual_eq_ox_based(OF, pc, At, ox_mdot, eta_cstar, cea_instance: CEA_Obj):
    cstar = cea_instance.get_Cstar(pc, OF)
    fuel_mdot = ox_mdot / OF
    total_mdot = fuel_mdot + ox_mdot
    return At * pc * 1e5 / (cstar * eta_cstar) - total_mdot

def cea_ox_mdot_solver(base_time, mask, pc, fuel_mdot, At, eta_cstar, cea_instance: CEA_Obj):
    ox_mdot = np.zeros_like(base_time)
    for i in mask:
        try:           
            OF = root_scalar(cstar_residual_eq_fuel_based, args=(pc[i], At, fuel_mdot[i], eta_cstar, cea_instance), bracket=[0, 6], method='brentq', xtol=1e-4).root
            ox_mdot[i] = fuel_mdot[i] * OF
        except Exception as e:
            print(f"ox mdot: Warning: CEA OF solver did not converge at index {i} with error {e}. Ending iterative solver.")
    return ox_mdot

def cea_fuel_mdot_solver(base_time, mask, pc, ox_mdot, At, eta_cstar, cea_instance: CEA_Obj):
    fuel_mdot = np.zeros_like(base_time)
    for i in mask:
        try:
            OF = root_scalar(cstar_residual_eq_ox_based, args=(pc[i], At, ox_mdot[i], eta_cstar, cea_instance), bracket=[0, 6], method='brentq', xtol=1e-4).root
            fuel_mdot[i] = ox_mdot[i] / OF
        except ValueError:
            print(f"fuel solver Warning: CEA OF solver did not converge at index {i}. Ending iterative solver.")
    return fuel_mdot

def cea_theoretical_cstar(base_time, mask, cea_instance: CEA_Obj, pc, OF):
    cstar = np.zeros_like(base_time)
    for i in mask:
        if pc[i] > 0 and OF[i] > 0:
            cstar[i] = cea_instance.get_Cstar(pc[i]/1e5, OF[i])
    return cstar
    
if not cold_flow:
    ox_mdot_flowmeter = cea_ox_mdot_solver(base_time, engine_running_mask, chamber_p.data, fuel_mdot_flowmeter.data, At, eta_cstar_default, CEA)
    ox_mdot_cea_flowmeter = Channel(ox_mdot_flowmeter, base_time, 'Ox mdot (from ox CEA + fuel flowmeter)')
    ox_mdot_dp = cea_ox_mdot_solver(base_time, engine_running_mask, chamber_p.data, fuel_mdot_dP.data, At, eta_cstar_default, CEA)
    ox_mdot_cea_dp = Channel(ox_mdot_dp, base_time, 'Ox mdot (from ox CEA + fuel dP)')

    fuel_mdot_dp = cea_fuel_mdot_solver(base_time, engine_running_mask, chamber_p.data, ox_mdot_dP.data, At, eta_cstar_default, CEA)
    fuel_mdot_cea_dp = Channel(fuel_mdot_dp, base_time, 'Fuel mdot (from ox dP + fuel CEA)')

    cstar_flowmeter_data = np.zeros_like(base_time)
    cstar_flowmeter_data[engine_running_mask] = chamber_p.data[engine_running_mask] * 1e5 * At / (fuel_mdot_flowmeter.data[engine_running_mask] + ox_mdot_cea_flowmeter.data[engine_running_mask])

    cstar_dp_data = np.zeros_like(base_time)
    cstar_dp_data[engine_running_mask] = chamber_p.data[engine_running_mask] * 1e5 * At / (fuel_mdot_dP.data[engine_running_mask] + ox_mdot_cea_dp.data[engine_running_mask])

    cstar_flowmeter = Channel(cstar_flowmeter_data, base_time, 'C* (from ox dP + fuel flowmeter)')
    cstar_dp = Channel(cstar_dp_data, base_time, 'C* (from ox dP + fuel dP)')

    OF_cea_fuel_flowmeter_arr = np.divide(ox_mdot_cea_flowmeter.data, fuel_mdot_flowmeter.data, 
                                          out=np.zeros_like(base_time), 
                                          where=(ox_mdot_cea_flowmeter.data > 0) & (fuel_mdot_flowmeter.data > 0))
    OF_cea_fuel_flowmeter = Channel(OF_cea_fuel_flowmeter_arr, base_time, 'OF (from ox CEA + fuel flowmeter)')

    OF_cea_fuel_dp_arr = np.divide(ox_mdot_cea_dp.data, fuel_mdot_dP.data,
                                   out=np.zeros_like(base_time),
                                   where=(ox_mdot_cea_dp.data > 0) & (fuel_mdot_dP.data > 0))
    OF_cea_fuel_dp = Channel(OF_cea_fuel_dp_arr, base_time, 'OF (from ox CEA + fuel dP)')

    OF_cea_ox_dp_arr = np.divide(ox_mdot_dP.data, fuel_mdot_cea_dp.data,
                                 out=np.zeros_like(base_time),
                                 where=(ox_mdot_dP.data > 0) & (fuel_mdot_cea_dp.data > 0))
    OF_cea_ox_dp = Channel(OF_cea_ox_dp_arr, base_time, 'OF (from fuel CEA + ox dP)')

    OF_dp_arr = np.divide(ox_mdot_dP.data, fuel_mdot_dP.data,
                          out=np.zeros_like(base_time),
                          where=(ox_mdot_dP.data > 0) & (fuel_mdot_dP.data > 0))
    OF_dp_both = Channel(OF_dp_arr, base_time, 'OF (from ox dP + fuel dP)')

    OF_dp_flowmeter_arr = np.divide(ox_mdot_dP.data, fuel_mdot_flowmeter.data,
                                    out=np.zeros_like(base_time),
                                    where=(ox_mdot_dP.data > 0) & (fuel_mdot_flowmeter.data > 0))
    OF_dp_flowmeter = Channel(OF_dp_flowmeter_arr, base_time, 'OF (from ox dP + fuel flowmeter)')

    cstar_theoretical_flowmeter = Channel(cea_theoretical_cstar(base_time, engine_running_mask, CEA, chamber_p.data, OF_cea_fuel_flowmeter_arr), base_time, 'C* theoretical (from ox CEA + fuel flowmeter)')
    cstar_theoretical_dp = Channel(cea_theoretical_cstar(base_time, engine_running_mask, CEA, chamber_p.data, OF_cea_fuel_dp_arr), base_time, 'C* theoretical (from ox CEA + fuel dP)')

    cstar_eff_flowmeter_arr = np.divide(cstar_flowmeter.data, cstar_theoretical_flowmeter.data,
                                        out=np.zeros_like(base_time),
                                        where=cstar_theoretical_flowmeter.data > 0)
    cstar_eff_flowmeter = Channel(cstar_eff_flowmeter_arr, base_time, 'C* efficiency (from ox CEA + fuel flowmeter)')

    cstar_eff_dp_arr = np.divide(cstar_dp.data, cstar_theoretical_dp.data,
                                 out=np.zeros_like(base_time),
                                 where=cstar_theoretical_dp.data > 0)
    cstar_eff_dp = Channel(cstar_eff_dp_arr, base_time, 'C* efficiency (from ox CEA + fuel dP)')

## Plots
# Pressure plot
pressure_plot = Plot(title='Pressures', xlabel='Time (s)', ylabel_primary='Pressure (bar(a))')
pressure_plot.add_channel(n2_tank_p)
pressure_plot.add_channel(ox_tank_p)
pressure_plot.add_channel(fuel_tank_p)
pressure_plot.add_channel(fuel_inj_p)
pressure_plot.add_channel(ox_inj_p)
pressure_plot.add_channel(chamber_p)
# pressure_plot.add_channel(ereg_fuel_tank_p)

# Pressure drops plot
pressure_drops_plot = Plot(title='Pressure Drops', xlabel='Time (s)', ylabel_primary='Pressure Drop (bar)')
pressure_drops_plot.add_channel(fuel_inj_dP)
pressure_drops_plot.add_channel(fuel_system_dP)
pressure_drops_plot.add_channel(ox_inj_dP)
pressure_drops_plot.add_channel(ox_system_dP)

# Mass flow rates plot
mdot_plot = Plot(title='Mass Flow Rates', xlabel='Time (s)', ylabel_primary='Mass Flow Rate (kg/s)')
mdot_plot.add_channel(fuel_mdot_flowmeter)
mdot_plot.add_channel(fuel_mdot_dP)
if not cold_flow:
    mdot_plot.add_channel(ox_mdot_dP)
    mdot_plot.add_channel(ox_mdot_cea_flowmeter)
    mdot_plot.add_channel(ox_mdot_cea_dp)
    mdot_plot.add_channel(fuel_mdot_cea_dp)

# CdA plot
cda_plot = Plot(title='CdA Values', xlabel='Time (s)', ylabel_primary='CdA (m²)')
# Calculate fuel CdA from flowmeter and dP
fuel_cda_flowmeter = Channel(spi_CdA(fuel_mdot_flowmeter.data, fuel_inj_dP.data, rho_fuel_chosen), base_time, 'Fuel CdA (from flowmeter)')
# fuel_cda_dp = Channel(spi_CdA(fuel_mdot_dP.data, fuel_inj_dP.data, rho_fuel_chosen), base_time, 'Fuel CdA (from dP)')
cda_plot.add_channel(fuel_cda_flowmeter)
# cda_plot.add_channel(fuel_cda_dp)

if not cold_flow:
    # Calculate ox CdA
    ox_cda_dp = Channel(spi_CdA(ox_mdot_dP.data, ox_inj_dP.data, rho_ox_chosen), base_time, 'Ox CdA (from dP)')
    ox_cda_cea_flowmeter = Channel(spi_CdA(ox_mdot_cea_flowmeter.data, ox_inj_dP.data, rho_ox_chosen), base_time, 'Ox CdA (from CEA + flowmeter)')
    ox_cda_cea_dp = Channel(spi_CdA(ox_mdot_cea_dp.data, ox_inj_dP.data, rho_ox_chosen), base_time, 'Ox CdA (from CEA + dP)')
    # cda_plot.add_channel(ox_cda_dp)
    cda_plot.add_channel(ox_cda_cea_flowmeter)
    cda_plot.add_channel(ox_cda_cea_dp)

# Valve angles plot
valve_angles_plot = Plot(title='Valve Angles', xlabel='Time (s)', ylabel_primary='Angle (degrees)')
valve_angles_plot.add_channel(ox_valve_angle)
valve_angles_plot.add_channel(fuel_valve_angle)
# valve_angles_plot.add_channel(ereg_angle)

# # Create figures for basic plots
pressure_figure = Figure(pressure_plot, figsize=(15, 8))
pressure_drops_figure = Figure(pressure_drops_plot, figsize=(15, 8))
mdot_figure = Figure(mdot_plot, figsize=(15, 8))
cda_figure = Figure(cda_plot, figsize=(15, 8))
valve_angles_figure = Figure(valve_angles_plot, figsize=(15, 8))

# pressure_figure.show()
# pressure_drops_figure.show()
# mdot_figure.show()
cda_figure.show()
# valve_angles_figure.show()

if not cold_flow:
    # OF ratios plot
    of_plot = Plot(title='Oxidizer-to-Fuel Ratios', xlabel='Time (s)', ylabel_primary='OF Ratio')
    of_plot.add_channel(OF_cea_fuel_flowmeter)
    of_plot.add_channel(OF_cea_fuel_dp)
    of_plot.add_channel(OF_cea_ox_dp)
    of_plot.add_channel(OF_dp_both)
    of_plot.add_channel(OF_dp_flowmeter)
    
    # Theoretical C* plot
    cstar_theoretical_plot = Plot(title='Theoretical C* Values', xlabel='Time (s)', ylabel_primary='C* (m/s)')
    cstar_theoretical_plot.add_channel(cstar_theoretical_flowmeter)
    cstar_theoretical_plot.add_channel(cstar_theoretical_dp)
    
    # C* efficiency plot
    cstar_eff_plot = Plot(title='C* Efficiencies', xlabel='Time (s)', ylabel_primary='C* Efficiency')
    cstar_eff_plot.add_channel(cstar_eff_flowmeter)
    cstar_eff_plot.add_channel(cstar_eff_dp)
    
    # Create figures for hot fire plots
    of_figure = Figure(of_plot, figsize=(15, 8))
    cstar_theoretical_figure = Figure(cstar_theoretical_plot, figsize=(15, 8))
    cstar_eff_figure = Figure(cstar_eff_plot, figsize=(15, 8))
    
    # of_figure.show()
    # cstar_theoretical_figure.show()
    # cstar_eff_figure.show()

plt.show()