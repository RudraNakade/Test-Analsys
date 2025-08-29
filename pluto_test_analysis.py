from test_analysis import *
from rocketcea.cea_obj_w_units import CEA_Obj
from os import system
system('cls')

# Import data and make channels
base_data_folder = "C:\\Users\\Rudra\\Desktop\\Propulsion\\Test Data\\2025\\29_08_25_PLUTO_FLIGHT_QUAL"
test_folder = "sample_data"
data_folder = base_data_folder + "\\" + test_folder + "\\raw_data"

## Board sensor outputs:
# Kermit: ch0sens, ch1sens, ch2sens, ch3sens, temp0, temp1
# Stark: servo_voltage, ch0sens, ch1sens, ch2sens, ch3sens, ch4sens, ch5sens, flowmeter, oxAngle, fuelAngle
# Greg: Feedforward, FuelTankP, regAngle, Proportional_Term, Kp

# File names
kermit_tank_data_file = ""
kermit_fill_data_file = ""
kermit_hose_data_file = ""
greg_ereg_data_file   = ""
stark_data_file       = ""

# Datasets
kermit_tank_data = Dataset(data_folder + kermit_tank_data_file) # N2, Ox, Fuel tank PTs
kermit_fill_data = Dataset(data_folder + kermit_fill_data_file) # Filling panel PTs, Tank weight LC
kermit_hose_data = Dataset(data_folder + kermit_hose_data_file) # Hoses PTs
greg_ereg_data   = Dataset(data_folder + greg_ereg_data_file)   # Ereg PTs
stark_data       = Dataset(data_folder + stark_data_file)       # Fuel + Ox Inj, Chamber PTs, Fuel Inj TC

# Reference time
base_time = kermit_tank_data.backend_time

# Make channels
n2_tank_p = kermit_tank_data.convert_to_channel('', 'N2 Tank Pressure')
ox_tank_p = kermit_tank_data.convert_to_channel('', 'Ox Tank Pressure')
fuel_tank_p = kermit_tank_data.convert_to_channel('', 'Fuel Tank Pressure')

fuel_ereg_tank_p = greg_ereg_data.convert_to_channel('', 'Fuel Tank Pressure (Ereg)', sync_dataset=kermit_tank_data)

fuel_inj_p = stark_data.convert_to_channel('', 'Fuel Injector Pressure', sync_dataset=kermit_tank_data)
fuel_inj_t = stark_data.convert_to_channel('', 'Fuel Injector Temperature', sync_dataset=kermit_tank_data)
ox_inj_p = stark_data.convert_to_channel('', 'Ox Injector Pressure', sync_dataset=kermit_tank_data)
chamber_p = stark_data.convert_to_channel('', 'Chamber Pressure', sync_dataset=kermit_tank_data)

ox_valve_angle = stark_data.convert_to_channel('', 'Ox Valve Angle', sync_dataset=kermit_tank_data)
fuel_valve_angle = stark_data.convert_to_channel('', 'Fuel Valve Angle', sync_dataset=kermit_tank_data)

# Injector params
fuel_annulus_id = 17.29e-3
fuel_annulus_od = 18.18e-3
fuel_annulus_A = 0.25 * np.pi * (fuel_annulus_od**2 - fuel_annulus_id**2)
fuel_film_A = 56 * 0.25 * np.pi * (0.4e-3)**2
fuel_inj_A = fuel_annulus_A + fuel_film_A

ox_inj_CdA = 79e-6

# Engine params
eta_cstar_default = 0.95
eta_cf_default = 0.90

dc = 92.00e-3
dt = 42.41e-3
de = 82.23e-3

Ac = np.pi * (dc/2)**2
At = np.pi * (dt/2)**2
Ae = np.pi * (de/2)**2

cr = Ac/At
eps = Ae/At

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

plt.show()