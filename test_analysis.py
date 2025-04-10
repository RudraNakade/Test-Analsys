import pandas
import matplotlib.pyplot as plt
import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj

# Engine params
pintle_OD = 5.569
fuel_core_A = 0.25e-6 * np.pi * (6**2 - pintle_OD**2)
fuel_film_A = 0.25e-6 * np.pi * (0.2 ** 2) * 42
fuel_total_A = fuel_core_A + fuel_film_A
fuel_rho = 786 * 0.97 + 965 * 0.03
fuel_film_frac = fuel_film_A / fuel_core_A
fuel_Cd = 0.75
fuel_core_CdA = fuel_Cd * fuel_core_A
fuel_total_CdA = fuel_Cd * fuel_total_A

ox_A = 0.25e-6 * np.pi * (0.7 ** 2) * 24
ox_rho = 900
ox_Cd = 0.2
ox_CdA = ox_Cd * ox_A
ox_CdA = 1.7e-6

throat_D = 13.542e-3
throat_A = (np.pi * throat_D**2) / 4
cstar_eff = 1
CR = 8.7248

###########################################

CEA = CEA_Obj(
    oxName = 'N2O',
    fuelName = 'Isopropanol',
    isp_units='sec',
    cstar_units = 'm/s',
    pressure_units='Bar',
    temperature_units='K',
    sonic_velocity_units='m/s',
    enthalpy_units='J/g',
    density_units='kg/m^3',
    specific_heat_units='J/kg-K',
    viscosity_units='centipoise', # stored value in pa-s
    thermal_cond_units='W/cm-degC', # stored value in W/m-K
    fac_CR=CR,
    make_debug_prints=False)

# test_folder = "2025\\22_02_25_HOPPER_ENGINE\\cold_flow_1\\"
# test_folder = "2025\\22_02_25_HOPPER_ENGINE\\hotfire_1\\"
# test_folder = "2025\\23_02_25_HOPPER_ENGINE\\cold_flow_1\\"
# test_folder = "2025\\23_02_25_HOPPER_ENGINE\\hotfire_1\\"
test_folder = "2025\\23_02_25_HOPPER_ENGINE\\hotfire_2\\"
# test_folder = "2025\\23_02_25_HOPPER_ENGINE\\hotfire_3\\"

kermit_cf = pandas.read_csv(test_folder + "kermit_telem_trim.csv")
stark_cf  = pandas.read_csv(test_folder + "stark_telem_trim.csv")

kermit_cf = pandas.DataFrame(kermit_cf)
stark_cf = pandas.DataFrame(stark_cf)

stark_cfTime = np.array(stark_cf['BackendTime'])/1e3
kermit_cfTime = np.array(kermit_cf['BackendTime'])/1e3

kermit_cfTime = (kermit_cfTime - stark_cfTime[0]) - 5
stark_cfTime = (stark_cfTime - stark_cfTime[0]) - 5

endtime = min(stark_cfTime[-1], kermit_cfTime[-1])

fuel_tank_P = stark_cf['ch4sens']
ox_tank_P = np.array(kermit_cf['ch0sens'])
ox_tank_mass = kermit_cf['ch2sens']
n2_P = np.array(kermit_cf['ch1sens'])
thrust_loadcell = -1 * np.array(kermit_cf['ch3sens'])
chamber_P = np.array(stark_cf['ch3sens'])
fuel_inj_P = np.array(stark_cf['ch1sens'])
regen_inlet_P = np.array(stark_cf['ch2sens'])
ox_inj_P = np.array(stark_cf['ch0sens'])
fuel_flowmeter = np.array(stark_cf['flowmeter'])
ox_main_ang = np.array(stark_cf['oxAngle'])
fuel_main_ang = np.array(stark_cf['fuelAngle'])
fuel_inj_T = np.array(kermit_cf['temp1'])

ox_tank_P = np.interp(stark_cfTime, kermit_cfTime, ox_tank_P)

loadcell_zero_n = len(kermit_cfTime[kermit_cfTime < -1])
loadcell_zero = np.mean(thrust_loadcell[loadcell_zero_n])
thrust = thrust_loadcell - loadcell_zero
thrust = np.interp(stark_cfTime, kermit_cfTime, thrust)
cf = np.divide(thrust, chamber_P*1e5*throat_A)

fuel_inj_dP = fuel_inj_P - chamber_P
ox_inj_dP = ox_inj_P - chamber_P
regen_dP = regen_inlet_P - fuel_inj_P

dP_zero_n = len(stark_cfTime[stark_cfTime < -1])
fuel_inj_dP_zero = np.mean(fuel_inj_dP[dP_zero_n])
ox_inj_dP_zero = np.mean(ox_inj_dP[dP_zero_n])
regen_dP_zero = np.mean(regen_dP[dP_zero_n])

fuel_inj_dP = fuel_inj_dP - fuel_inj_dP_zero
ox_inj_dP = ox_inj_dP - ox_inj_dP_zero
regen_dP = regen_dP - regen_dP_zero
fuel_system_dP = fuel_tank_P - fuel_inj_P
ox_system_dP = ox_tank_P - ox_inj_P

fuel_inj_dP[fuel_inj_dP < 0.1] = 0
ox_inj_dP[ox_inj_dP < 0.1] = 0
regen_dP[regen_dP < 0] = 0

fuel_tot_mdot_dP = fuel_total_CdA * np.sqrt(2e5 * fuel_inj_dP * fuel_rho)
fuel_tot_mdot_flowmeter = fuel_flowmeter * fuel_rho / 1000

fuel_core_mdot_dP = fuel_tot_mdot_dP / (1 + fuel_film_frac)
fuel_core_mdot_flowmeter = fuel_tot_mdot_flowmeter / (1 + fuel_film_frac)

fuel_inj_total_CdA_exp = np.divide(fuel_tot_mdot_flowmeter, np.sqrt(2e5 * fuel_inj_dP * fuel_rho),
              out=np.zeros_like(fuel_tot_mdot_flowmeter),
              where=fuel_inj_dP != 0.0)
fuel_inj_Cd_exp = fuel_inj_total_CdA_exp / fuel_total_A

fuel_inj_core_CdA_exp = np.divide(fuel_core_mdot_flowmeter, np.sqrt(2e5 * fuel_inj_dP * fuel_rho),
              out=np.zeros_like(fuel_core_mdot_flowmeter),
              where=fuel_inj_dP != 0.0)
fuel_inj_Cd_exp = fuel_inj_core_CdA_exp / fuel_core_A

fuel_sys_CdA = np.divide(fuel_tot_mdot_flowmeter, np.sqrt(2e5 * fuel_system_dP * fuel_rho),
              out=np.zeros_like(fuel_tot_mdot_flowmeter),
              where=fuel_system_dP != 0.0)

ox_mdot_inj_dP = ox_CdA * np.sqrt(2e5 * ox_inj_dP * ox_rho)

mdot_tot_cea = np.zeros(len(chamber_P))
ox_mdot_cea = np.zeros(len(chamber_P))
core_OF_cea = np.zeros(len(chamber_P))

for i in range(len(chamber_P)):
    OF = 1.5
    diff = 1
    while diff > 1e-3:
        OFold = OF
        if (fuel_core_mdot_flowmeter[i] < 1e-3) or (chamber_P[i] < 3) or (ox_mdot_cea[i] < 0):
            OF = 0
            ox_mdot_cea[i] = 0
            break
        mdot_tot_cea[i] = throat_A * chamber_P[i] * 1e5 / (CEA.get_Cstar(chamber_P[i], OF) * cstar_eff)
        ox_mdot_cea[i] = mdot_tot_cea[i] - fuel_core_mdot_dP[i]
        if fuel_core_mdot_dP[i] > 0:
            OF = ox_mdot_cea[i] / fuel_core_mdot_dP[i]
        else:
            OF = 0
        diff = abs((OFold - OF) / OF) if OF != 0 else 1
    core_OF_cea[i] = OF

core_OF_ox_dP = np.divide(ox_mdot_inj_dP, fuel_core_mdot_dP,
                          out=np.zeros_like(ox_mdot_inj_dP),
                          where=fuel_core_mdot_dP > 0)
core_OF_ox_dP[stark_cfTime < 0] = 0

ox_inj_CdA_cea = np.divide(ox_mdot_cea, np.sqrt(2e5 * ox_inj_dP * ox_rho),
                      out=np.zeros_like(ox_mdot_cea),
                      where=ox_inj_dP != 0.0)

ox_inj_Cd_cea = ox_inj_CdA_cea / ox_A

ox_sys_CdA = np.divide(ox_mdot_cea, np.sqrt(2e5 * ox_system_dP * ox_rho),
              out=np.zeros_like(ox_mdot_cea),
              where=ox_inj_dP != 0.0)

# isp = np.divide(thrust, mdot_tot_cea * 9.81,
#                 out=np.zeros_like(thrust),
#                 where=mdot_tot_cea > 0)
# plt.figure()
# plt.plot(stark_cfTime, isp, label='Isp', color="tab:blue")


## Plotting
# Create a single figure with a 2x3 grid of subplots
analsysis_fig, axs = plt.subplots(2, 3, figsize=(24, 13.5), sharex=True, dpi = 100)
analsysis_fig.suptitle(f"Test Analysis, \u03B7_c* = {cstar_eff:.2f}")

# Pressure plot (top left)
ax1 = axs[0, 0]
ax2 = ax1.twinx()

ln1 = ax1.plot(stark_cfTime, fuel_tank_P, label='Fuel Tank', color="tab:red", linestyle='--')
ln2 = ax1.plot(stark_cfTime, ox_tank_P, label='Ox Tank', color="tab:blue", linestyle='--')
ln3 = ax1.plot(stark_cfTime, ox_inj_P, label='Ox Inj', color="tab:blue")
ln4 = ax1.plot(stark_cfTime, fuel_inj_P, label='Fuel Inj', color="tab:red")
ln5 = ax1.plot(stark_cfTime, chamber_P, label='Chamber', color="tab:orange")
ln6 = ax1.plot(stark_cfTime, regen_inlet_P, label='Regen Inlet', color="tab:gray", linestyle='--')
ln7 = ax2.plot(kermit_cfTime, n2_P, label='N2', color="tab:purple")
ax2.set_ylabel("N2 Pressure (bar)")
ax2.set_ylim(bottom=0, top=np.max(n2_P)*1.02)

lns = ln1 + ln2 + ln3 + ln4 + ln5 + ln6 + ln7
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='best')
ax1.set_ylabel("Pressure (bar)")
ax1.set_ylim(bottom=0)
ax1.set_title("Pressures")
ax1.grid()
ax1.grid(which="minor", alpha=0.5)
ax1.minorticks_on()

# Thrust plot (top middle)
ax3 = axs[0, 1]
ax3.plot(stark_cfTime, thrust, label='Thrust', color="tab:blue")
ax3.grid()
ax3.grid(which="minor", alpha=0.5)
ax3.minorticks_on()
ax3.legend()
ax3.set_ylabel("Thrust (N)")
ax3.set_title("Thrust")

# Flow measurements (top right)
ax4 = axs[0, 2]
ax4.plot(stark_cfTime, fuel_tot_mdot_dP, label='Total Fuel (dP)', color="tab:purple")
ax4.plot(stark_cfTime, fuel_core_mdot_dP, label='Core Fuel (dP)', color="tab:red")
ax4.plot(stark_cfTime, fuel_tot_mdot_flowmeter, label='Total Fuel (Flowmeter)', color="tab:purple", linestyle='--')
ax4.plot(stark_cfTime, fuel_core_mdot_flowmeter, label='Core Fuel (Flowmeter)', color="tab:red", linestyle='--')
ax4.plot(stark_cfTime, ox_mdot_inj_dP, label='Ox (dP)', color="tab:blue")
ax4.plot(stark_cfTime, ox_mdot_cea, label='Ox (CEA)', color="tab:blue", linestyle='--')
ax4.grid()
ax4.grid(which="minor", alpha=0.5)
ax4.minorticks_on()
ax4.legend(loc='best')
ax4.set_ylabel("Mass Flow Rate (kg/s)")
ax4.set_title("Mass Flow")

# OF Ratio plot (bottom left)
ax5 = axs[1, 0]
ax5.plot(stark_cfTime, core_OF_ox_dP, label='Ox dP Based', color="tab:blue")
ax5.plot(stark_cfTime, core_OF_cea, label='CEA + Fuel dP Based', color="tab:red")
ax5.legend(loc='best')
ax5.set_ylabel("OF Ratio")
ax5.grid()
ax5.grid(which="minor", alpha=0.5)
ax5.minorticks_on()
ax5.set_title("Core OF Ratio")
ax5.set_xlabel("Time (s)")
ax5.set_ylim(-0.1, 3)

# Pressure Drop plot (bottom middle)
ax6 = axs[1, 1]
ax6.plot(stark_cfTime, fuel_inj_dP, label='Fuel Inj ', color="tab:red")
ax6.plot(stark_cfTime, ox_inj_dP, label='Ox Inj', color="tab:blue")
ax6.plot(stark_cfTime, regen_dP, label='Regen Channels', color="tab:green")
ax6.plot(stark_cfTime, fuel_system_dP, label='Fuel System', color="tab:red", linestyle='--')
ax6.plot(stark_cfTime, ox_system_dP, label='Ox System', color="tab:blue", linestyle='--')
ax6.grid()
ax6.grid(which="minor", alpha=0.5)
ax6.minorticks_on()
ax6.legend(loc='best')
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Pressure Drop (bar)")
ax6.set_title("Pressure Drop")

# Add valve angles plot (bottom right)
ax7 = axs[1, 2]
ax7.plot(stark_cfTime, ox_main_ang, label='Ox Main', color="tab:blue")
ax7.plot(stark_cfTime, fuel_main_ang, label='Fuel Main', color="tab:red")
ax7.grid()
ax7.grid(which="minor", alpha=0.5)
ax7.minorticks_on()
ax7.legend(loc='best')
ax7.set_xlabel("Time (s)")
ax7.set_ylabel("Valve Angle")
ax7.set_title("Valve Angles")

# Set x-limits for all subplots
for row in axs:
    for ax in row:
        ax.set_xlim(-1, endtime - 3.5)

# Adjust layout to prevent overlap
analsysis_fig.tight_layout()
analsysis_fig.savefig(test_folder + "test_analysis.png")

plt.show()

plt.figure(figsize=(25.6, 14.4), dpi=100)
plt.plot(stark_cfTime, fuel_inj_Cd_exp, label='Fuel Cd', color="tab:red")
plt.plot(stark_cfTime, ox_inj_Cd_cea, label='Ox Cd', color="tab:blue")
plt.grid(True)
plt.minorticks_on()
plt.grid(which="minor", alpha=0.5)
plt.xlim(-0.5, endtime - 4)
plt.ylim(0, 1)
plt.xlabel("Time (s)")
plt.ylabel("Discharge Coefficient (Cd)")
plt.title(f"Injector Cd's, \u03B7_c* = {cstar_eff:.2f}")
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig(test_folder + "Cd_values.png")

plt.figure(figsize=(25.6, 14.4), dpi=100)
plt.plot(stark_cfTime, fuel_inj_total_CdA_exp, label='Fuel Injector Total CdA', color="tab:red")
plt.plot(stark_cfTime, fuel_inj_core_CdA_exp, label='Fuel Injector Core CdA', color="tab:purple")
plt.plot(stark_cfTime, ox_inj_CdA_cea, label='Ox Injector CdA', color="tab:blue")
plt.plot(stark_cfTime, fuel_sys_CdA, label='Fuel System CdA', color="tab:red", linestyle='--')
plt.plot(stark_cfTime, ox_sys_CdA, label='Ox System CdA', color="tab:blue", linestyle='--')
plt.grid(True)
plt.minorticks_on()
plt.grid(which="minor", alpha=0.5)
plt.xlim(-0.5, endtime - 4)
plt.xlabel("Time (s)")
plt.ylabel("Flow Coefficient (m^2)")
plt.title(f"System Flow Coefficients, \u03B7_c* = {cstar_eff:.2f}")
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig(test_folder + "CdA_values.png")

plt.figure(figsize=(25.6, 14.4), dpi=100)
plt.plot(stark_cfTime, cf, color="tab:blue")
plt.grid(True)
plt.minorticks_on()
plt.grid(which="minor", alpha=0.5)
plt.xlim(-0.5, endtime - 4)
plt.xlabel("Time (s)")
plt.ylabel("Thrust Coefficient (Cf)")
plt.title("Thrust Coefficient")
plt.tight_layout()

plt.savefig(test_folder + "Cf.png")