import pandas
import matplotlib.pyplot as plt
import numpy as np

# Load the data files
kermit_cf = pandas.read_csv("2025\\23_02_25_HOPPER_ENGINE\\raw_data\\20250223_142750.968298Z_sts_sen0_telem.csv")
stark_cf  = pandas.read_csv("2025\\23_02_25_HOPPER_ENGINE\\raw_data\\20250223_142750.966056Z_sts_stark_telem.csv")

# Adjust timing to start from the same point
min_time = stark_cf['BackendTime'][0]
kermit_cf = pandas.DataFrame(kermit_cf[kermit_cf['BackendTime'] > min_time])
stark_cf = pandas.DataFrame(stark_cf[stark_cf['BackendTime'] > min_time])

# Convert absolute times to relative seconds from start
stark_cfTime = stark_cf['BackendTime']
kermit_cfTime = kermit_cf['BackendTime']

kermit_cfTime = (kermit_cfTime - stark_cfTime[stark_cfTime.index[0]]) / 1e3
stark_cfTime = (stark_cfTime - stark_cfTime[stark_cfTime.index[0]]) / 1e3

# Extract sensor data
fuel_tank_P = stark_cf['ch4sens']
ox_tank_P = np.array(kermit_cf['ch0sens'])
thrust = np.array(kermit_cf['ch3sens'])
chamber_P = np.array(stark_cf['ch3sens'])
fuel_inj_P = np.array(stark_cf['ch1sens'])
regen_inlet_P = np.array(stark_cf['ch2sens'])
ox_inj_P = np.array(stark_cf['ch0sens'])
fuel_flowmeter = np.array(stark_cf['flowmeter'])
n2_tank_P = np.array(kermit_cf['ch1sens'])
ox_main_ang = np.array(stark_cf['oxAngle'])
fuel_main_ang = np.array(stark_cf['fuelAngle'])
fuel_inj_T = np.array(kermit_cf['temp1'])

# Create plots to visualize the data
fig = plt.figure(figsize=(15, 10))

# Calculate the full time range to be used for all plots
min_time = min(stark_cfTime.min(), kermit_cfTime.min())
max_time = max(stark_cfTime.max(), kermit_cfTime.max())

# Plot 1: Valve Angles
ax1 = plt.subplot(3, 2, 1)
ax1.plot(stark_cfTime, fuel_main_ang, label='Fuel Valve Angle')
ax1.plot(stark_cfTime, ox_main_ang, label='Ox Valve Angle')
ax1.set_title('Valve Angles')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angle')
ax1.legend()
ax1.grid(True)
ax1.set_xlim(min_time, max_time)

# Plot 2: Combined Pressures (with dual y-axis)
ax2 = plt.subplot(3, 2, 2, sharex=ax1)
ax2_twin = ax2.twinx()  # Create a secondary y-axis

# Primary y-axis for most pressures
ln1 = ax2.plot(stark_cfTime, fuel_tank_P, label='Fuel Tank', color="tab:red", linestyle='--')
ln2 = ax2.plot(kermit_cfTime, ox_tank_P, label='Ox Tank', color="tab:blue", linestyle='--')
ln3 = ax2.plot(stark_cfTime, chamber_P, label='Chamber P', color="tab:orange")
ln4 = ax2.plot(stark_cfTime, fuel_inj_P, label='Fuel Inj P', color="tab:red")
ln5 = ax2.plot(stark_cfTime, ox_inj_P, label='Ox Inj P', color="tab:blue")
ln6 = ax2.plot(stark_cfTime, regen_inlet_P, label='Regen Inlet', color="tab:gray", linestyle='--')

# Secondary y-axis for N2 pressure
ln7 = ax2_twin.plot(kermit_cfTime, n2_tank_P, label='N2', color="tab:purple")
ax2_twin.set_ylabel("N2 Pressure")

# Combine all lines for a single legend
lns = ln1 + ln2 + ln3 + ln4 + ln5 + ln6 + ln7
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc='best')
ax2.set_title('All Pressures')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Pressure')
ax2.grid(True)

# Plot 3: Flowmeter
ax3 = plt.subplot(3, 2, 3, sharex=ax1)
ax3.plot(stark_cfTime, fuel_flowmeter, label='Fuel Flowmeter')
ax3.set_title('Fuel Flow')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Flow Rate')
ax3.legend()
ax3.grid(True)

# Plot 4: Thrust
ax4 = plt.subplot(3, 2, 4, sharex=ax1)
ax4.plot(kermit_cfTime, -thrust, label='Thrust')
ax4.set_title('Thrust')
ax4.set_xlabel('Time (s)')
ax4.grid(True)

# Plot 5: Temperature
ax5 = plt.subplot(3, 2, 5, sharex=ax1)
ax5.plot(kermit_cfTime, fuel_inj_T, label='Fuel Inj Temp')
ax5.set_title('Fuel Inj Temp')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Temperature (°C)')
ax5.grid(True)

# Plot 6: Pressure differentials
ax6 = plt.subplot(3, 2, 6, sharex=ax1)
# Calculate pressure differentials
fuel_inj_dP = fuel_inj_P - chamber_P
ox_inj_dP = ox_inj_P - chamber_P
ax6.plot(stark_cfTime, fuel_inj_dP, label='Fuel Inj ΔP')
ax6.plot(stark_cfTime, ox_inj_dP, label='Ox Inj ΔP')
ax6.set_title('Pressure Drops')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Pressure Drop (bar)')
ax6.legend()
ax6.grid(True)

# Apply the same x-axis limits to all plots
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_xlim(min_time, max_time)

plt.tight_layout()
plt.show()
