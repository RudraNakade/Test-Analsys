import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

data_path = "2025/20250704_ICLR/20250704-001-release.h5"
data = h5py.File(data_path, 'r')

class dataChannel:
    def __init__(self, channel, data):
        self.data = data['channels'][channel]['data'][:]
        self.time = data['channels'][channel]['time'][:]
        self.name = data['channels'][channel].attrs['name']
        self.units = data['channels'][channel].attrs['units']
        if self.units == 'bar(g)':
            self.data = self.data + 1
            self.units = 'bar(a)'

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.data)
        plt.xlabel('Time')
        plt.ylabel(self.units)
        plt.title(self.name)
        plt.grid(True)

# Available channels: ['DAU1002_demand', 'DAU1002_dmdot_filt_dt', 'DAU1002_mdot', 'DAU1002_mdot_filt', 'DAU1002_mdot_req', 'DAU1002_p1', 'DAU1002_p2', 'DAU1002_p_offset', 'DAU1002_tc', 'DAU1003_demand', 'DAU1003_mdot_req', 'DAU1018_armed', 'DAU1018_safearm_timer', 'DAU1018_seq_start', 'DAU1019_armed', 'DAU1020_armed', 'DAU1020_safearm_timer', 'DAU1020_seq_start', 'DP730', 'DP850', 'DT730', 'DT850', 'LC600', 'M730', 'M850', 'PT290', 'PT401', 'PT402', 'PT520', 'PT521', 'PT522', 'PT523', 'PT524', 'PT730', 'PT731', 'PT732', 'PT733', 'PT850', 'PT851', 'PT852', 'PT853', 'PT890', 'PTX101', 'PTX102', 'TC731', 'TC732', 'TC850', 'TC851', 'TC852', 'TC890', 'TCX101', 'V291', 'V520', 'V521', 'V522', 'V525', 'V526', 'V527', 'V531', 'V690', 'V691', 'V730', 'V731', 'V851', 'V857', 'V891', 'V893', 'XT731', 'XT852']

for channel in data['channels']:
    name = data['channels'][channel].attrs['name']
    print(f"{channel}: {name}")

plt.show()

data.close()