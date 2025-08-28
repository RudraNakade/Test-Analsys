import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.constants import g
from rocketcea.cea_obj_w_units import CEA_Obj
import time

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

class engine:
    def __init__(self, chamber_d, throat_d, exit_d):
        self.dc = chamber_d
        self.dt = throat_d
        self.de = exit_d

        self.chamber_area = 0.25 * np.pi * self.dc**2
        self.throat_area = 0.25 * np.pi * self.dt**2
        self.exit_area = 0.25 * np.pi * self.de**2

        self.cr = self.chamber_area / self.throat_area
        self.eps = self.exit_area / self.throat_area

        self.cea = CEA_Obj(
            oxName='N2O',
            fuelName='Isopropanol',
            isp_units='sec',
            cstar_units='m/s',
            pressure_units='Bar',
            temperature_units='K',
            sonic_velocity_units='m/s',
            enthalpy_units='J/kg',
            density_units='kg/m^3',
            specific_heat_units='J/kg-K',
            viscosity_units='centipoise',  # stored value in pa-s
            thermal_cond_units='W/cm-degC',  # stored value in W/m-K
            fac_CR=self.cr,
            make_debug_prints=False
        )

class test_data_analysis:
    def __init__(self, test_name, data_path, engine_obj: engine):
        self.test_name = test_name
        self.data_path = data_path
        self.data_file = h5py.File(data_path, 'r')
        channel_mapping = {
            'thrust': 'LC600',
            'chamber_p': 'PTX101',
            'fuel_inj_p': 'PTX102',
            'fuel_inj_t': 'TCX101',
            'fuel_inlet_p': 'PT732',
            'fuel_inlet_t': 'TC732',
            'fuel_mdot': 'M730',
            'fuel_density': 'DT730',
            'ox_inj_p': 'PT852',
            'ox_inj_t': 'TC852',
            'ox_mdot': 'M850',
            'ox_density': 'DT850'
        }
        
        self.channels = {}
        for name, channel_id in channel_mapping.items():
            self.channels[name] = channel(channel_id, self.data_file)

        self.time = self.channels['thrust'].time

        self.total_mdot = self.channels['ox_mdot'].data + self.channels['fuel_mdot'].data
        self.t_start = self.time[0]
        self.channels['thrust'].zero(self.t_start, self.t_start + 0.5)
        
        self.engine = engine_obj
        self.of = np.zeros_like(self.total_mdot)
        self.isp_exp = np.zeros_like(self.total_mdot)
        self.cstar_exp = np.zeros_like(self.total_mdot)

        mdot_tol = 0.01
        self.mask = (self.channels['ox_mdot'].data >= mdot_tol) & (self.channels['fuel_mdot'].data >= mdot_tol)

        self.isp_ideal = np.zeros_like(self.total_mdot)
        self.cstar_ideal = np.zeros_like(self.total_mdot)

        self.cf = np.zeros_like(self.total_mdot)

        self.cf_eff = np.zeros_like(self.total_mdot)
        self.cstar_eff = np.zeros_like(self.total_mdot)
        self.isp_eff = np.zeros_like(self.total_mdot)

        self.fuel_inj_dp = self.channels['fuel_inj_p'].data - self.channels['chamber_p'].data
        self.ox_inj_dp = self.channels['ox_inj_p'].data - self.channels['chamber_p'].data
        self.regen_dp = self.channels['fuel_inlet_p'].data - self.channels['fuel_inj_p'].data

        # Handle negative pressure differentials to avoid sqrt warnings
        self.ox_inj_CdA = np.where(self.ox_inj_dp > 0,
                                   self.channels['ox_mdot'].data / np.sqrt(2 * self.channels['ox_density'].data * self.ox_inj_dp * 1e5),
                                   0)
        self.fuel_inj_CdA = np.where(self.fuel_inj_dp > 0,
                                     self.channels['fuel_mdot'].data / np.sqrt(2 * self.channels['fuel_density'].data * self.fuel_inj_dp * 1e5),
                                     0)
        self.regen_CdA = np.where(self.regen_dp > 0,
                                  self.channels['fuel_mdot'].data / np.sqrt(2 * self.channels['fuel_density'].data * self.regen_dp * 1e5),
                                  0)
    
    def get_channel(self, name):
        """Get a specific channel by name"""
        return self.channels.get(name)
    
    def calc_exp_performance(self):
        self.of = np.where(self.mask, 
                  self.channels['ox_mdot'].data / self.channels['fuel_mdot'].data,
                  0)

        self.cstar_exp = np.where(self.mask, 
                     self.channels['chamber_p'].data * 1e5 * self.engine.throat_area / self.total_mdot,
                     0)

        self.isp_exp = np.where(self.mask, 
                       self.channels['thrust'].data / (self.total_mdot * g),
                       0)

        self.cf = np.where(self.mask,
                        self.channels['thrust'].data * 1e3 / (self.channels['chamber_p'].data * self.engine.throat_area * 1e5),
                        0)

    def calc_theoretical_performance(self):
        t = time.time()
        print("Calculating theoretical performance metrics...")
        valid_indices = np.where(self.mask)[0]
        for i in valid_indices:
            if i % 100 == 0 or i == valid_indices[-1] or i == valid_indices[0]:
            # if True:
                print(f"i: {i}/{valid_indices[-1]}, time: {self.time[i]:.2f}s", end='\r')
                pc_val = self.channels['chamber_p'].data[i]
                of_val = self.of[i]
                if pc_val > 0 and of_val > 0:
                    self.cstar_ideal[i] = self.engine.cea.get_Cstar(Pc=pc_val, MR=of_val)
                    self.isp_ideal[i] = self.engine.cea.estimate_Ambient_Isp(Pc=pc_val, MR=of_val, eps=self.engine.eps, Pamb=1.01325)[0]
                prev_i = i
            else:
                self.cstar_ideal[i] = self.cstar_ideal[prev_i]
                self.isp_ideal[i] = self.isp_ideal[prev_i]

        print(f"Calculated theoretical performance metrics in {time.time() - t:.4f} seconds")

    def calc_efficiencies(self):
        self.cstar_eff = np.where((self.mask) & (self.cstar_ideal != 0), 
                                  self.cstar_exp / self.cstar_ideal,
                                  0)
        self.cf_eff = np.where((self.mask) & (self.cstar_ideal != 0), 
                               self.cf / self.cstar_ideal,
                               0)
        self.isp_eff = np.where((self.mask) & (self.isp_ideal != 0), 
                                self.isp_exp / self.isp_ideal,
                                0)

    def filter_data(self, time_data, y_data, cutoff_freq=10, order=5):
        """
        Apply low-pass filter to data
        
        Args:
            time_data: Time array
            y_data: Data array to filter
            cutoff_freq: Cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered data array
        """
        dt = np.mean(np.diff(time_data))
        fs = 1 / dt
        return butter_lowpass_filter(y_data, cutoff_freq, fs, order)

    def plot_graph(self, x_data, y_data, x_label='Time (s)', y_label='', title='', label=None, 
                   color=None, linestyle='-', figsize=(10, 6), grid=True, show=True, 
                   apply_filter=False, cutoff_freq=10, order=5, ylim=None):
        """Plot a single graph with optional filtering and y-axis limits"""
        plt.figure(figsize=figsize)
        
        if apply_filter:
            y_filtered = self.filter_data(x_data, y_data, cutoff_freq, order)
            plt.plot(x_data, y_filtered, label=label, color=color, linestyle=linestyle)
        else:
            plt.plot(x_data, y_data, label=label, color=color, linestyle=linestyle)
            
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        if grid:
            plt.grid(True)
            plt.grid(which="minor", alpha=0.5)
            plt.minorticks_on()
        if label:
            plt.legend()
        if ylim:
            plt.ylim(ylim)
        if show:
            plt.show()

    def subplot(self, data_sets, rows, cols, figsize=(15, 10), suptitle='', show=True, 
                apply_filter=False, cutoff_freq=10, order=5, ylims=(None, None)):
        """
        Create subplots with multiple data sets and optional filtering
        data_sets: list of dictionaries with keys: 'x', 'y', 'xlabel', 'ylabel', 'title', 'label', 'color', 'linestyle', 'ylim'
        ylims: list of tuples for y-axis limits [(ymin, ymax), ...] or single tuple for all subplots
        """
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if suptitle:
            fig.suptitle(suptitle)
        
        # Ensure axes is always a list for consistent indexing
        if rows * cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, data_set in enumerate(data_sets):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Handle multiple lines on same subplot
            if isinstance(data_set['x'], list):
                num_lines = len(data_set['x'])
                
                # Ensure all list attributes have the same length as x data
                labels = data_set.get('label', [None] * num_lines)
                colors = data_set.get('color', [None] * num_lines)
                linestyles = data_set.get('linestyle', ['-'] * num_lines)
                
                # Extend lists if they're shorter than num_lines
                if len(labels) < num_lines:
                    labels.extend([None] * (num_lines - len(labels)))
                if len(colors) < num_lines:
                    colors.extend([None] * (num_lines - len(colors)))
                if len(linestyles) < num_lines:
                    linestyles.extend(['-'] * (num_lines - len(linestyles)))
                
                for j in range(num_lines):
                    x = data_set['x'][j]
                    y = data_set['y'][j]
                    
                    # Apply filtering if requested
                    if apply_filter or data_set.get('apply_filter', False):
                        filter_freq = data_set.get('cutoff_freq', cutoff_freq)
                        filter_order = data_set.get('order', order)
                        y = self.filter_data(x, y, filter_freq, filter_order)
                    
                    ax.plot(x, y, label=labels[j], color=colors[j], linestyle=linestyles[j])
            else:
                x = data_set['x']
                y = data_set['y']
                
                # Apply filtering if requested
                if apply_filter or data_set.get('apply_filter', False):
                    filter_freq = data_set.get('cutoff_freq', cutoff_freq)
                    filter_order = data_set.get('order', order)
                    y = self.filter_data(x, y, filter_freq, filter_order)
                
                ax.plot(x, y, 
                       label=data_set.get('label'), 
                       color=data_set.get('color'), 
                       linestyle=data_set.get('linestyle', '-'))
            
            ax.set_xlabel(data_set.get('xlabel', 'Time (s)'))
            ax.set_ylabel(data_set.get('ylabel', ''))
            ax.set_title(data_set.get('title', ''))
            ax.grid(True)
            
            # Set y-axis limits
            if data_set.get('ylim'):
                ax.set_ylim(data_set['ylim'])
            elif ylims:
                if isinstance(ylims, list) and len(ylims) > i:
                    ax.set_ylim(ylims[i])
                elif isinstance(ylims, tuple):
                    ax.set_ylim(ylims)
            
            if data_set.get('label'):
                ax.legend()
        
        if show:
            plt.tight_layout()
            plt.show()

    def plot_efficiencies(self, figsize=(12, 8), show=True, apply_filter=False, cutoff_freq=10, ylims=(0, 1.1)):
        """Plot engine performance efficiencies with optional filtering and y-axis limits"""
        time_data = self.time[self.mask]
        
        data_sets = [
            {
                'x': time_data,
                'y': self.cstar_eff[self.mask],
                'xlabel': 'Time (s)',
                'ylabel': 'C* Efficiency',
                'title': 'C* Efficiency',
                'label': 'C* Efficiency',
                'color': 'blue'
            },
            {
                'x': time_data,
                'y': self.cf_eff[self.mask],
                'xlabel': 'Time (s)',
                'ylabel': 'Cf Efficiency',
                'title': 'Cf Efficiency',
                'label': 'Cf Efficiency',
                'color': 'red'
            },
            {
                'x': time_data,
                'y': self.isp_eff[self.mask],
                'xlabel': 'Time (s)',
                'ylabel': 'Isp Efficiency',
                'title': 'Isp Efficiency',
                'label': 'Isp Efficiency',
                'color': 'green'
            }
        ]
        
        self.subplot(data_sets, 3, 1, figsize=figsize, 
                    suptitle=f'Engine Performance Efficiencies - {self.test_name}', 
                    show=show, apply_filter=apply_filter, cutoff_freq=cutoff_freq, ylims=ylims)

    def plot_performance_comparison(self, figsize=(15, 20), show=True, apply_filter=False, cutoff_freq=10, ylims=None):
        """Plot experimental vs theoretical performance comparison with thrust, pressures, mass flows, and CdA values"""
        time_data = self.time[self.mask]
        
        data_sets = [
            {
                'x': [time_data, time_data],
                'y': [self.cstar_exp[self.mask], self.cstar_ideal[self.mask]],
                'xlabel': 'Time (s)',
                'ylabel': 'C* (m/s)',
                'title': 'C* Comparison',
                'label': ['Experimental', 'Theoretical'],
                'color': ['blue', 'red'],
                'linestyle': ['-', '-']
            },
            {
                'x': [time_data, time_data],
                'y': [self.isp_exp[self.mask], self.isp_ideal[self.mask]],
                'xlabel': 'Time (s)',
                'ylabel': 'Isp (s)',
                'title': 'Isp Comparison',
                'label': ['Experimental', 'Theoretical'],
                'color': ['blue', 'red'],
                'linestyle': ['-', '-']
            },
            {
                'x': self.time,
                'y': self.channels['thrust'].data,
                'xlabel': 'Time (s)',
                'ylabel': 'Thrust (N)',
                'title': 'Thrust',
                'label': 'Thrust',
                'color': 'blue',
                'linestyle': '-'
            },
            {
                'x': [self.time, self.time, self.time, self.time],
                'y': [self.channels['chamber_p'].data, self.channels['fuel_inj_p'].data, 
                      self.channels['fuel_inlet_p'].data, self.channels['ox_inj_p'].data],
                'xlabel': 'Time (s)',
                'ylabel': 'Pressure (bar(a))',
                'title': 'All Pressures',
                'label': ['Chamber', 'Fuel Injector', 'Fuel Inlet', 'Ox Injector'],
                'color': ['orange', 'red', 'purple', 'blue'],
                'linestyle': ['-', '-', '-', '-']
            },
            {
                'x': [self.time, self.time],
                'y': [self.channels['ox_mdot'].data, self.channels['fuel_mdot'].data],
                'xlabel': 'Time (s)',
                'ylabel': 'Mass Flow (kg/s)',
                'title': 'Mass Flows',
                'label': ['Oxidizer', 'Fuel'],
                'color': ['blue', 'red'],
                'linestyle': ['-', '-']
            },
            {
                'x': [self.time, self.time, self.time],
                'y': [self.ox_inj_CdA*1e6, self.fuel_inj_CdA*1e6, self.regen_CdA*1e6],
                'xlabel': 'Time (s)',
                'ylabel': 'CdA (mm²)',
                'title': 'CdA Values',
                'label': ['Ox Injector', 'Fuel Injector', 'Regen'],
                'color': ['blue', 'red', 'purple'],
                'linestyle': ['-', '-', '-']
            }
        ]
        
        self.subplot(data_sets, 2, 3, figsize=figsize,
                    suptitle=f'Test Data - {self.test_name}',
                    show=show, apply_filter=apply_filter, cutoff_freq=cutoff_freq, ylims=ylims)

    def close(self):
        """Close the h5 file"""
        self.data_file.close()

class channel:
    def __init__(self, channel, data, zero_start=None, zero_end=None):
        self.data = data['channels'][channel]['data'][:]
        self.time = data['channels'][channel]['time'][:]
        self.name = data['channels'][channel].attrs['name']
        self.units = data['channels'][channel].attrs['units']
        if self.units == 'bar(g)':
            self.data = self.data + 1
            self.units = 'bar(a)'
        self.zero(zero_start, zero_end)
        self.filter()

    def plot(self, show_filtered=False):
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.data, label='Raw')
        if show_filtered:
            plt.plot(self.time, self.filt_data, label='Filtered')
            plt.legend()
        plt.xlabel('Time')
        plt.ylabel(self.units)
        plt.title(self.name)
        plt.grid()
        plt.grid(which="minor", alpha=0.5)
        plt.minorticks_on()

    def filter(self):
        cutoff_freq = 100
        dt = np.mean(np.diff(self.time))
        fs = 1 / dt
        self.filt_data = butter_lowpass_filter(self.data, cutoff_freq, fs)

    def zero(self, t_start, t_end):
        """Zero the channel data between t_start and t_end"""
        if t_start is not None and t_end is not None:
            mask = (self.time >= t_start) & (self.time <= t_end)
            offset = np.mean(self.data[mask])
            self.data -= offset

pluto_engine = engine(
    chamber_d=92e-3,  # mm
    # throat_d=42.41e-3,  # mm
    throat_d=42.2e-3,  # mm
    exit_d=82.23e-3,
)

hotfire_008 = test_data_analysis("hotfire_008", "C:\\Users\\Rudra\\Desktop\\Propulsion\\Test Data\\Race2Space\\2025\\20250704_ICLR\\20250704-008-release.h5", pluto_engine)
hotfire_008.calc_exp_performance()
hotfire_008.calc_theoretical_performance()
hotfire_008.calc_efficiencies()
hotfire_008.plot_efficiencies(apply_filter=True, cutoff_freq=100, ylims=(0.7, 1.1))
hotfire_008.plot_performance_comparison(apply_filter=True, cutoff_freq=100)
hotfire_008.subplot([
    {
        'x': hotfire_008.time,
        'y': hotfire_008.channels['thrust'].data,
        'xlabel': 'Time (s)',
        'ylabel': 'Thrust (N)',
        'title': 'Thrust',
        'color': 'blue'
    },
    {
        'x': hotfire_008.time,
        'y': hotfire_008.channels['chamber_p'].data,
        'xlabel': 'Time (s)',
        'ylabel': 'Chamber Pressure (bar)',
        'title': 'Chamber Pressure',
        'color': 'red'
    }
], 2, 1, figsize=(12, 8), suptitle=f'Thrust and Chamber Pressure - {hotfire_008.test_name}', apply_filter=True, cutoff_freq=100)
hotfire_008.close()

hotfire_007 = test_data_analysis("hotfire_007", "2025/20250704_ICLR/20250704-007-release.h5", pluto_engine)
hotfire_007.calc_exp_performance()
hotfire_007.calc_theoretical_performance()
hotfire_007.calc_efficiencies()
hotfire_007.plot_efficiencies(apply_filter=True, cutoff_freq=100, ylims=(0.7, 1.1))
hotfire_007.plot_performance_comparison(apply_filter=True, cutoff_freq=100)
hotfire_007.close()

hotfire_006 = test_data_analysis("hotfire_006", "2025/20250704_ICLR/20250704-006-release.h5", pluto_engine)
hotfire_006.calc_exp_performance()
hotfire_006.calc_theoretical_performance()
hotfire_006.calc_efficiencies()
hotfire_006.plot_efficiencies(apply_filter=True, cutoff_freq=100, ylims=(0.7, 1.1))
hotfire_006.plot_performance_comparison(apply_filter=True, cutoff_freq=100)
hotfire_006.close()


hotfire_005 = test_data_analysis("hotfire_005", "2025/20250704_ICLR/20250704-005-release.h5", pluto_engine)
hotfire_005.calc_exp_performance()
hotfire_005.calc_theoretical_performance()
hotfire_005.calc_efficiencies()
hotfire_005.plot_efficiencies(apply_filter=True, cutoff_freq=100, ylims=(0.7, 1.1))
hotfire_005.plot_performance_comparison(apply_filter=True, cutoff_freq=100)
hotfire_005.close()

hotfire_003 = test_data_analysis("hotfire_003", "2025/20250704_ICLR/20250704-003-release.h5", pluto_engine)
hotfire_003.calc_exp_performance()
hotfire_003.calc_theoretical_performance()
hotfire_003.calc_efficiencies()
hotfire_003.plot_efficiencies(apply_filter=True, cutoff_freq=50, ylims=(0.7, 1.1))
hotfire_003.plot_performance_comparison(apply_filter=True, cutoff_freq=50)
hotfire_003.close()

hotfire_002 = test_data_analysis("hotfire_002", "2025/20250704_ICLR/20250704-002-release.h5", pluto_engine)
hotfire_002.calc_exp_performance()
hotfire_002.calc_theoretical_performance()
hotfire_002.calc_efficiencies()
hotfire_002.plot_efficiencies(apply_filter=True, cutoff_freq=50, ylims=(0.7, 1.1))
hotfire_002.plot_performance_comparison(apply_filter=True, cutoff_freq=50)
hotfire_002.close()

# hotfire_001 = test_data_analysis("hotfire_001", "2025/20250704_ICLR/20250704-001-release.h5", pluto_engine)
# hotfire_001.calc_exp_performance()
# hotfire_001.calc_theoretical_performance()
# hotfire_001.calc_efficiencies()
# hotfire_001.plot_efficiencies(apply_filter=True, cutoff_freq=100, ylims=(0.7, 1.1))
# hotfire_001.plot_performance_comparison(apply_filter=True, cutoff_freq=100)
# hotfire_001.close()