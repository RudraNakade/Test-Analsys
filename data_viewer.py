import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import os

class Dataset:
    """
    A class to handle CSV data loading, channel management, and data processing
    for propulsion test data analysis.
    """
    
    def __init__(self, csv_path: str, time_column: str = 'BackendTime'):
        """
        Initialize Dataset with a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file
            time_column (str): Name of the time column (default: 'BackendTime')
        """
        self.csv_path = csv_path
        self.time_column = time_column
        self.data = None
        self.channels = {}
        self.channel_names = []
        self.time_s = None  # Time in seconds (relative)
        self.backend_time = None  # Original backend time
        self.backend_time_start = None
        
        self.load_data()
    
    def load_data(self):
        """Load data from CSV file and extract channels."""
        try:
            self.data = pd.read_csv(self.csv_path)
            self.channel_names = [col for col in self.data.columns if col != self.time_column]
            
            # Store time data
            if self.time_column in self.data.columns:
                self.backend_time = np.array(self.data[self.time_column])
                self.backend_time_start = self.backend_time[0]
                self.time_s = (self.backend_time - self.backend_time_start) / 1e3  # Convert to seconds
            else:
                raise ValueError(f"Time column '{self.time_column}' not found in CSV")
            
            # Store each channel's data
            for channel in self.channel_names:
                self.channels[channel] = np.array(self.data[channel])
                
            print(f"Loaded dataset from {self.csv_path}")
            print(f"Available channels:")
            for channel in self.channel_names:
                print(f"-  {channel}")
            print()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def get_channel_data(self, channel_name: str) -> np.ndarray:
        """
        Get data for a specific channel.
        
        Args:
            channel_name (str): Name of the channel
            
        Returns:
            np.ndarray: Channel data
        """
        if channel_name not in self.channels:
            raise ValueError(f"Channel '{channel_name}' not found. Available channels: {self.channel_names}")
        return self.channels[channel_name]
    
    def get_time_data(self) -> np.ndarray:
        """
        Get time data in seconds.
        
        Returns:
            np.ndarray: Time data in seconds
        """
        return self.time_s
    
    def get_backend_time_data(self) -> np.ndarray:
        """
        Get backend time data.
        
        Returns:
            np.ndarray: Backend time data
        """
        return self.backend_time
    
    def synchronize_with_dataset(self, other_dataset: 'Dataset', method: str = 'trim'):
        """
        Synchronize this dataset with another dataset.
        
        Args:
            other_dataset (Dataset): Other dataset to synchronize with
            method (str): Synchronization method ('trim' or 'interpolate')
        """
        if method == 'trim':
            # Find common time range and trim both datasets
            min_time = max(self.time_s.min(), other_dataset.time_s.min())
            
            # Filter this dataset
            mask = self.time_s >= min_time
            self._apply_mask(mask)
    
    def filter_time_range(self, start_time: float, end_time: float):
        """
        Filter data to a specific time range.
        
        Args:
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
        """
        mask = (self.time_s >= start_time) & (self.time_s <= end_time)
        self._apply_mask(mask)
    
    def _apply_mask(self, mask: np.ndarray):
        """
        Apply a boolean mask to filter all data arrays.
        
        Args:
            mask (np.ndarray): Boolean mask to apply
        """
        # Update time data
        self.time_s = self.time_s[mask]
        self.backend_time = self.backend_time[mask]
        
        # Update channels
        for channel in self.channel_names:
            self.channels[channel] = self.channels[channel][mask]
        
        # Update dataframe
        self.data = self.data[mask].reset_index(drop=True)

    def trim_and_save(self, start_time_s: float, end_time_s: float,  output_path: str = None, output_suffix: str = "_trimmed"):
        """
        Trim dataset to specified time range and save to new CSV file.
        
        Args:
            start_time_s (float): Start time in seconds
            end_time_s (float): End time in seconds
            output_suffix (str): Suffix to add to output filename (default: "_trimmed")
            output_path (str, optional): Full path for output file. If None, uses original path with suffix
        """
        # Create a copy of current state to restore later
        original_time_s = self.time_s.copy()
        original_backend_time = self.backend_time.copy()
        original_channels = {name: data.copy() for name, data in self.channels.items()}
        original_data = self.data.copy()
        
        try:
            # Apply time filter
            self.filter_time_range(start_time_s, end_time_s)
            
            # Create output filename
            if output_path is None:
                base_path, ext = os.path.splitext(self.csv_path)
                output_path = f"{base_path}{output_suffix}{ext}"
            
            # Create new dataframe with trimmed data
            trimmed_data = pd.DataFrame()
            trimmed_data[self.time_column] = self.backend_time
            
            for channel in self.channel_names:
                trimmed_data[channel] = self.channels[channel]
            
            # Save to CSV
            trimmed_data.to_csv(output_path, index=False)
            print(f"Trimmed dataset saved to: {output_path}")
            print(f"Time range: {start_time_s:.3f}s to {end_time_s:.3f}s")
            print(f"Data points: {len(trimmed_data)}")
            
        finally:
            # Restore original state
            self.time_s = original_time_s
            self.backend_time = original_backend_time
            self.channels = original_channels
            self.data = original_data
    
    def calculate_differential(self, channel1: str, channel2: str, name: str = None) -> str:
        """
        Calculate differential between two channels and add as new channel.
        
        Args:
            channel1 (str): First channel name
            channel2 (str): Second channel name
            name (str, optional): Name for the differential channel
            
        Returns:
            str: Name of the created differential channel
        """
        if name is None:
            name = f"{channel1}_{channel2}_diff"
        
        data1 = self.get_channel_data(channel1)
        data2 = self.get_channel_data(channel2)
        
        # Ensure same length
        min_length = min(len(data1), len(data2))
        diff_data = data1[:min_length] - data2[:min_length]
        
        self.channels[name] = diff_data
        self.channel_names.append(name)
        
        return name
    
    def get_summary(self) -> Dict:
        """Get summary statistics for all channels."""
        summary = {}
        for channel in self.channel_names:
            data = self.channels[channel]
            summary[channel] = {
                'min': np.min(data),
                'max': np.max(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'length': len(data)
            }
        return summary
    
    def backend_time_from_seconds(self, time_sec: float) -> float:
        """
        Convert relative time in seconds back to absolute BackendTime.
        
        Args:
            time_sec (float): Relative time in seconds
            
        Returns:
            float: Absolute BackendTime
        """
        if self.backend_time_start is None:
            raise ValueError("Backend time start not set.")
        return self.backend_time_start + time_sec * 1e3

    def joint_trim_and_save(self, other_datasets: List['Dataset'], start_time_s: float, end_time_s: float, 
                           output_paths: List[str] = None, output_suffix: str = "_trimmed", 
                           output_folder: str = None, output_names: List[str] = None):
        """
        Trim this dataset and other datasets to the same time range and save all to CSV files.
        
        Args:
            other_datasets (List[Dataset]): List of other datasets to trim to same time range
            start_time_s (float): Start time in seconds (relative to this dataset)
            end_time_s (float): End time in seconds (relative to this dataset)
            output_paths (List[str], optional): List of full paths for output files. If None, uses original paths with suffix
            output_suffix (str): Suffix to add to output filenames (default: "_trimmed")
            output_folder (str, optional): Output folder path. If provided, overrides original file locations
            output_names (List[str], optional): List of custom output filenames (without extension). If provided, must match number of datasets
        """
        # Convert relative time to backend time for this dataset
        start_backend_time = self.backend_time_from_seconds(start_time_s)
        end_backend_time = self.backend_time_from_seconds(end_time_s)
        
        # Prepare output paths
        if output_paths is None:
            output_paths = []
            all_datasets = [self] + other_datasets
            
            if output_names is not None:
                if len(output_names) != len(all_datasets):
                    raise ValueError(f"Number of output names ({len(output_names)}) must match number of datasets ({len(all_datasets)})")
                
                # Use custom names
                for i, (dataset, name) in enumerate(zip(all_datasets, output_names)):
                    if output_folder is not None:
                        output_path = os.path.join(output_folder, f"{name}.csv")
                    else:
                        # Use original folder with custom name
                        original_folder = os.path.dirname(dataset.csv_path)
                        output_path = os.path.join(original_folder, f"{name}.csv")
                    output_paths.append(output_path)
            else:
                # Use original names with suffix
                for dataset in all_datasets:
                    base_path, ext = os.path.splitext(dataset.csv_path)
                    if output_folder is not None:
                        # Use custom folder with original filename + suffix
                        original_filename = os.path.basename(base_path)
                        output_path = os.path.join(output_folder, f"{original_filename}{output_suffix}{ext}")
                    else:
                        # Use original location with suffix
                        output_path = f"{base_path}{output_suffix}{ext}"
                    output_paths.append(output_path)
        else:
            if len(output_paths) != len(other_datasets) + 1:
                raise ValueError(f"Number of output paths ({len(output_paths)}) must match number of datasets ({len(other_datasets) + 1})")
        
        # Create output folder if it doesn't exist
        if output_folder is not None and not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        
        # Store original states for all datasets
        original_states = []
        
        # Store this dataset's original state
        original_states.append({
            'dataset': self,
            'time_s': self.time_s.copy(),
            'backend_time': self.backend_time.copy(),
            'channels': {name: data.copy() for name, data in self.channels.items()},
            'data': self.data.copy()
        })
        
        # Store other datasets' original states
        for dataset in other_datasets:
            original_states.append({
                'dataset': dataset,
                'time_s': dataset.time_s.copy(),
                'backend_time': dataset.backend_time.copy(),
                'channels': {name: data.copy() for name, data in dataset.channels.items()},
                'data': dataset.data.copy()
            })
        
        try:
            # Trim this dataset
            self.filter_time_range(start_time_s, end_time_s)
            
            # Trim other datasets to same backend time range
            for dataset in other_datasets:
                # Convert backend times to relative seconds for each dataset
                dataset_start_s = (start_backend_time - dataset.backend_time_start) / 1e3
                dataset_end_s = (end_backend_time - dataset.backend_time_start) / 1e3
                
                # Apply filter
                dataset.filter_time_range(dataset_start_s, dataset_end_s)
            
            # Save all datasets
            all_datasets = [self] + other_datasets
            
            for i, (dataset, output_path) in enumerate(zip(all_datasets, output_paths)):
                # Create new dataframe with trimmed data
                trimmed_data = pd.DataFrame()
                trimmed_data[dataset.time_column] = dataset.backend_time
                
                for channel in dataset.channel_names:
                    trimmed_data[channel] = dataset.channels[channel]
                
                # Save to CSV
                trimmed_data.to_csv(output_path, index=False)
                print(f"Dataset {i+1} saved to: {output_path}")
                print(f"  Data points: {len(trimmed_data)}")
            
            print(f"Joint trim completed for {len(all_datasets)} datasets")
            print(f"Time range: {start_time_s:.3f}s to {end_time_s:.3f}s")
            print(f"Backend time range: {start_backend_time:.0f} to {end_backend_time:.0f}")
            
        finally:
            # Restore original states for all datasets
            for state in original_states:
                dataset = state['dataset']
                dataset.time_s = state['time_s']
                dataset.backend_time = state['backend_time']
                dataset.channels = state['channels']
                dataset.data = state['data']

class Plotter:
    """
    A simplified class to handle plotting of dataset channels.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (12, 8)):
        """
        Initialize Plotter.
        
        Args:
            figsize (tuple): Figure size (width, height)
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
                           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        self.color_index = 0
    
    def create_plot(self, dataset: Dataset, channel: str, 
                   label: str = None, color: str = None, linestyle: str = '-',
                   title: str = None, xlabel: str = "Time", ylabel: str = None, **kwargs):
        """
        Create a new plot with the first channel.
        
        Args:
            dataset (Dataset): Dataset containing the channel
            channel (str): Channel name to plot
            label (str, optional): Label for the line
            color (str, optional): Color for the line
            linestyle (str): Line style
            title (str, optional): Plot title
            xlabel (str): X-axis label
            ylabel (str, optional): Y-axis label
            **kwargs: Additional matplotlib plot arguments
        """
        # Create new figure and axes
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        # Get data
        y_data = dataset.get_channel_data(channel)
        x_data = dataset.get_time_data()
        
        # Ensure same length
        min_length = min(len(x_data), len(y_data))
        x_data = x_data[:min_length]
        y_data = y_data[:min_length]
        
        # Set defaults
        if label is None:
            label = channel
        if color is None:
            color = self.color_cycle[self.color_index % len(self.color_cycle)]
            self.color_index += 1
        
        # Plot
        self.ax.plot(x_data, y_data, label=label, color=color, linestyle=linestyle, **kwargs)
        
        # Set labels
        if title:
            self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)
        else:
            self.ax.set_ylabel(channel)
        
        # Add grid
        self.ax.grid(True)
    
    def add_plot(self, dataset: Dataset, channel: str, 
                label: str = None, color: str = None, linestyle: str = '-', **kwargs):
        """
        Add another channel to the existing plot.
        
        Args:
            dataset (Dataset): Dataset containing the channel
            channel (str): Channel name to plot
            label (str, optional): Label for the line
            color (str, optional): Color for the line
            linestyle (str): Line style
            **kwargs: Additional matplotlib plot arguments
        """
        if self.ax is None:
            raise ValueError("Must create a plot first using create_plot()")
        
        # Get data
        y_data = dataset.get_channel_data(channel)
        x_data = dataset.get_time_data()
        
        # Ensure same length
        min_length = min(len(x_data), len(y_data))
        x_data = x_data[:min_length]
        y_data = y_data[:min_length]
        
        # Set defaults
        if label is None:
            label = channel
        if color is None:
            color = self.color_cycle[self.color_index % len(self.color_cycle)]
            self.color_index += 1
        
        # Plot
        self.ax.plot(x_data, y_data, label=label, color=color, linestyle=linestyle, **kwargs)
    
    def show_plot(self):
        """
        Finalize and show the plot.
        """
        if self.ax is None:
            raise ValueError("No plot to show. Create a plot first using create_plot()")
        
        # Add legend if there are multiple lines
        handles, labels = self.ax.get_legend_handles_labels()
        if len(handles) > 1:
            self.ax.legend()
        
        # Apply tight layout and show
        self.fig.tight_layout()
        # plt.show()

if __name__ == "__main__":
    from os import system
    system('cls')

    kermit_path = "C:\\Users\\Rudra\\Desktop\\Propulsion\\Test Data\\2025\\29_08_25_PLUTO_FLIGHT_QUAL\\sample_data\\raw_data\\DMTHotfireSen0.csv"
    greg_path = "C:\\Users\\Rudra\\Desktop\\Propulsion\\Test Data\\2025\\29_08_25_PLUTO_FLIGHT_QUAL\\sample_data\\raw_data\\20250828_205052.521507Z_plt_greg_telem.csv"
    stark_path = "C:\\Users\\Rudra\\Desktop\\Propulsion\\Test Data\\2025\\29_08_25_PLUTO_FLIGHT_QUAL\\sample_data\\raw_data\\DMTHotfireStark.csv"

    kermit_data = Dataset(kermit_path)
    greg_data = Dataset(greg_path)
    stark_data = Dataset(stark_path)

    # stark_plotter = Plotter(figsize=(15, 10))
    # stark_plotter.create_plot(stark_data, 'ch0sens', label='ch0sens', title='Stark Data Channels')
    # stark_plotter.add_plot(stark_data, 'ch1sens', label='ch1sens')
    # stark_plotter.add_plot(stark_data, 'ch2sens', label='ch2sens')
    # stark_plotter.add_plot(stark_data, 'ch3sens', label='ch3sens')
    # stark_plotter.add_plot(stark_data, 'ch4sens', label='ch4sens')
    # stark_plotter.add_plot(stark_data, 'ch5sens', label='ch5sens')
    # stark_plotter.add_plot(stark_data, 'flowmeter', label='flowmeter')
    # stark_plotter.add_plot(stark_data, 'oxAngle', label='oxAngle')
    # stark_plotter.add_plot(stark_data, 'fuelAngle', label='fuelAngle')
    # stark_plotter.show_plot()

    # kermit_plotter = Plotter(figsize=(15, 10))
    # kermit_plotter.create_plot(kermit_data, 'ch0sens', title='Kermit Data Channels')
    # kermit_plotter.add_plot(kermit_data, 'ch1sens')
    # kermit_plotter.add_plot(kermit_data, 'ch2sens')
    # kermit_plotter.add_plot(kermit_data, 'ch3sens')
    # kermit_plotter.add_plot(kermit_data, 'temp0')
    # kermit_plotter.add_plot(kermit_data, 'temp1')
    # kermit_plotter.show_plot()

    # greg_plotter = Plotter(figsize=(15, 10))
    # greg_plotter.create_plot(greg_data, 'Feedforward', title='Greg Data Channels')
    # greg_plotter.add_plot(greg_data, 'FuelTankP')
    # greg_plotter.add_plot(greg_data, 'regAngle')
    # greg_plotter.add_plot(greg_data, 'Proportional_Term')
    # greg_plotter.add_plot(greg_data, 'Kp')
    # greg_plotter.show_plot()

    # stark_data.joint_trim_and_save([kermit_data], 12630, 12910)
    plt.show()