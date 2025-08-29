import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
from scipy import interpolate
import matplotlib.pyplot as plt

class Dataset:
    """
    A comprehensive class for handling CSV data loading and processing for propulsion test data analysis.
    
    This class provides functionality to load CSV files, manage data channels, perform time synchronization,
    calculate averages, and convert channels to Channel objects. It supports backend time conversion and
    channel renaming operations.
    
    Attributes:
        csv_path (str): Path to the source CSV file
        time_column (str): Name of the time column in the dataset
        data (pd.DataFrame): Raw pandas DataFrame containing all data
        channels (dict): Dictionary mapping channel names to numpy arrays
        channel_names (list): List of available channel names
        time_s (np.ndarray): Time data in seconds (relative to start)
        backend_time (np.ndarray): Original backend time data
        backend_time_start (float): Starting backend time value
    """
    
    def __init__(self, csv_path: str, time_column: str = 'BackendTime', show_channels: bool = False):
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
        
        self.load_data(show_channels)
    
    def load_data(self, show_channels: bool = False):
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

            print(f"Loaded {len(self.channels)} channels from {self.csv_path}\n")
            if show_channels:
                print(f"Available channels:")
                for channel in self.channel_names:
                    print(f"-  {channel}")
                print()
            
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
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
    
    def sync_channel_to_dataset(self, channel_name: str, target_dataset: 'Dataset') -> np.ndarray:
        """
        Synchronize a channel from this dataset to another dataset's time base using linear interpolation.
        
        This method interpolates the specified channel data to match the time base of the target dataset,
        enabling comparison and analysis of data from different sources with potentially different sampling rates.
        
        Args:
            channel_name (str): Name of the channel to synchronize
            target_dataset (Dataset): Target dataset whose time base will be used for interpolation
            
        Returns:
            np.ndarray: Interpolated channel data aligned to target dataset's time base
            
        Raises:
            ValueError: If the specified channel is not found in this dataset
        """
        if channel_name not in self.channels:
            raise ValueError(f"Channel '{channel_name}' not found. Available channels: {self.channel_names}")
        
        # Get source data
        source_data = self.channels[channel_name]
        source_time = self.time_s
        
        # Get target time
        target_time = target_dataset.time_s
        
        # Create interpolation function
        # Use linear interpolation with extrapolation bounds
        interp_func = interpolate.interp1d(
            source_time, source_data, 
            kind='linear', 
            bounds_error=False, 
            fill_value='extrapolate'
        )
        
        # Interpolate to target time base
        synced_data = interp_func(target_time)
        
        return synced_data
    
    def get_channel_average(self, channel_name: str, start_time_s: float, end_time_s: float) -> float:
        """
        Calculate the arithmetic mean of a channel over a specified time period.
        
        This method filters the channel data to the specified time range and computes the average value,
        useful for steady-state analysis and performance calculations.
        
        Args:
            channel_name (str): Name of the channel to analyze
            start_time_s (float): Start time in seconds (relative to dataset start)
            end_time_s (float): End time in seconds (relative to dataset start)
            
        Returns:
            float: Average value of the channel over the specified time period
            
        Raises:
            ValueError: If channel not found or no data exists in the specified time range
        """
        if channel_name not in self.channels:
            raise ValueError(f"Channel '{channel_name}' not found. Available channels: {self.channel_names}")
        
        # Create time mask
        mask = (self.time_s >= start_time_s) & (self.time_s <= end_time_s)
        
        if not np.any(mask):
            raise ValueError(f"No data found in time range {start_time_s:.3f}s to {end_time_s:.3f}s")
        
        # Calculate average
        channel_data = self.channels[channel_name]
        filtered_data = channel_data[mask]
        
        return np.mean(filtered_data)
    
    def rename_channel(self, old_name: str, new_name: str):
        """
        Rename a channel in the dataset.
        
        Args:
            old_name (str): Current name of the channel
            new_name (str): New name for the channel
        """
        if old_name not in self.channels:
            raise ValueError(f"Channel '{old_name}' not found. Available channels: {self.channel_names}")
        
        if new_name in self.channels:
            raise ValueError(f"Channel '{new_name}' already exists. Choose a different name.")
        
        # Update channels dictionary
        self.channels[new_name] = self.channels.pop(old_name)
        
        # Update channel names list
        channel_index = self.channel_names.index(old_name)
        self.channel_names[channel_index] = new_name
        
        # Update dataframe if it exists
        if self.data is not None and old_name in self.data.columns:
            self.data = self.data.rename(columns={old_name: new_name})
        
        print(f"Channel renamed from '{old_name}' to '{new_name}'")

    def convert_to_channel(self, channel_name: str, label: str = None, sync_dataset: 'Dataset' = None, zero_offset: float = 0.0, gain: float = 1.0) -> 'Channel':
        """
        Convert a dataset channel to a standalone Channel class instance.
        
        This method creates a Channel object from the specified dataset channel, optionally applying
        gain and zero offset correction and time synchronization with another dataset.
        
        Args:
            channel_name (str): Name of the channel to convert
            label (str, optional): Custom display name for the channel. Defaults to channel_name
            sync_dataset (Dataset, optional): Dataset to synchronize time base with. If None, uses original time base
            zero_offset (float, optional): Offset value to subtract from channel data. Defaults to 0.0
            gain (float, optional): Gain factor to multiply channel data. Defaults to 1.0
            
        Returns:
            Channel: New Channel instance containing the processed data
            
        Raises:
            ValueError: If the specified channel is not found in this dataset
        """
        if channel_name not in self.channels:
            raise ValueError(f"Channel '{channel_name}' not found. Available channels: {self.channel_names}")
        
        # Get channel data and backend time
        if sync_dataset is not None:
            # Sync to target dataset's time base
            channel_data = self.sync_channel_to_dataset(channel_name, sync_dataset) * gain
            backend_time = sync_dataset.backend_time
        else:
            # Use original time base
            channel_data = self.channels[channel_name]
            backend_time = self.backend_time
        
        # Use custom label if provided, otherwise use channel name
        if label is None:
            label = channel_name

        # Create and return Channel instance
        return Channel(np.array(channel_data), backend_time, label, zero_offset)

class Channel:
    """
    A class representing a single data channel with associated time information.
    
    This class encapsulates channel data with its corresponding time base, providing methods for
    data manipulation, statistical analysis, and interpolation. It supports zero offset correction
    and time-based data retrieval.
    
    Attributes:
        name (str): Display name of the channel
        raw_data (np.ndarray): Original unprocessed channel data
        data (np.ndarray): Processed channel data (with zero offset applied)
        backend_time (np.ndarray): Backend time array
        time_s (np.ndarray): Time in seconds relative to start
        zero_offset (float): Offset value applied to raw data
        backend_time_start (float): Starting backend time value
    """
    
    def __init__(self, data: np.ndarray, backend_time: np.ndarray, name: str, zero_offset: float = 0.0):
        """
        Initialize a Channel with data and time arrays.
        
        Args:
            data (np.ndarray): Channel data values
            backend_time (np.ndarray): Backend time array
            name (str): Name of the channel
            zero_offset (float, optional): Offset to apply to data (default: 0.0)
        """
        self.name = name
        self.raw_data = np.array(data)
        self.backend_time = np.array(backend_time)
        self.zero_offset = zero_offset
        
        # Apply zero offset
        self.data = self.raw_data - zero_offset
        
        # Calculate relative time in seconds
        self.backend_time_start = self.backend_time[0]
        self.time_s = (self.backend_time - self.backend_time_start) / 1e3
        
        # Ensure data and time arrays have same length
        min_length = min(len(self.data), len(self.time_s))
        self.data = self.data[:min_length]
        self.time_s = self.time_s[:min_length]
        self.backend_time = self.backend_time[:min_length]
    
    def set_zero_offset(self, new_offset: float):
        """
        Update the zero offset for the channel data.
        
        Args:
            new_offset (float): New zero offset to apply
        """
        self.zero_offset = new_offset
        self.data = self.raw_data - new_offset
    
    def get_average(self, start_time_s: float, end_time_s: float) -> float:
        """
        Calculate the arithmetic mean of the channel over a specified time period.
        
        Args:
            start_time_s (float): Start time in seconds (relative to channel start)
            end_time_s (float): End time in seconds (relative to channel start)
            
        Returns:
            float: Average value over the specified time period
            
        Raises:
            ValueError: If no data exists in the specified time range
        """
        mask = (self.time_s >= start_time_s) & (self.time_s <= end_time_s)
        
        if not np.any(mask):
            raise ValueError(f"No data found in time range {start_time_s:.3f}s to {end_time_s:.3f}s")
        
        filtered_data = self.data[mask]
        return np.mean(filtered_data)
    
    def interpolate_to_time(self, target_time_s: np.ndarray) -> np.ndarray:
        """
        Interpolate channel data to a new time base using linear interpolation.
        
        This method allows resampling of channel data to match different time bases,
        with extrapolation beyond the original time range when necessary.
        
        Args:
            target_time_s (np.ndarray): Target time array in seconds for interpolation
            
        Returns:
            np.ndarray: Interpolated channel data at the target time points
        """
        interp_func = interpolate.interp1d(
            self.time_s, self.data,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        return interp_func(target_time_s)
    
    def get_value_at_time(self, time_s: float) -> float:
        """
        Get the interpolated channel value at a specific time point.
        
        Args:
            time_s (float): Time in seconds at which to retrieve the value
            
        Returns:
            float: Interpolated channel value at the specified time
        """
        return float(self.interpolate_to_time(np.array([time_s]))[0])

class Plot:
    """
    A class for creating and managing plot configurations with dual y-axis support.
    
    This class allows building plots with primary and secondary y-axes, automatic color cycling,
    and support for both Channel objects and raw dataset channels. It stores plot configuration
    but doesn't create matplotlib objects until rendered.
    
    Attributes:
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel_primary (str): Primary y-axis label
        ylabel_secondary (str): Secondary y-axis label
        primary_lines (list): Configuration for primary axis lines
        secondary_lines (list): Configuration for secondary axis lines
    """
    
    def __init__(self, title: str = None, xlabel: str = "Time (s)", ylabel_primary: str = None, ylabel_secondary: str = None):
        """
        Initialize a Plot.
        
        Args:
            title (str, optional): Plot title
            xlabel (str): X-axis label
            ylabel_primary (str, optional): Primary y-axis label
            ylabel_secondary (str, optional): Secondary y-axis label
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel_primary = ylabel_primary
        self.ylabel_secondary = ylabel_secondary
        
        # Plot data storage
        self.primary_lines = []  # List of (x_data, y_data, label, style_kwargs)
        self.secondary_lines = []  # List of (x_data, y_data, label, style_kwargs)
        
        # Color cycling
        self.primary_color_cycle = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        self.secondary_color_cycle = ['tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        self.primary_color_index = 0
        self.secondary_color_index = 0
    
    def add_channel(self, channel: Channel, label: str = None, secondary_axis: bool = False, **style_kwargs):
        """
        Add a Channel object to the plot configuration.
        
        Args:
            channel (Channel): Channel object containing data and time information
            label (str, optional): Custom label for the line. Defaults to channel name
            secondary_axis (bool): Whether to plot on secondary y-axis. Defaults to False
            **style_kwargs: Additional matplotlib styling arguments (color, linestyle, etc.)
        """
        if label is None:
            label = channel.name
        
        x_data = channel.time_s
        y_data = channel.data
        
        # Set default color if not provided
        if 'color' not in style_kwargs:
            if secondary_axis:
                style_kwargs['color'] = self.secondary_color_cycle[self.secondary_color_index % len(self.secondary_color_cycle)]
                self.secondary_color_index += 1
            else:
                style_kwargs['color'] = self.primary_color_cycle[self.primary_color_index % len(self.primary_color_cycle)]
                self.primary_color_index += 1
        
        # Add to appropriate axis
        if secondary_axis:
            self.secondary_lines.append((x_data, y_data, label, style_kwargs))
        else:
            self.primary_lines.append((x_data, y_data, label, style_kwargs))
    
    def add_dataset_channel(self, dataset: Dataset, channel_name: str, label: str = None, secondary_axis: bool = False, **style_kwargs):
        """
        Add a channel directly from a Dataset to the plot configuration.
        
        Args:
            dataset (Dataset): Dataset containing the channel
            channel_name (str): Name of the channel within the dataset
            label (str, optional): Custom label for the line. Defaults to channel_name
            secondary_axis (bool): Whether to plot on secondary y-axis. Defaults to False
            **style_kwargs: Additional matplotlib styling arguments (color, linestyle, etc.)
        """
        if label is None:
            label = channel_name
        
        x_data = dataset.time_s
        y_data = dataset.channels[channel_name]
        
        # Set default color if not provided
        if 'color' not in style_kwargs:
            if secondary_axis:
                style_kwargs['color'] = self.secondary_color_cycle[self.secondary_color_index % len(self.secondary_color_cycle)]
                self.secondary_color_index += 1
            else:
                style_kwargs['color'] = self.primary_color_cycle[self.primary_color_index % len(self.primary_color_cycle)]
                self.primary_color_index += 1
        
        # Add to appropriate axis
        if secondary_axis:
            self.secondary_lines.append((x_data, y_data, label, style_kwargs))
        else:
            self.primary_lines.append((x_data, y_data, label, style_kwargs))
    
    def render_on_axes(self, ax_primary: plt.Axes, ax_secondary: plt.Axes = None) -> plt.Axes:
        """
        Render the plot configuration on provided matplotlib axes.
        
        This method creates the actual matplotlib plot from the stored configuration,
        including automatic legend generation and grid setup.
        
        Args:
            ax_primary (plt.Axes): Primary matplotlib axes for plotting
            ax_secondary (plt.Axes, optional): Secondary axes for dual y-axis plots
            
        Returns:
            plt.Axes: Secondary axes if created, otherwise None
        """
        # Plot primary lines
        for x_data, y_data, label, style_kwargs in self.primary_lines:
            ax_primary.plot(x_data, y_data, label=label, **style_kwargs)
        
        # Create secondary axis if needed and not provided
        if self.secondary_lines and ax_secondary is None:
            ax_secondary = ax_primary.twinx()
        
        # Plot secondary lines
        if ax_secondary is not None:
            for x_data, y_data, label, style_kwargs in self.secondary_lines:
                ax_secondary.plot(x_data, y_data, label=label, **style_kwargs)
        
        # Set labels
        if self.title:
            ax_primary.set_title(self.title)
        ax_primary.set_xlabel(self.xlabel)
        
        if self.ylabel_primary:
            ax_primary.set_ylabel(self.ylabel_primary)
        if self.ylabel_secondary and ax_secondary is not None:
            ax_secondary.set_ylabel(self.ylabel_secondary)
        
        # Add grid
        ax_primary.grid(True, alpha=0.7)
        ax_primary.minorticks_on()
        ax_primary.grid(which="minor", alpha=0.3)
        
        # Handle legend
        lines1, labels1 = ax_primary.get_legend_handles_labels()
        if ax_secondary is not None:
            lines2, labels2 = ax_secondary.get_legend_handles_labels()
            if lines1 or lines2:
                ax_primary.legend(lines1 + lines2, labels1 + labels2, loc='best')
        elif lines1:
            ax_primary.legend(loc='best')
        
        return ax_secondary

class Figure:
    """
    A wrapper class for matplotlib figures containing a single Plot.
    
    This class manages the creation and display of matplotlib figures for individual plots,
    providing a clean interface for figure management without immediate display.
    
    Attributes:
        plot (Plot): Plot configuration to display
        figsize (tuple): Figure size (width, height) in inches
    """
    
    def __init__(self, plot: Plot, figsize: Tuple[float, float] = (12, 8)):
        """
        Initialize a Figure with a single plot.
        
        Args:
            plot (Plot): Plot object to display
            figsize (tuple): Figure size (width, height)
        """
        self.plot = plot
        self.figsize = figsize
        self._fig = None
        self._ax_primary = None
        self._ax_secondary = None
    
    def create_figure(self):
        """Create the matplotlib figure and render the plot."""
        self._fig, self._ax_primary = plt.subplots(figsize=self.figsize)
        self._ax_secondary = self.plot.render_on_axes(self._ax_primary)
        self._fig.tight_layout()
    
    def show(self):
        """Create and show the figure."""
        if self._fig is None:
            self.create_figure()
        # Don't call plt.show() here - let the user control when to show

class SubplotManager:
    """
    A class for managing multiple plots in a subplot arrangement.
    
    This class handles the creation and layout of multiple plots in a single figure,
    with automatic subplot arrangement calculation and shared axis support.
    
    Attributes:
        plots (List[Plot]): List of Plot objects to display
        nplots (int): Total number of plots
        nrows (int): Number of subplot rows
        ncols (int): Number of subplot columns
        figsize (tuple): Figure size (width, height) in inches
        sharex (bool): Whether to share x-axis across subplots
    """
    
    def __init__(self, plots: List[Plot], nrows: int = None, ncols: int = None, figsize: Tuple[float, float] = None, sharex: bool = False):
        """
        Initialize a SubplotManager.
        
        Args:
            plots (List[Plot]): List of Plot objects to display
            nrows (int, optional): Number of rows. If None, calculated automatically
            ncols (int, optional): Number of columns. If None, calculated automatically
            figsize (tuple, optional): Figure size. If None, calculated based on number of plots
        """
        self.plots = plots
        self.nplots = len(plots)
        
        # Calculate subplot arrangement if not provided
        if nrows is None and ncols is None:
            # Default to single column for small numbers, otherwise try to make roughly square
            if self.nplots <= 3:
                self.nrows, self.ncols = self.nplots, 1
            else:
                self.ncols = int(np.ceil(np.sqrt(self.nplots)))
                self.nrows = int(np.ceil(self.nplots / self.ncols))
        elif nrows is None:
            self.ncols = ncols
            self.nrows = int(np.ceil(self.nplots / ncols))
        elif ncols is None:
            self.nrows = nrows
            self.ncols = int(np.ceil(self.nplots / nrows))
        else:
            self.nrows, self.ncols = nrows, ncols
        
        # Calculate figure size if not provided
        if figsize is None:
            width = min(6 * self.ncols, 24)
            height = min(4 * self.nrows, 12)
            self.figsize = (width, height)
        else:
            self.figsize = figsize
        
        self._fig = None
        self._axes = None
        self.sharex = sharex
    
    def create_figure(self):
        """Create the matplotlib figure with subplots and render all plots."""
        self._fig, axes = plt.subplots(self.nrows, self.ncols, figsize=self.figsize, sharex=self.sharex)
        
        # Handle single subplot case
        if self.nplots == 1:
            axes = [axes]
        elif self.nrows == 1 or self.ncols == 1:
            # axes is 1D array
            pass
        else:
            # axes is 2D array, flatten it
            axes = axes.flatten()
        
        self._axes = axes
        
        # Render each plot
        for i, plot in enumerate(self.plots):
            if i < len(axes):
                plot.render_on_axes(axes[i])
        
        # Hide unused subplots
        for i in range(self.nplots, len(axes)):
            axes[i].set_visible(False)
        
        self._fig.tight_layout()
    
    def show(self):
        """Create and show the subplot figure."""
        if self._fig is None:
            self.create_figure()
        # Don't call plt.show() here - let the user control when to show

class PlottingManager:
    """
    A convenience class for managing multiple figures and subplot managers.
    
    This class provides a centralized way to manage multiple plotting objects,
    allowing batch creation and display of complex multi-figure layouts.
    
    Attributes:
        figures (List[Figure]): List of individual Figure objects
        subplot_managers (List[SubplotManager]): List of SubplotManager objects
    """
    
    def __init__(self):
        """Initialize a PlottingManager."""
        self.figures = []
        self.subplot_managers = []
    
    def add_figure(self, plot: Plot, figsize: Tuple[float, float] = (12, 8)) -> Figure:
        """
        Add a single plot as a figure.
        
        Args:
            plot (Plot): Plot to add
            figsize (tuple): Figure size
            
        Returns:
            Figure: Created figure object
        """
        figure = Figure(plot, figsize)
        self.figures.append(figure)
        return figure
    
    def add_subplot_manager(self, plots: List[Plot], nrows: int = None, ncols: int = None, figsize: Tuple[float, float] = None) -> SubplotManager:
        """
        Add multiple plots as a subplot manager.
        
        Args:
            plots (List[Plot]): List of plots to add
            nrows (int, optional): Number of rows
            ncols (int, optional): Number of columns
            figsize (tuple, optional): Figure size
            
        Returns:
            SubplotManager: Created subplot manager object
        """
        subplot_manager = SubplotManager(plots, nrows, ncols, figsize)
        self.subplot_managers.append(subplot_manager)
        return subplot_manager
    
    def show_all(self):
        """
        Prepare all managed figures and subplot managers for display.
        
        This method creates all matplotlib objects but does not call plt.show().
        The user must call plt.show() separately to actually display the plots.
        """
        for figure in self.figures:
            figure.show()
        
        for subplot_manager in self.subplot_managers:
            subplot_manager.show()
        
        # Note: User should call plt.show() after this to actually display

if __name__ == "__main__":
    from os import system
    system('cls')

    kermit_path = "C:\\Users\\Rudra\\Desktop\\Propulsion\\Test Data\\2025\\29_08_25_PLUTO_FLIGHT_QUAL\\sample_data\\raw_data\\DMTHotfireSen0_trimmed.csv"
    stark_path = "C:\\Users\\Rudra\\Desktop\\Propulsion\\Test Data\\2025\\29_08_25_PLUTO_FLIGHT_QUAL\\sample_data\\raw_data\\DMTHotfireStark_trimmed.csv"
   
    kermit_data = Dataset(kermit_path)
    stark_data = Dataset(stark_path)

    tank_p_1 = kermit_data.convert_to_channel('ch0sens', 'Tank Pressure (1)')
    tank_p_2 = kermit_data.convert_to_channel('ch1sens', 'Tank Pressure (2)')
    tank_p_3 = stark_data.convert_to_channel('ch4sens', 'Tank Pressure (3)', sync_dataset=kermit_data)
    engine_t_1 = kermit_data.convert_to_channel('temp0', 'Engine Temperature (1)')
    engine_t_2 = kermit_data.convert_to_channel('temp1', 'Engine Temperature (2)')
    chamber_p = stark_data.convert_to_channel('ch3sens', 'Chamber Pressure', sync_dataset=kermit_data)

    # Plotting
    pressure_plot = Plot(title='Pressures', xlabel='Time (s)', ylabel_primary='Pressure (bar(a))')
    pressure_plot.add_channel(tank_p_1)
    pressure_plot.add_channel(tank_p_2)
    pressure_plot.add_channel(tank_p_3)
    pressure_plot.add_channel(chamber_p)
    
    temperature_plot = Plot(title='Temperatures', xlabel='Time (s)', ylabel_primary='Temperature (°C)')
    temperature_plot.add_channel(engine_t_1)
    temperature_plot.add_channel(engine_t_2)
    
    subplot_manager = SubplotManager([pressure_plot, temperature_plot], sharex=True)
    subplot_manager.show()
    
    plt.show()