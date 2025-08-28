import pandas
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
from rocketcea.cea_obj_w_units import CEA_Obj

kermit_cf = pandas.read_csv("2025\\23_02_25_HOPPER_ENGINE\\raw_data\\20250223_142750.968298Z_sts_sen0_telem.csv")
stark_cf  = pandas.read_csv("2025\\23_02_25_HOPPER_ENGINE\\raw_data\\20250223_142750.966056Z_sts_stark_telem.csv")

output_folder = "2025\\23_02_25_HOPPER_ENGINE\\cold_flow_1\\"

min_time = stark_cf['BackendTime'][0]
kermit_cf = pandas.DataFrame(kermit_cf[kermit_cf['BackendTime'] > min_time])
stark_cf = pandas.DataFrame(stark_cf[stark_cf['BackendTime'] > min_time])

stark_cfTime = stark_cf['BackendTime']
kermit_cfTime = kermit_cf['BackendTime']

kermit_cfTime = (kermit_cfTime - stark_cfTime[stark_cfTime.index[0]]) / 1e3
stark_cfTime = (stark_cfTime - stark_cfTime[stark_cfTime.index[0]]) / 1e3

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

# Identify continuous regions where both valves are open
valves_open_indices = []
for idx in range(len(fuel_main_ang)):
    if fuel_main_ang[idx] > 0 and ox_main_ang[idx] > 0:
        valves_open_indices.append(idx)

# Find continuous segments of valve openings
valve_open_segments = []
if valves_open_indices:
    # Start first segment
    current_segment = [valves_open_indices[0]]
    
    # Check for continuity and create segments
    for i in range(1, len(valves_open_indices)):
        if valves_open_indices[i] == valves_open_indices[i-1] + 1:
            # Continuous - add to current segment
            current_segment.append(valves_open_indices[i])
        else:
            # Gap found - end current segment and start a new one
            valve_open_segments.append(current_segment)
            current_segment = [valves_open_indices[i]]
    
    # Add the last segment
    valve_open_segments.append(current_segment)

# Calculate duration of each segment
valve_open_durations = []
for segment in valve_open_segments:
    start_time = stark_cfTime.iloc[segment[0]]
    end_time = stark_cfTime.iloc[segment[-1]]
    duration = end_time - start_time
    valve_open_durations.append({
        'start_idx': segment[0],
        'end_idx': segment[-1],
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration
    })

# Original condition check (valves open + pressure thresholds)
filtered_indices = []
for idx in range(len(fuel_main_ang)):
    # Check if fuel and ox main angles are above zero
    if fuel_main_ang[idx] > 0 and ox_main_ang[idx] > 0:
        # Check if all three pressures are above 2
        if (chamber_P[idx] > 2 and 
            fuel_inj_P[idx] > 2 and 
            ox_inj_P[idx] > 2):
            filtered_indices.append(idx)

times = np.array(filtered_indices)

if len(valve_open_segments) > 0:
    # Properly access the first and last segment's start and end times
    first_segment_start = valve_open_durations[0]['start_time']
    last_segment_end = valve_open_durations[-1]['end_time']
    total_span = last_segment_end - first_segment_start
    
    # Print total time both valves were open (sum of all periods)
    total_open_time = sum(segment_info['duration'] for segment_info in valve_open_durations)

if len(times) > 0:
    try:
        first_time = stark_cfTime.iloc[times[0]]
        last_time = stark_cfTime.iloc[times[-1]]
    except IndexError as e:
        print(f"Error accessing time: {e}")
        print(f"stark_cfTime length: {len(stark_cfTime)}")
else:
    print("No data points meeting all conditions detected")

# Find periods where all conditions are met (both valves open AND pressures above threshold)
all_conditions_indices = []
for idx in range(len(fuel_main_ang)):
    # Check if fuel and ox main angles are above zero
    if fuel_main_ang[idx] > 0 and ox_main_ang[idx] > 0:
        # Check if all three pressures are above 2
        if (chamber_P[idx] > 2 and 
            fuel_inj_P[idx] > 2 and 
            ox_inj_P[idx] > 2):
            all_conditions_indices.append(idx)

# Find continuous segments where all conditions are met
all_conditions_segments = []
if all_conditions_indices:
    # Start first segment
    current_segment = [all_conditions_indices[0]]
    
    # Check for continuity and create segments
    for i in range(1, len(all_conditions_indices)):
        if all_conditions_indices[i] == all_conditions_indices[i-1] + 1:
            # Continuous - add to current segment
            current_segment.append(all_conditions_indices[i])
        else:
            # Gap found - end current segment and start a new one
            all_conditions_segments.append(current_segment)
            current_segment = [all_conditions_indices[i]]
    
    # Add the last segment
    all_conditions_segments.append(current_segment)

# Calculate duration of each segment where all conditions are met
all_conditions_durations = []
for segment in all_conditions_segments:
    start_time = stark_cfTime.iloc[segment[0]]
    end_time = stark_cfTime.iloc[segment[-1]]
    duration = end_time - start_time
    all_conditions_durations.append({
        'start_idx': segment[0],
        'end_idx': segment[-1],
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration
    })

# Print information about each segment where all conditions are met
print(f"\nFound {len(all_conditions_segments)} continuous periods where test conditions are met:")
for i, segment_info in enumerate(all_conditions_durations):
    print(f"  Period {i+1}: Start={segment_info['start_time']:.2f}s, End={segment_info['end_time']:.2f}s, Duration={segment_info['duration']:.2f}s")

if len(all_conditions_segments) > 0:
    # Properly access the first and last segment's start and end times
    first_segment_start = all_conditions_durations[0]['start_time']
    last_segment_end = all_conditions_durations[-1]['end_time']
    total_span = last_segment_end - first_segment_start

# Ask user which period to save data for
if len(all_conditions_durations) > 0:
    print("\nWhich period would you like to save trimmed data for?")    
    
    # Add a loop to allow multiple period selections
    selected_period = None
    while True:
        try:
            if selected_period is None:
                period_input = input(f"Enter period number (1-{len(all_conditions_durations)}): ")
            else:
                period_input = input(f"Type 's' to save or enter another period number (1-{len(all_conditions_durations)}): ")
            
            # Check if user wants to save
            if period_input.lower() == 's':
                # Save the currently displayed data
                break
            
            # Otherwise, try to parse as a period number
            selected_period = int(period_input)
            if 1 <= selected_period <= len(all_conditions_durations):
                # Get the selected period info (convert from 1-based to 0-based indexing)
                selected_info = all_conditions_durations[selected_period-1]
                
                # Calculate time boundaries with padding (5s before and after)
                start_time_with_padding = max(0, selected_info['start_time'] - 5)
                end_time_with_padding = selected_info['end_time'] + 5
                
                print(f"\nSelected data from {start_time_with_padding:.2f}s to {end_time_with_padding:.2f}s")
                
                # Filter stark_cf data for the selected period with padding
                stark_mask = (stark_cfTime >= start_time_with_padding) & (stark_cfTime <= end_time_with_padding)
                stark_trimmed = stark_cf.iloc[stark_mask.values]
                
                # Filter kermit_cf data for the selected period with padding
                kermit_mask = (kermit_cfTime >= start_time_with_padding) & (kermit_cfTime <= end_time_with_padding)
                kermit_trimmed = kermit_cf.iloc[kermit_mask.values]
                
                # Create a preview plot of the trimmed data
                plt.figure(figsize=(12, 8))
                
                # Convert the trimmed times to relative times for plotting
                stark_trimmed_times = (stark_trimmed['BackendTime'] - stark_trimmed['BackendTime'].iloc[0]) / 1e3
                
                plt.subplot(2, 1, 1)
                plt.plot(stark_trimmed_times, stark_trimmed['fuelAngle'], label='Fuel Angle')
                plt.plot(stark_trimmed_times, stark_trimmed['oxAngle'], label='Ox Angle')
                plt.title('Trimmed Data Preview - Valve Angles')
                plt.xlabel('Time (s)')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(2, 1, 2)
                plt.plot(stark_trimmed_times, stark_trimmed['ch3sens'], label='Chamber P')
                plt.plot(stark_trimmed_times, stark_trimmed['ch1sens'], label='Fuel Inj P')
                plt.plot(stark_trimmed_times, stark_trimmed['ch0sens'], label='Ox Inj P')
                plt.title('Trimmed Data Preview - Pressures')
                plt.xlabel('Time (s)')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.show()
            else:
                print(f"Please enter a valid number between 1 and {len(all_conditions_durations)}")
        except ValueError:
            if period_input.lower() != 's':
                print("Please enter a valid number or 's' to save")
    
    # Save data if the user has selected a period
    if selected_period is not None:
        # Create output folder if it doesn't exist
        import os
        os.makedirs(output_folder, exist_ok=True)
        
        # Save trimmed data as CSV
        stark_output_path = os.path.join(output_folder, "stark_telem_trim.csv")
        kermit_output_path = os.path.join(output_folder, "kermit_telem_trim.csv")
        
        stark_trimmed.to_csv(stark_output_path, index=False)
        kermit_trimmed.to_csv(kermit_output_path, index=False)
        
        print(f"Saved {len(stark_trimmed)} rows to {stark_output_path}")
        print(f"Saved {len(kermit_trimmed)} rows to {kermit_output_path}")
    else:
        print("Data was not saved.")
    
else:
    print("\nNo periods with all conditions met were found. Nothing to save.")