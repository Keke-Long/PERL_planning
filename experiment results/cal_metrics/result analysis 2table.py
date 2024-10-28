'''
Load data from ./data/NGSIM_I80_results/
Calculate all the required metrics
'''

import pandas as pd
import numpy as np
import math
import os


def calculate_ttc(data, pos_column, vel_column, lead_pos_column, lead_vel_column):
    # Calculate relative distance and relative velocity
    relative_distance = data[lead_pos_column] - data[pos_column]
    relative_velocity = data[vel_column] - data[lead_vel_column]

    # Calculate TTC, only when relative velocity is positive
    ttc = np.where(relative_velocity > 0, relative_distance / relative_velocity, np.inf)

    # Calculate minimum TTC and total duration where TTC is less than 2 seconds
    min_ttc = np.min(ttc[np.isfinite(ttc)])
    ttc_under_2 = np.sum(ttc < 2) * (data['t1'][1] - data['t1'][0])  # Assuming time intervals are equal

    return min_ttc, ttc_under_2, ttc


def analyze_ttc(file_path):
    data = pd.read_csv(file_path)

    # Define baselines
    baselines = {
        'Baseline 1': ('Y5', 'v5'),
        'Baseline 2': ('Y5_mpc_base2', 'v5_mpc_base2'),
        'Baseline 3': ('Y5_mpc_base3', 'v5_mpc_base3')
    }

    for baseline, (pos_col, vel_col) in baselines.items():
        if pos_col in data.columns and vel_col in data.columns:
            min_ttc, ttc_under_2, _ = calculate_ttc( data, pos_col, vel_col, 'Y4', 'v4' )
            print(f"{baseline} Minimum TTC: {min_ttc:.2f} seconds")
            # print(f"{baseline} Total Time with TTC < 2 seconds: {ttc_under_2} seconds")
        else:
            print(f"{baseline} data not available.")



def second_order_difference(data):
    """Calculate second-order difference"""
    first_diff = data.diff().dropna()  # Drop NaN resulting from the difference
    second_diff = first_diff.diff().dropna()
    return second_diff


def calculate_rmse(data, window_size=20):
    """Calculate RMSE over every 20 steps and return the mean of all RMSE values"""
    # Initialize a list to store RMSE values for each segment
    rmse_values = []
    for start in range(0, len(data), window_size):
        # Ensure index bounds are not exceeded at the end of the data
        end = start + window_size
        segment = data[start:end]

        # Calculate RMSE for this segment only if the length equals window_size
        if len(segment) == window_size:
            mean_diff = segment.mean()
            rmse = np.sqrt(((segment - mean_diff) ** 2).mean())
            rmse_values.append(rmse)

    # Compute the mean of all RMSE values
    if rmse_values:
        average_rmse = np.mean(rmse_values)
    else:
        average_rmse = np.nan  # Return NaN if no valid segments were found

    return average_rmse


def analyze_velocity_oscillation(file_path):
    data = pd.read_csv(file_path)

    # Define columns for vehicle speed data
    velocity_columns = {
        'Baseline 1': 'v5',
        'Baseline 2': 'v5_mpc_base2',
        'Baseline 3': 'v5_mpc_base3'
    }

    # Analyze each data column
    for label, column in velocity_columns.items():
        if column in data.columns:
            speed_data = data[column].dropna()
            #print(f"Analyzing {label} with {len(speed_data)} data points.")
            second_diff_speed = second_order_difference(speed_data)
            speed_rmse = calculate_rmse(second_diff_speed)
            print(f"{label} traffic oscillation metric: {speed_rmse:.4f}")
        else:
            print(f"{label} column not found in data.")



def VT_Micro(v, a):
    """Calculate instantaneous fuel consumption given speed v and acceleration a"""
    # Define matrices
    PE = np.matrix([[-8.27978, 0.36696, -0.04112, 0.00139],
                    [0.06229, -0.02143, 0.00245, 3.71 * 10 ** (-6)],
                    [-0.00124, 0.000518, 6.77 * 10 ** (-6), -7.4 * 10 ** (-6)],
                    [7.72 * 10 ** (-6), -2.3 * 10 ** (-6), -5 * 10 ** (-7), 1.05 * 10 ** (-7)]])

    NE = np.matrix([[-8.27978, -0.27907, -0.05888, -0.00477],
                    [0.06496, 0.03282, 0.00705, 0.000434],
                    [-0.00131, -0.00066, -0.00013, -7.6 * 10 ** (-6)],
                    [8.23 * 10 ** (-6), 3.54 * 10 ** (-6), 6.48 * 10 ** (-7), 3.98 * 10 ** (-8)]])

    # Select matrix based on the sign of acceleration
    if a >= 0:
        fuel = np.exp(np.matrix([1, v, np.power(v, 2), np.power(v, 3)]) * PE * np.transpose(np.matrix([1, a, np.power(a, 2), np.power(a, 3)])))
    else:
        fuel = np.exp(np.matrix([1, v, np.power(v, 2), np.power(v, 3)]) * NE * np.transpose(np.matrix([1, a, np.power(a, 2), np.power(a, 3)])))
    return fuel[0, 0]

def calculate_total_fuel_per_100km(data, v_col, a_col, distance_col):
    """Calculate total fuel consumption based on speed and acceleration columns"""
    total_fuel = 0
    for index, row in data.iterrows():
        v = row[v_col]
        a = row[a_col]
        if pd.notna(v) and pd.notna(a):
            total_fuel += VT_Micro(v, a)

    # Calculate total distance using the max and min values of the distance column
    total_distance = data[distance_col].max() - data[distance_col].min()
    #print('total_distance',total_distance)
    # Convert fuel consumption to per 100 km
    fuel_per_100km = (total_fuel / (total_distance/100000) ) if total_distance > 0 else float('inf')
    return fuel_per_100km

def analyze_fuel_consumption(data):
    trajectories = {
        'Baseline 1': ('v5', 'a5', 'Y5'),
        'Baseline 2': ('v5_mpc_base2', 'a5_mpc_base2', 'Y5_mpc_base2'),
        'Baseline 3': ('v5_mpc_base3', 'a5_mpc_base3', 'Y5_mpc_base3')
    }
    # Calculate fuel consumption for each trajectory
    for label, (v_col, a_col, distance_col) in trajectories.items():
        fuel_per_100km = calculate_total_fuel_per_100km(data, v_col, a_col, distance_col)
        print(f"Fuel consumption per 100 km for {label}: {fuel_per_100km:.2f}")
    else:
        print(f"Data for {label} not available.")


def calculate_relative_dampening_ratio(data, local_accel_column, reference_accel_column):
    """Calculate cumulative dampening ratio dp,i relative to the reference vehicle"""
    reference_accel = data[reference_accel_column]  # Reference vehicle's acceleration
    local_accel = data[local_accel_column]  # Local vehicle's acceleration
    dp_i = np.linalg.norm(local_accel, 2) / np.linalg.norm(reference_accel, 2)
    return dp_i


def calculate_local_stability(data, pos_column, vel_column):
    """Calculate local stability metrics Δdt and Δvt"""
    delta_dt = data[pos_column].diff().dropna()  # Position difference
    delta_vt = data[vel_column].diff().dropna()  # Speed difference
    return delta_dt, delta_vt



# Load data
file_name = '../../data/NGSIM_I80_results1/1_11_24_44_55_54.csv'
print('Process', file_name)
data = pd.read_csv(file_name)
data = pd.read_csv(file_name).iloc[0:200]

# Fuel consumption analysis
analyze_fuel_consumption(data)


# Calculate cumulative dampening ratio for the fifth vehicle relative to the fourth vehicle
dp_base1 = calculate_relative_dampening_ratio(data, 'a5', 'a4')
dp_base2 = calculate_relative_dampening_ratio(data, 'a5_mpc_base2', 'a4')
dp_base3 = calculate_relative_dampening_ratio(data, 'a5_mpc_base3', 'a4')
print(f"Cumulative Dampening Ratio base1: {dp_base1:.2f}")
print(f"Cumulative Dampening
