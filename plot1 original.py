import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
file_path = './data/NGSIM_I80/'

# Get all CSV files
files = [f for f in os.listdir(file_path) if f.endswith('.csv')]

# Loop through each file, plot and save
for file in files:
    data = pd.read_csv(os.path.join(file_path, file))

    # Set up the figure
    fig, ax = plt.subplots(3, 1, figsize=(3, 5), sharex=True)  # Three subplots, sharing x-axis

    # Define vehicle colors and linestyles
    colors = ['#8391a5', '#5c6f89', '#354f6d', '#003153', 'red', '#0d151f']
    linestyles = [(0, (5, 7)), (0, (4, 5)), (0, (3, 3)), (0, (2, 1)), '-', (0, (1, 1))]
    linewidths = [0.5, 0.7, 1, 1.2, 1.5, 2]

    # Plot the acceleration, velocity, and position for each vehicle
    for i in range(6):
        # Acceleration
        ax[0].plot(data['t1'], data[f'a{i+1}'], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i], label=f'Vehicle {i+1}')
        # Velocity
        ax[1].plot(data['t1'], data[f'v{i+1}'], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i], label=f'Vehicle {i+1}')
        # Position
        ax[2].plot(data['t1'], data[f'Y{i+1}'], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i], label=f'Vehicle {i+1}')

    # Set y-axis labels
    ax[0].set_ylabel('Acceleration (m/s²)')
    ax[1].set_ylabel('Velocity (m/s)')
    ax[2].set_ylabel('Position (m)')

    # Set x-axis label
    ax[2].set_xlabel('Time (s)')

    # Set axis limits
    ax[0].set_ylim(-4, 4)
    ax[1].set_ylim(0, 15)
    ax[2].set_ylim(0, 300)

    # Display and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(file_path, file.replace('.csv', '.png')))
    plt.close()
