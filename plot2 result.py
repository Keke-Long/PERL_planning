import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_results(file_path):
    # Reload data
    data = pd.read_csv(file_path)
    data = pd.read_csv(file_path).iloc[:230]
    ts = 0.1  # Assume the time step ts is 0.1, adjust based on actual conditions

    # Set up the figure
    fig, ax = plt.subplots(3, 4, figsize=(12, 5), sharex=True)  # Shared x-axis

    # Define vehicle colors and linestyles
    colors = ['#8391a5', '#5c6f89', '#354f6d', '#003153', 'red', '#0d151f']
    linestyles = [(0, (5, 7)), (0, (4, 5)), (0, (3, 3)), (0, (2, 1)), '-', (0, (1, 1))]
    linewidths = [0.6, 0.7, 0.8, 1, 1.2, 1.7]

    # First column
    c = 0
    for i in range(6):
        # Acceleration
        ax[0, c].plot(data['t1'], data[f'a{i + 1}'], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i],
                   label=f'Vehicle {i + 1}')
        # Velocity
        ax[1, c].plot(data['t1'], data[f'v{i + 1}'], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i],
                   label=f'Vehicle {i + 1}')
        # Position
        ax[2, c].plot(data['t1'], data[f'Y{i + 1}'], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i],
                   label=f'Vehicle {i + 1}')


    # Columns 2, 3, and 4 for plotting trajectories of the first four vehicles
    for c in [1, 2, 3]:
        for i in range(4):
            ax[0, c].plot(data['t1'], data[f'a{i + 1}'], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
            ax[1, c].plot(data['t1'], data[f'v{i + 1}'], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
            ax[2, c].plot(data['t1'], data[f'Y{i + 1}'], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])

    # Column 2: Acceleration, velocity, and position for the fifth vehicle
    c = 1
    ax[0, c].plot(data['t1'], data['a5_mpc_base2'], color=colors[4], linestyle=linestyles[4], linewidth=linewidths[4])
    ax[1, c].plot(data['t1'], data['v5_mpc_base2'], color=colors[4], linestyle=linestyles[4], linewidth=linewidths[4])
    ax[2, c].plot(data['t1'], data['Y5_mpc_base2'], color=colors[4], linestyle=linestyles[4], linewidth=linewidths[4])

    # Acceleration, velocity, and position for the sixth vehicle
    if 'v6_mpc_base2' in data.columns:
        n = 5
        ax[0, c].plot(data['t1'], data['a6_mpc_base2'], color=colors[n], linestyle=linestyles[n], linewidth=linewidths[n])
        ax[1, c].plot(data['t1'], data['v6_mpc_base2'], color=colors[n], linestyle=linestyles[n], linewidth=linewidths[n])
        ax[2, c].plot(data['t1'], data['Y6_mpc_base2'], color=colors[n], linestyle=linestyles[n], linewidth=linewidths[n])


    # Column 3
    if 'v5_mpc_base3' in data.columns:
        c = 2
        ax[0, c].plot(data['t1'], data['a5_mpc_base3'], color=colors[4], linestyle=linestyles[4], linewidth=linewidths[4])
        ax[1, c].plot(data['t1'], data['v5_mpc_base3'], color=colors[4], linestyle=linestyles[4], linewidth=linewidths[4])
        ax[2, c].plot(data['t1'], data['Y5_mpc_base3'], color=colors[4], linestyle=linestyles[4], linewidth=linewidths[4])
    if 'v6_mpc_base3' in data.columns:
        n = 5
        ax[0, c].plot(data['t1'], data['a6_mpc_base3'], color=colors[n], linestyle=linestyles[n],
                      linewidth=linewidths[n])
        ax[1, c].plot(data['t1'], data['v6_mpc_base3'], color=colors[n], linestyle=linestyles[n],
                      linewidth=linewidths[n])
        ax[2, c].plot(data['t1'], data['Y6_mpc_base3'], color=colors[n], linestyle=linestyles[n],
                      linewidth=linewidths[n])


    # Set y-axis labels
    for i in range(3):
        ax[i, 0].set_ylabel(['Acceleration (m/s²)', 'Velocity (m/s)', 'Position (m)'][i])
    for i in range(4):
        ax[2, i].set_xlabel('Time (s)')

    # Set y-axis limits for each column
    for j in range(4):  # Apply to each column
        ax[0, j].set_ylim(-3, 3)
        ax[1, j].set_ylim(0, 13)
        ax[2, j].set_ylim(0, 260)

    # Set column titles
    titles = ["Baseline1", "Baseline2", "Baseline3", "Proposed"]
    for i in range(4):
        ax[2, i].text(0.5, -0.5, titles[i], ha='center', va='top', transform=ax[2, i].transAxes, fontsize=12)

    # Show legend
    # ax[0, 0].legend()
    # Show the overall legend
    handles, labels = ax[0, 0].get_legend_handles_labels()  # Get legend info from the first subplot
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False)

    # Display and save the figure
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.15, top=0.95)  # Adjust to leave enough space
    plt.savefig(file_path.replace('.csv', '_results.png'))
    # plt.show()


# Load data
file_path = './data/NGSIM_I80_results/'

# Get all CSV files
files = [f for f in os.listdir(file_path) if f.endswith('.csv')]

# Loop through each file, plot and save
for file in files:
    plot_results(os.path.join(file_path, file))
