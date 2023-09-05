import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_data_from_csv(file_path, DataName):
    df = pd.read_csv(file_path)

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, figsize=(10, 10))

    axs[0].plot(df['t'], df['Y0'], label='veh0_Y', color='gray')
    axs[0].plot(df['t'], df['Y1'], label='veh1_Y', color='royalblue')
    axs[0].plot(df['t'], df['Y2'], label='veh2_Y', color='darkorange')
    axs[0].set_ylabel('Y (m)')
    axs[0].legend()

    axs[1].plot(df['t'], df['v0'], label='veh0_v', color='gray')
    axs[1].plot(df['t'], df['v1'], label='veh1_v', color='royalblue')
    axs[1].plot(df['t'], df['v2'], label='veh2_v', color='darkorange')
    axs[1].set_ylabel('v (m/s)')
    axs[1].legend()

    axs[2].plot(df['t'], df['a0'], label='veh0_a', color='gray')
    axs[2].plot(df['t'], df['a0_IDM'], label='veh0_a IDM predicted', color='firebrick', linestyle='dashed')
    axs[2].plot(df['t'], df['a0_PERL'], label='veh0_a PERL predicted', color='maroon')
    # axs[2].plot(df['t'], df['a1'], label='veh1_a', color='royalblue')
    # axs[2].plot(df['t'], df['a2'], label='veh2_a', color='darkorange')
    axs[2].set_ylabel('a (m/s²)')
    axs[2].set_xlabel('t')
    axs[2].legend()

    for ax in axs:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig_name = os.path.splitext(os.path.basename(file_path))[0]  # remove the .csv and get just the file name
    fig.savefig(f'./data/{DataName}_predicted/{fig_name}.png')
    plt.close(fig)


def plot_all_csv_in_directory(DataName):
    directory_path = f"./data/{DataName}_predicted/"
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            plot_data_from_csv(os.path.join(directory_path, file_name), DataName)


plot_all_csv_in_directory("NGSIM_US101")