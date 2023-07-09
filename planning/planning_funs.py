import numpy as np
import matplotlib.pyplot as plt
import os

def plot_results(origin_trj, mpc_trj, filename, scenario, n):
    ts = 0.1
    t = np.arange(0, ts * n, ts)
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    axs[0].plot(t, origin_trj.a_0_origin, label='Veh0', color='gray')
    axs[0].plot(t, mpc_trj.a_1, label='Veh1_new', color='royalblue', linestyle='solid')
    axs[0].plot(t, mpc_trj.a_2, label='Veh2_new', color='darkorange', linestyle='solid')
    axs[0].plot(t, origin_trj.a_1_origin, label='Veh1_origin', color='royalblue', linestyle='dashed')
    axs[0].plot(t, origin_trj.a_2_origin, label='Veh2_origin', color='darkorange', linestyle='dashed')
    axs[0].set_ylim([-3, 3])
    axs[0].set_xlabel('Sampling time T')
    axs[0].set_ylabel('Acceleration (m/s²)')
    axs[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    axs[0].grid(True)

    axs[1].plot(t, origin_trj.v_0_origin, label='Veh0', color='gray')
    axs[1].plot(t, mpc_trj.v_1, label='Veh1_new', color='royalblue', linestyle='solid')
    axs[1].plot(t, mpc_trj.v_2, label='Veh2_new', color='darkorange', linestyle='solid')
    axs[1].plot(t, origin_trj.v_1_origin, label='Veh1_origin', color='royalblue', linestyle='dashed')
    axs[1].plot(t, origin_trj.v_2_origin, label='Veh2_origin', color='darkorange', linestyle='dashed')
    axs[1].set_xlabel('Sampling time T')
    axs[1].set_ylabel('Speed (m/s)')
    axs[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    axs[1].grid(True)

    axs[2].plot(t, origin_trj.x_0_origin, label='Veh0', color='gray')
    axs[2].plot(t, mpc_trj.x_1, label='Veh1_new', color='royalblue', linestyle='solid')
    axs[2].plot(t, mpc_trj.x_2, label='Veh2_new', color='darkorange', linestyle='solid')
    axs[2].plot(t, origin_trj.x_1_origin, label='Veh1_origin', color='royalblue', linestyle='dashed')
    axs[2].plot(t, origin_trj.x_2_origin, label='Veh2_origin', color='darkorange', linestyle='dashed')
    axs[2].set_xlabel('Sampling time T')
    axs[2].set_ylabel('Position (m)')
    axs[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/Documents/PINN/planning/Results/{filename}_{scenario}_a_v_x.png', dpi=300)
    #plt.show()


import pandas as pd
def save_trj(origin_trj, mpc_trj, filename, scenario):
    # Convert instances to DataFrames

    origin_df = pd.DataFrame({
        't': origin_trj.t,
        'a_0_origin': origin_trj.a_0_origin,
        'a_1_origin': origin_trj.a_1_origin,
        'a_2_origin': origin_trj.a_2_origin,
        'v_0_origin': origin_trj.v_0_origin,
        'v_1_origin': origin_trj.v_1_origin,
        'v_2_origin': origin_trj.v_2_origin,
        'x_0_origin': origin_trj.x_0_origin,
        'x_1_origin': origin_trj.x_1_origin,
        'x_2_origin': origin_trj.x_2_origin
    })

    mpc_df = pd.DataFrame({
        't': mpc_trj.t,
        'a_1': mpc_trj.a_1.flatten(),
        'a_2': mpc_trj.a_2.flatten(),
        'v_1': mpc_trj.v_1,
        'v_2': mpc_trj.v_2,
        'x_1': mpc_trj.x_1,
        'x_2': mpc_trj.x_2
    })

    # Save DataFrames to CSV
    origin_df.to_csv(f'/home/ubuntu/Documents/PINN/planning/Results/{filename}_origin_trj.csv', index=False)
    mpc_df.to_csv(f'/home/ubuntu/Documents/PINN/planning/Results/{filename}_{scenario}_mpc_trj.csv', index=False)
    current_dir = os.getcwd()
    print('Save results to', current_dir)


def save_measures(origin_trj, mpc_trj, filename, scenario):
    measurements = pd.DataFrame(columns=['veh01_distance_mean', 'veh01_distance_var',
                                         'veh12_distance_mean', 'veh12_distance_var',
                                         'veh1_a_mean', 'veh1_a_var',
                                         'veh2_a_mean', 'veh2_a_var',])
    # 计算车辆间距的均值和方差
    m1 = np.mean(origin_trj.x_1_origin - origin_trj.x_0_origin)
    m2 = np.var(origin_trj.x_1_origin - origin_trj.x_0_origin)
    m3 = np.mean(origin_trj.x_2_origin - origin_trj.x_1_origin)
    m4 = np.var(origin_trj.x_2_origin - origin_trj.x_1_origin)
    # 计算加速度的均值和方差
    m5 = np.mean(origin_trj.a_1_origin)
    m6 = np.var(origin_trj.a_1_origin)
    m7 = np.mean(origin_trj.a_2_origin)
    m8 = np.var(origin_trj.a_2_origin)
    measurements.loc[0] = [m1, m2, m3, m4, m5, m6, m7, m8]

    # 计算车辆间距的均值和方差
    m1 = np.mean(mpc_trj.x_1 - origin_trj.x_0_origin)
    m2 = np.var(mpc_trj.x_1 - origin_trj.x_0_origin)
    m3 = np.mean(mpc_trj.x_2 - mpc_trj.x_1)
    m4 = np.var(mpc_trj.x_2 - mpc_trj.x_1)
    # 计算加速度的均值和方差
    m5 = np.mean(mpc_trj.a_1)
    m6 = np.var(mpc_trj.a_1)
    m7 = np.mean(mpc_trj.a_2)
    m8 = np.var(mpc_trj.a_2)
    measurements.loc[1] = [m1, m2, m3, m4, m5, m6, m7, m8]

    # 保存数据帧到CSV文件
    measurements.to_csv(f'/home/ubuntu/Documents/PINN/planning/Results/{filename}_{scenario}_measure.csv', index=False)