import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

# def IDM(arg, vi, delta_v, delta_d):
#     vf, A, b, s0, T = arg
#     s_star = s0 + np.max([0, vi*T + (vi * delta_v) / (2 * (A*b) ** 0.5)], axis=0)
#     #print('A*b',A*b)
#     epsilon = 1e-5
#     ahat = A*(1 - (vi/vf)**4 - (s_star/(delta_d+epsilon))**2)
#     return ahat


def plot_results(origin_trj, mpc_trj, DataName, filename, scenario, n):
    ts = 0.1
    t = np.arange(0, ts * n, ts)
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    axs[0].plot(t, origin_trj.a_0_origin, label='Veh0_origin', color='gray', linestyle='dashed')
    axs[0].plot(t, origin_trj.a_1_origin, label='Veh1_origin', color='royalblue', linestyle='dashed')
    axs[0].plot(t, origin_trj.a_2_origin, label='Veh2_origin', color='darkorange', linestyle='dashed')
    axs[0].plot(t, mpc_trj.a_1, label='Veh1_new', color='royalblue', marker='o', markersize=2, linewidth=1)
    axs[0].plot(t, mpc_trj.a_2, label='Veh2_new', color='darkorange', marker='o', markersize=2, linewidth=1)
    axs[0].plot(t, origin_trj.a_0_IDM, label='Veh0_IDM', color='black', linestyle='dashed')
    axs[0].set_ylim([-3, 3])
    axs[0].set_xlabel('Sampling time T')
    axs[0].set_ylabel('Acceleration (m/s²)')
    axs[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    axs[0].grid(True)

    axs[1].plot(t, origin_trj.v_0_origin, label='Veh0_origin', color='gray', linestyle='dashed')
    axs[1].plot(t, origin_trj.v_1_origin, label='Veh1_origin', color='royalblue', linestyle='dashed')
    axs[1].plot(t, origin_trj.v_2_origin, label='Veh2_origin', color='darkorange', linestyle='dashed')
    axs[1].plot(t, mpc_trj.v_1, label='Veh1_new', color='royalblue', marker='o', markersize=2, linewidth=1)
    axs[1].plot(t, mpc_trj.v_2, label='Veh2_new', color='darkorange', marker='o', markersize=2, linewidth=1)
    axs[1].set_xlabel('Sampling time T')
    axs[1].set_ylabel('Speed (m/s)')
    axs[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    axs[1].grid(True)

    axs[2].plot(t, origin_trj.x_0_origin, label='Veh0_origin', color='gray', linestyle='dashed')
    axs[2].plot(t, origin_trj.x_1_origin, label='Veh1_origin', color='royalblue', linestyle='dashed')
    axs[2].plot(t, origin_trj.x_2_origin, label='Veh2_origin', color='darkorange', linestyle='dashed')
    axs[2].plot(t, mpc_trj.x_1, label='Veh1_new', color='royalblue', marker='o', markersize=1.5, linewidth=1)
    axs[2].plot(t, mpc_trj.x_2, label='Veh2_new', color='darkorange', marker='o', markersize=1.5, linewidth=1)
    axs[2].set_xlabel('Sampling time T')
    axs[2].set_ylabel('Position (m)')
    axs[2].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    axs[2].grid(True)

    axs[0].set_xlim([0, 20])
    axs[1].set_xlim([0, 20])
    axs[2].set_xlim([0, 20])

    plt.tight_layout()
    plt.savefig(f'./results/{DataName}/{filename}_{scenario}_a_v_x.png', dpi=300)
    #plt.show()


import pandas as pd
def save_trj(origin_trj, mpc_trj, DataName, filename, scenario):
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
    origin_df.to_csv(f'./results/{DataName}/{filename}_origin_trj.csv', index=False)
    mpc_df.to_csv(f'./results/{DataName}/{filename}_{scenario}_mpc_trj.csv', index=False)
    print(f'Save results to ./results/{DataName}/{filename}_{scenario}_mpc_trj.csv')


def save_measures(origin_trj, mpc_trj, DataName, filename, scenario):
    measurements = pd.DataFrame(columns=['veh0_veh1_distance_mean', 'veh0_veh1_distance_var',
                                         'veh1_veh2_distance_mean', 'veh1_veh2_distance_var',
                                         'veh1_a_mean', 'veh1_a_var',
                                         'veh2_a_mean', 'veh2_a_var',
                                         'veh1_v_mean', 'veh1_v_var',
                                         'veh2_v_mean', 'veh2_v_var'])
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
    # 计算速度的均值和方差
    m9 = np.mean(origin_trj.v_1_origin)
    m10= np.var(origin_trj.v_1_origin)
    m11= np.mean(origin_trj.v_2_origin)
    m12= np.var(origin_trj.v_2_origin)
    measurements.loc[0] = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12]

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
    # 计算速度的均值和方差
    m9 = np.mean(mpc_trj.v_1)
    m10 = np.var(mpc_trj.v_1)
    m11 = np.mean(mpc_trj.v_2)
    m12 = np.var(mpc_trj.v_2)
    measurements.loc[1] = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12]

    # 保存数据帧到CSV文件
    measurements.to_csv(f'./results/{DataName}/{filename}_{scenario}_measure.csv', index=False)