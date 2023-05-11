# 给定车辆id， 画图，预测加速度和实际加速度，用
import matplotlib.pyplot as plt
import pandas as pd
from IDM import IDM

def plot_comparison(d, save_file=None):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # 绘制速度对比图
    axs[0].plot(d['Time'], d['Speed2'], label='Real v')
    axs[0].plot(d['Time'], d['vhat'], label='Predicted v')
    axs[0].set_xlabel('Time (seconds)')
    axs[0].set_ylabel('Speed (m/s)')
    axs[0].legend()
    axs[0].set_title('Vehicle Speed Comparison')

    # 绘制加速度对比图
    axs[1].plot(d['Time'], d['A2'], label='Real a')
    axs[1].plot(d['Time'], d['ahat'], label='Predicted a')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Acceleration (m/s^2)')
    axs[1].legend()
    axs[1].set_title('Vehicle Acceleration Comparison')

    plt.subplots_adjust(hspace=0.4)

    if save_file:
        plt.savefig('IDM_result_comparison.jpg')
    plt.show()


df = pd.read_csv("../Data/new_file.csv")
arg = (47.68326931, 2., 3., 9.91649325, 0.59416805)# vf, A, b, s0, T
df['A_hat'] = df.apply(lambda row: IDM(arg, row['Speed2'], row['Speed2'] - row['Speed1'], row['IVS1']), axis=1)
df['V_hat'] = df['Speed2'] + df['A_hat'] * 0.1
df['A_error'] = df['A_hat'] - df['A2']
df.to_csv('../Data/new_file.csv', index=False)

plot_comparison(df, save_file=True)
