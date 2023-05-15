# 给定车辆id， 画图，预测加速度和实际加速度，用
import matplotlib.pyplot as plt
import pandas as pd
from IDM import IDM
from sklearn.metrics import mean_squared_error
import numpy as np

def plot_comparison(d, save_file):

    # 绘制加速度对比图
    plt.figure(figsize=(10, 4))
    plt.plot(d['Time'], d['A2'], label='Real a')
    plt.plot(d['Time'], d['A_hat'], label='Predicted a')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.ylim(-2, 2)
    plt.legend()
    plt.title('Vehicle Acceleration Prediction Result from IDM model')

    plt.savefig(save_file + '.jpg')
    plt.show()


df = pd.read_csv("../Data/ASta_050719_platoon1_new.csv")
df = df[1:]
# vf, A, b, s0, T
#arg = (25.4,   1.3,  4.0,  3.9,  1.57) # platoon 3
arg = (23, 0.51, 4, 3.5, 0.56) # platoon 1

df['A_hat'] = df.apply(lambda row: IDM(arg, row['Speed2'], row['Speed2'] - row['Speed1'], row['IVS1']), axis=1)
df['V_hat'] = df['Speed2'] + df['A_hat'] * 0.1
df['A_error'] = df['A_hat'] - df['A2']

df.to_csv('../Data/ASta_050719_platoon1_new1.csv', index=False)

# df.fillna(df.mean(), inplace=True)
# mse = mean_squared_error(df['A2'], df['A_hat'])

error = []
for index, row in df.iterrows():
    error.append(row['A_error']**2)
mse = np.mean(error)

print('MSE when predicting acceleration:', mse)

plot_comparison(df, 'Platoon1_IDM_result_comparison')