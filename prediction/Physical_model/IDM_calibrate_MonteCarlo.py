import pandas as pd
import random
from IDM import IDM
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def monte_carlo_optimization(df, num_iterations):
    best_mse = 1000000000
    best_arg = None

    # 使用tqdm显示进度条
    with tqdm(total=num_iterations, desc='Iterations', postfix={'Best MSE': float('inf')}) as pbar:
        for _ in range(num_iterations):
            vf = random.uniform(20, 30)
            A = random.uniform(0.5, 3)
            b = random.uniform(0.5, 4)
            s0 = random.uniform(2, 5)
            T = random.uniform(0.5, 3)
            arg = (round(vf, 4), round(A, 4), round(b, 4), round(s0, 4), round(T, 4))

            df['A_hat'] = df.apply(lambda row: IDM(arg, row['V'], row['V'] - row['pre_V'], row['pre_Y']-row['Local_Y']),axis=1)
            df['A_error'] = df['A_hat'] - df['A']

            error = []
            for _, row in df.iterrows():
                error.append(row['A_error'] ** 2)
            mse = np.mean(error)

            if mse < best_mse:
                best_mse = mse
                best_arg = arg

            # 更新最小MSE的值
            pbar.set_postfix_str({'Best MSE': round(best_mse, 3), 'best_arg': best_arg})
            pbar.update(1)

    # plt.hist(df['A_error'], bins=20, color='blue', alpha=0.5)
    # plt.title('A_error Distribution')
    return best_arg, best_mse



file_names = ["../Data/NGSIM/i-80_1_trj_for_calibration.csv",
              "../Data/NGSIM/i-80_2_trj_for_calibration.csv",
              "../Data/NGSIM/i-80_3_trj_for_calibration.csv",
              "../Data/NGSIM/i-80_4_trj_for_calibration.csv"]  # 根据实际文件名进行调整
dataframes = []
for file_name in file_names:
    d = pd.read_csv(file_name)
    dataframes.append(d)
df = pd.concat(dataframes, ignore_index=True)
print('Before filtering len(df)=', len(df))

# 筛选
df = df[df['Preceding'] != 0]
df = df[df['Space_Headway'] > 5] # 这个阈值直接决定了标定结果
df = df.dropna(subset=['V', 'pre_V', 'Space_Headway'])
print('After filtering  len(df)=', len(df))

num_iterations = 50  # 设置迭代次数
best_arg, best_mse = monte_carlo_optimization(df, num_iterations)


