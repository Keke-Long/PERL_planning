import pandas as pd
import random
from IDM import IDM
import numpy as np
from tqdm import tqdm

def monte_carlo_optimization(df, num_iterations):
    best_mse = 100000
    best_arg = None

    # 使用tqdm显示进度条
    with tqdm(total=num_iterations, desc='Iterations', postfix={'Best MSE': float('inf')}) as pbar:
        for _ in range(num_iterations):
            vf = random.uniform(27, 30)
            A = random.uniform(0.4, 0.7)
            b = random.uniform(2, 5)
            s0 = random.uniform(2, 5)
            T = random.uniform(0.1, 2)
            arg = (round(vf,4), round(A,4), round(b,4), round(s0,4), round(T,4))

            df['A_hat'] = df.apply(lambda row: IDM(arg, row['Speed2'], row['Speed2'] - row['Speed1'], row['IVS1']), axis=1)
            df['V_hat'] = df['Speed2'] + df['A_hat'] * 0.1
            df['A_error'] = df['A_hat'] - df['A2']

            error = []
            for _, row in df.iterrows():
                error.append(row['A_error'] ** 2)
            mse = np.mean(error)

            if mse < best_mse:
                best_mse = mse
                best_arg = arg

            # 更新最小MSE的值
            pbar.set_postfix_str({'Best MSE': round(best_mse,3), 'best_arg': best_arg})
            pbar.update(1)

    return best_arg, best_mse

df = pd.read_csv("../Data/ASta_platoon20_new.csv")
train_length = int(0.7 * len(df))  # 计算训练集的长度
train_df = df[:train_length]
num_iterations = 1000  # 设置迭代次数
best_arg, best_mse = monte_carlo_optimization(train_df, num_iterations)
print('Best arg:', best_arg)
print('Best MSE:', best_mse)


