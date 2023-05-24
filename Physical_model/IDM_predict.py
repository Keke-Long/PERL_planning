# 给定车辆id， 画图，预测加速度和实际加速度，用
import matplotlib.pyplot as plt
import pandas as pd
from IDM import IDM
import numpy as np


platoon_num = 1

df = pd.read_csv("../Data/ASta_platoon{}_new.csv".format(platoon_num))

# vf, A, b, s0, T
arg_mapping = {
    1: (28.7350, 0.5668, 3.8063, 4.0139, 0.7252),
    3: (28.5400, 0.5123, 2.6502, 2.4869, 1.5042),
    10: (28.1857, 0.5491, 3.9198, 3.7268, 1.0057),
    20: (29.2005, 0.4253, 4.0378, 2.8185, 1.5558)
}
arg = arg_mapping.get(platoon_num)

df['A_hat'] = df.apply(lambda row: IDM(arg, row['Speed2'], row['Speed2'] - row['Speed1'], row['IVS1']), axis=1)
df['V_hat'] = df['Speed2'] + df['A_hat'] * 0.1
df['A_error'] = df['A_hat'] - df['A2']

df.to_csv("../Data/ASta_platoon{}_new1.csv".format(platoon_num), index=False)

error = []
for index, row in df.iterrows():
    error.append(row['A_error']**2)
mse = np.mean(error)
print('MSE when predicting acceleration:', mse)


test_length = int(0.3 * len(df))
d = df[-test_length:]
plt.figure(figsize=(10, 4))
plt.plot(d['Time'], d['A2'], label='Real a')
plt.plot(d['Time'], d['A_hat'], label='Predicted a')
plt.xlabel('Time (seconds)')
plt.ylabel('Acceleration (m/s^2)')
plt.ylim(-2, 2)
plt.title('MSE: {:.4f}'.format(mse))
plt.legend()
plt.savefig('Platoon{}_IDM_result.jpg'.format(platoon_num))
plt.show()