from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
sys.path.append('/home/ubuntu/Documents/PINN/IDM_NN')
from IDM_NN import data as dt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platoon_num', type=int, help='Platoon number')
    args = parser.parse_args()
    platoon_num = args.platoon_num
    platoon_num = 20

    look_back = 10

    # 准备数据
    _, test_x, _, test_y, test_rows, A_error_min, A_error_max = dt.prepare_data(platoon_num,look_back)

    # 加载模型
    model = load_model("./model/platoon{}.h5".format(platoon_num))

    # 在测试集上进行预测
    A_error_hat = model.predict(test_x)

    # 反归一化
    A_error_hat = A_error_hat * (A_error_max - A_error_min) + A_error_min
    A_error_hat = A_error_hat.flatten()

    # 获取对应行的A_hat值
    df = pd.read_csv('../Data/ASta_platoon{}_new1.csv'.format(platoon_num))
    A_hat = df.loc[test_rows, 'A_hat'].values
    A2 = df.loc[test_rows, 'A2'].values

    # 计算A_pinn
    A_pinn = A_hat - A_error_hat

    # 计算MSE
    mse = mean_squared_error(A2, A_pinn)
    print('MSE when predicting acceleration:', mse)

    plt.figure(figsize=(10, 4))
    plt.plot(test_rows, A2, '.', color='b', markersize=1, label='Original a')
    plt.plot(test_rows, A_hat, '.', color='g', markersize=0.5, label='IDM predict a')
    plt.plot(test_rows, A_pinn, '.', color='r', markersize=1, label='PINN Predicted a')
    plt.xlabel('index')
    plt.ylabel('Acceleration error (m/s^2)')
    plt.ylim(-2, 2)
    plt.title('MSE: {:.4f}'.format(mse))
    plt.legend()
    plt.savefig('../Results/Platoon{}_PINN_result.png'.format(platoon_num))
    plt.show()

