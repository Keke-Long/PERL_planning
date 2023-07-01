import numpy as np
from keras.models import load_model
import data as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    # 准备数据
    _, test_x, _, _, test_rows, A_error_min, A_error_max, A2_test_list, A_hat_test_list = dt.prepare_data(look_back = 50, look_forward = 30)
    # 加载模型
    model = load_model("./model/NGSim.h5")
    print(model.input_shape)
    print(model.output_shape)

    # 在测试集上进行预测
    A_error_hat = model.predict(test_x) #为什么只预测了一个值

    # 反归一化
    A_error_hat = A_error_hat * (A_error_max - A_error_min) + A_error_min

    # 获取对应行的A_hat值
    A_hat = A_hat_test_list
    A2 = A2_test_list

    # 计算A_pinn
    A_pinn = A_hat + A_error_hat

    # 计算MSE
    mse = mean_squared_error(A2, A_pinn)
    print('MSE when predicting all acceleration:', mse)
    mse2 = mean_squared_error(A2[:][0], A_pinn[:][0])
    print('MSE when predicting first acceleration:', mse2)

    # # 绘制预测结果和真实值的图形
    # plt.figure(figsize=(10, 4))
    # for i in range(A_pinn.shape[0]):
    #     x = range(len(A_pinn[i]))
    #     plt.plot(x, A2[i],  color='b', markersize=0.5, label='Original a')
    #     plt.plot(x, A_hat[i], color='g', markersize=0.2, label='IDM predict a')
    #     plt.plot(x, A_pinn[i], color='r', markersize=0.5, label='PINN(IDM+NN) Predicted a')
    # plt.xlabel('Index')
    # plt.ylabel('Acceleration (m/s^2)')
    # plt.ylim(-2, 2)
    # plt.title('MSE: {:.4f}'.format(mse))
    # plt.legend()
    # plt.savefig('../Results/NGSIM/PINN(IDM+NN)_result.png')
    # plt.show()
