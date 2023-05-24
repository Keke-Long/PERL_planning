import numpy as np
from keras.models import load_model
import data as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platoon_num', type=int, help='Platoon number')
    args = parser.parse_args()
    platoon_num = args.platoon_num

    train_x, train_y, test_x, test_y, A_error_min, A_error_max = dt.load_data(platoon_num)
    model = load_model("./model/platoon{}.h5".format(platoon_num))
    y_predict = model.predict(test_x) # error
    y_predict = y_predict.reshape(-1, 10, 1)

    index_list = test_y[:,0,1]
    print('len(t_list)', len(index_list))
    #A_error = test_y[:, 0]

    df = pd.read_csv('../Data/ASta_platoon{}_new1.csv'.format(platoon_num))
    matching_rows = df.loc[df.index.isin(index_list)]
    A2 = matching_rows['A2'].values
    A_hat = matching_rows['A_hat'].values

    A_error_hat = y_predict[:, 0] # PINN predict residual
    A_error_hat = A_error_hat.flatten()
    A_error_hat = A_error_hat * (A_error_max - A_error_min) + A_error_min

    A_pinn = A_hat - A_error_hat #A2+A_error = A_hat; A_hat-PINN预测的error=A_pinn

    mse = mean_squared_error(A2, A_pinn)
    print('MSE when predicting acceleration:', mse)

    plt.figure(figsize=(10, 4))
    plt.plot(index_list, A2, '.', color='b', markersize=1, label='Original a')
    plt.plot(index_list, A_hat, '.', color='g', markersize=0.5, label='IDM predict a')
    plt.plot(index_list, A_pinn, '.', color='r', markersize=1, label='PINN Predicted a')
    plt.xlabel('index')
    plt.ylabel('Acceleration error (m/s^2)')
    plt.ylim(-2, 2)
    plt.title('MSE: {:.4f}'.format(mse))
    plt.legend()
    plt.savefig('Platoon{}_PINN_result.png'.format(platoon_num))
    plt.show()

