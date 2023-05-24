import numpy as np
from keras.models import load_model
import data as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, A_error_min, A_error_max = dt.load_data()
    model = load_model("./model/platoon3.h5")
    y_predict = model.predict(test_x) # error
    y_predict = y_predict.reshape(-1, 10, 1)

    index_list = test_y[:,0,1]
    print('len(t_list)', len(index_list))
    #A_error = test_y[:, 0]

    df = pd.read_csv('../Data/ASta_platoon3_new1.csv')
    matching_rows = df.loc[df.index.isin(index_list)]
    A2 = matching_rows['A2'].values
    A_hat = matching_rows['A_hat'].values

    A_error_hat = y_predict[:, 0] # PINN predict residual
    A_error_hat = A_error_hat.flatten()
    A_error_hat = A_error_hat * (A_error_max - A_error_min) + A_error_min

    A_pinn = A_hat - A_error_hat #A-A_error+新predict的error

    print('MSE when predicting acceleration:', mean_squared_error(A2, A_pinn))

    plt.figure(figsize=(8, 4))
    plt.plot(index_list, A2, '.', color='b', markersize=1, label='Original a')
    plt.plot(index_list, A_hat, '.', color='g', markersize=1, label='IDM predict a')
    plt.plot(index_list, A_pinn, '.', color='r', markersize=1, label='PINN Predicted a')
    plt.xlabel('index')
    plt.ylabel('Acceleration error (m/s^2)')
    plt.ylim(-2, 2)
    plt.legend()
    plt.savefig('Platoon3_PINN_result_plot.png')
    plt.show()

