import numpy as np
from keras.models import load_model
import data as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':
    data_size = 18000
    train_x, train_y, test_x, test_y = dt.load_data(data_size)
    model = load_model("./model_new/new.h5")
    y_predict = model.predict(test_x) # error
    y_predict = y_predict.reshape(-1, 10, 1)

    # print('shape of test_x', test_x.shape)
    # print('shape of test_y_real', test_y_real.shape)
    # print('shape of test_y_predict', test_y_predict.shape)

    # n = y_predict.shape[0]
    # t_list = np.arange(n)
    # t_list = t_list.reshape(n, 1)

    t_list = test_y[:,0,1]
    print('t',t_list)

    A_error = test_y[:, 0]

    df = pd.read_csv('../Data/new_file1.csv')
    matching_rows = df.loc[df['Time'].isin(t_list)]
    A2 = matching_rows['A2'].values
    A_hat = matching_rows['A_hat'].values
    print('shape of A2', A2.shape)

    A_error_hat = y_predict[:, 0] # PINN predict residual
    A_error_hat = A_error_hat.flatten()
    print('shape of A_error_hat', A_error_hat.shape)

    A_hat_pinn = A_hat + A_error_hat #A-A_error+新predict的error
    print('shape of A_hat_pinn', A_hat_pinn.shape)

    mse = mean_squared_error(A2, A_hat_pinn)
    print('Mean Squared Error (MSE):', mse)

    plt.figure(figsize=(8, 4))
    plt.plot(t_list, A2, 'b.', label='Original a')
    plt.plot(t_list, A_hat, 'g.', label='IDM predict a')
    plt.plot(t_list, A_hat_pinn, 'r.', label='Predicted a')
    plt.xlabel('Time')
    plt.ylabel('Acceleration error (m/s^2)')
    plt.legend()
    plt.savefig('PINN_result_plot.png')
    plt.show()

