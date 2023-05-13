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
    train_x, train_y, test_x, test_y_real = dt.load_data(data_size)
    model = load_model("./model_new/new.h5")
    test_y_predict = model.predict(test_x)
    del model
    test_y_predict = test_y_predict.reshape(-1, 10, 1)
    # print('shape of test_x', test_x.shape)
    # print('shape of test_y_real', test_y_real.shape)
    # print('shape of test_y_predict', test_y_predict.shape)

    n = test_y_predict.shape[0]
    t = np.arange(n)
    t = t.reshape(n, 1)

    data_csv = pd.read_csv('../Data/new_file1.csv')
    A2 = data_csv['A2'].tolist()
    A2 = A2[50:50+data_size]
    A2 = A2[12585:]
    A2 = A2[:5394]
    A_hat = data_csv['A_hat'].tolist()
    A_hat = A_hat[50: 50 + data_size]
    A_hat = A_hat[12585:] # physical model predict result
    A_hat = A_hat[:5394]
    del data_csv
    A_hat = np.array(A_hat).reshape(-1, 1)
    print('shape of A_hat', A_hat.shape)

    A_error_hat = test_y_predict[:, 0] # PINN predict residual
    print('shape of A_error_hat', A_error_hat.shape)
    A_hat_pinn = np.subtract(A_hat, A_error_hat)

    mse = mean_squared_error(A2, A_hat_pinn)
    print('Mean Squared Error (MSE):', mse)

    plt.figure(figsize=(8, 4))
    plt.plot(t, test_y_real[:, 0], 'b.', label='Original a')
    plt.plot(t, A_hat_pinn, 'r.', label='Predicted a')
    plt.xlabel('Time')
    plt.ylabel('Acceleration error (m/s^2)')
    plt.legend()
    plt.savefig('PINN_result_plot.png')
    plt.show()

