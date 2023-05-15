import numpy as np
from keras.models import load_model
import data as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == '__main__':
    data_size = 15500
    train_x, train_y, test_x, test_y = dt.load_data(data_size)
    model = load_model("./model/platoon1.h5")
    y_predict = model.predict(test_x) # error
    y_predict = y_predict.reshape(-1, 10, 1)

    t_list = test_y[:,0,1]
    #A_error = test_y[:, 0]

    df = pd.read_csv('../Data/ASta_050719_platoon1_new1.csv')
    matching_rows = df.loc[df['Time'].isin(t_list)]
    A2 = matching_rows['A2'].values
    A_hat = matching_rows['A_hat'].values
    A_error = matching_rows['A_error'].values
    A_error_hat = y_predict[:, 0] # PINN predict residual
    A_error_hat = A_error_hat.flatten()

    A_hat_pinn = A_hat - A_error_hat #A-A_error+新predict的error

    print('MSE when predicting A_error:', mean_squared_error(A_error, A_error_hat))
    print('MSE when predicting acceleration:', mean_squared_error(A2, A_hat_pinn))

    plt.figure(figsize=(8, 4))
    plt.plot(t_list, A2, 'b.', label='Original a')
    plt.plot(t_list, A_hat, 'g.', label='IDM predict a')
    plt.plot(t_list, A_hat_pinn, 'r.', label='PINN Predicted a')
    plt.xlabel('Time')
    plt.ylabel('Acceleration error (m/s^2)')
    plt.ylim(-2, 2)
    plt.legend()
    plt.savefig('Platoon1 PINN_result_plot.png')
    plt.show()

