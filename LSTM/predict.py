import numpy as np
from keras.models import load_model
import data as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd


if __name__ == '__main__':
    train_x, train_y, test_x, test_y_real, A_min, A_max = dt.load_data()
    model = load_model("./model/platoon3.h5")
    test_y_predict = model.predict(test_x)
    test_y_predict = test_y_predict.reshape(-1, 10, 1)

    A_real = test_y_real[:, 0]
    A_hat  = test_y_predict[:,0]
    A_real = A_real * (A_max - A_min) + A_min
    A_hat  = A_hat * (A_max - A_min) + A_min


    mse = mean_squared_error(A_real, A_hat)
    print('MSE when predicting acceleration:', mse)

    n = test_y_predict.shape[0]
    t = np.arange(n)
    t = t.reshape(n, 1)

    plt.figure(figsize=(10, 4))
    plt.plot(t, A_real, '.', color='b', markersize=1, label='Original a')
    plt.plot(t, A_hat, '.', color='r', markersize=1, label='LSTM Predicted a')
    plt.xlabel('Index')
    plt.ylabel('Acceleration error (m/s^2)')
    plt.legend()
    plt.ylim(-2, 2)
    plt.savefig('Platoon3_LSTM_result.png')
    plt.show()


