import numpy as np
from keras.models import load_model
import data as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd


if __name__ == '__main__':
    data_size = 18000
    train_x, train_y, test_x, test_y_real = dt.load_data(data_size)
    model = load_model("./model_new/new.h5")
    test_y_predict = model.predict(test_x)
    test_y_predict = test_y_predict.reshape(-1, 10, 1)
    # print('shape of test_x', test_x.shape)
    # print('shape of test_y_real', test_y_real.shape)
    # print('shape of test_y_predict', test_y_predict.shape)

    n = test_y_predict.shape[0]
    t = np.arange(n)
    t = t.reshape(n, 1)

    mse = mean_squared_error(test_y_real[:,0], test_y_predict[:,0])
    print('Mean Squared Error (MSE):', mse)


    plt.figure(figsize=(8, 4))
    plt.plot(t, test_y_real[:,0], 'b.', label='Original a')
    plt.plot(t, test_y_predict[:,0], 'r.', label='Predicted a')
    plt.xlabel('Time')
    plt.ylabel('Acceleration error (m/s^2)')
    plt.legend()
    plt.savefig('LSTM_result.png')
    plt.show()


