import numpy as np
from keras.models import load_model
import data as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platoon_num', type=int, help='Platoon number')
    args = parser.parse_args()
    platoon_num = args.platoon_num

    train_x, train_y, test_x, test_y_real, A_min, A_max = dt.load_data(platoon_num)
    model = load_model("./model/platoon{}.h5".format(platoon_num))
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
    plt.ylim(-2, 2)
    plt.title('MSE: {:.4f}'.format(mse))
    plt.legend()
    plt.savefig('Platoon{}_LSTM_result.png'.format(platoon_num))
    plt.show()


