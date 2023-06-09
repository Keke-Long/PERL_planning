from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
import argparse
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/ubuntu/Documents/PINN/IDM_NN')
from IDM_NN import data as dt


def train_model(train_x, train_y, epochs, batch_size, dropout=0.2):
    model = Sequential()
    model.add(LSTM(256,
                   input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(train_y.shape[1]))  # 10个输出的全连接层
    model.add(Activation("relu"))

    model.compile(loss='mse', optimizer='adam')
    loss_history = []  # 记录损失函数值的列表

    n = 5
    threshold = 0.0001
    for epoch in range(epochs):
        history = model.fit(train_x, train_y, epochs=1, batch_size=batch_size, verbose=1)
        loss = history.history['loss'][0]
        loss_history.append(loss)

        print("Epoch: {}/{} - Loss: {:.6f}".format(epoch + 1, epochs, loss))

        # 判断收敛条件，例如损失函数值连续n个轮次变化小于阈值
        if len(loss_history) > n and abs(loss - loss_history[-n - 1]) < threshold:
            print("Converged.")
            break

    # 保存convergence rate数据
    np.savetxt("convergence_rate.csv", loss_history, delimiter=",")

    # 可视化损失函数值的变化
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('../Results/Training Loss of MLP.png')
    plt.show()

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platoon_num', type=int, help='Platoon number')
    args = parser.parse_args()
    platoon_num = args.platoon_num
    platoon_num = 40

    look_back = 10

    train_x, _, train_y, _, _, _, _ = dt.prepare_data(platoon_num, look_back)
    print("Shape of train_x:", train_x.shape)
    print("Shape of train_y:", train_y.shape)
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
    train_y = train_y.reshape(train_y.shape[0], 1)

    model = train_model(train_x, train_y, epochs=100, batch_size=64, dropout=0.05)
    model.save("./model/platoon{}.h5".format(platoon_num))

