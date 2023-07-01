import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
import data as dt
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 准备数据
    train_x, _, train_y, _, _, _, _, _, _ = dt.prepare_data(look_back = 50, look_forward = 30)
    print("Shape of train_x:", train_x.shape)
    print("Shape of train_y:", train_y.shape)

    # 构建 MLP 模型
    model = Sequential([
        Dense(64, activation='relu', input_shape=(train_x.shape[1],)),
        Dense(64, activation='relu'),
        Dense(train_y.shape[1])
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(model.input_shape)
    print(model.output_shape)

    # 定义收敛判断的阈值和记录历史损失值
    threshold = 0.0001
    loss_history = []

    # 训练模型，并记录每个epoch的损失值
    for epoch in range(100):
        history = model.fit(train_x, train_y, epochs=1, batch_size=64, verbose=1)
        loss = history.history['loss'][0]
        loss_history.append(loss)

        # 判断收敛
        if len(loss_history) > 2 and abs(loss - loss_history[-2]) < threshold:
            print("Convergence achieved at epoch", epoch + 1)
            break

    model.save("./model/NGSim.h5")
    np.savetxt("convergence_rate.csv", loss_history, delimiter=",")

    # 可视化convergence rate
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Convergence Rate')
    plt.savefig('../Results/NGSIM/Training Loss of IDM MLP.png')
    plt.show()
