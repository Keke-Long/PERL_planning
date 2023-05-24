from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Embedding, LSTM, Dropout, Activation
import data as dt
import os
import argparse

def train_model(train_x, train_y, epochs, batch_size, dropout=0.2):
    model = Sequential()
    model.add(LSTM(256,
                   input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(train_y.shape[1]))  # 10个输出的全连接层
    model.add(Activation("relu"))

    model.compile(loss='mse', optimizer='adam')
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model

    # model = Sequential()
    # model.add(LSTM(256,
    #                input_shape=(train_x.shape[1], train_x.shape[2]),
    #                return_sequences=True))
    # model.add(Dropout(dropout))
    #
    # model.add(LSTM(256,
    #                return_sequences=False))
    # model.add(Dropout(dropout))
    #
    # model.add(Dense(train_y.shape[1]))  # 10个输出的全连接层
    # model.add(Activation("relu"))
    #
    # model.compile(loss='mse', optimizer='adam')
    # model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
    #
    # return model


if __name__ == '__main__':
    """
    根据前10步预测当前时刻后10步的轨迹
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--platoon_num', type=int, help='Platoon number')
    args = parser.parse_args()
    platoon_num = args.platoon_num

    train_x, train_y, test_x, test_y, _, _ = dt.load_data(platoon_num)

    epochs = 100
    batch_size = 64
    dropout = 0.05
    model = train_model(train_x, train_y[:,:,0], epochs, batch_size, dropout)
    model_name = "./model/platoon{}.h5".format(platoon_num)
    model.save(model_name)

