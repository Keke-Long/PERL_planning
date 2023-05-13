import pandas as pd
import numpy as np
import os

def create_dateset(info, look_back=10):
    dim = 10
    dataX, dataY = [], []
    for i in range(len(info) - 2 * look_back - 1):
        a = info[i: (i+look_back), 0:6] # train info
        dataX.append(a)
        b = info[(i+look_back):(i + 2 * look_back), 6] # test info
        dataY.append(b)
    dataX = np.array(dataX, dtype='float64')
    dataY = np.array(dataY, dtype='float64')
    # print("dataX shape:", dataX.shape)
    # print("dataY shape:", dataY.shape)
    return dataX, dataY

def normalization(x, data_size):
    scalar_x = np.max(x) - np.min(x)
    x = ((x - np.min(x)) / scalar_x) + 0.0000001
    x = x.reshape(data_size, 1)
    return x

def load_data(data_size):
    data_csv = pd.read_csv('../Data/new_file1.csv')

    # 取前data_size个数据作为预测对象
    pre_x = data_csv['E1'].tolist()[50:50+data_size]
    pre_y = data_csv['N1'].tolist()[50:50+data_size]
    sub_x = data_csv['E2'].tolist()[50:50+data_size]
    sub_y = data_csv['N2'].tolist()[50:50+data_size]
    sub_a = data_csv['A2'].tolist()[50:50 + data_size]
    a_error = data_csv['A_error'].tolist()[50:50+data_size]
    del data_csv
    seq = np.arange(data_size) # 序号

    # Normalization
    pre_x = normalization(pre_x, data_size)
    pre_y = normalization(pre_y, data_size)
    sub_x = normalization(sub_x, data_size)
    sub_y = normalization(sub_y, data_size)
    sub_a = normalization(sub_a, data_size)
    a_error = normalization(a_error, data_size)
    seq = normalization(seq, data_size)

    # Construct dataset: predict the next 10 points based on the previous 10 points 根据前10个轨迹估计后10个轨迹
    info = np.hstack([seq, pre_x, pre_y, sub_x, sub_y, sub_a, a_error])
    del seq, pre_x, pre_y, sub_x, sub_y, sub_a, a_error

    data_X, data_Y = create_dateset(info)

    # Split into training and test sets (7:3)
    train_size = int(len(data_X) * 0.7)
    print('train_size =', train_size)

    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]

    test_X = data_X[train_size:] #train_size+50*200]
    test_Y = data_Y[train_size:]

    train_X = train_X.reshape(-1, 10, 6)
    train_Y = train_Y.reshape(-1, 10)
    test_X = test_X.reshape(-1, 10, 6)
    test_Y = test_Y.reshape(-1, 10)

    print('shape of train_X', train_X.shape)
    print('shape of train_Y', train_Y.shape)
    print('shape of test_X', test_X.shape)
    print('shape of test_Y', test_Y.shape)

    return train_X, train_Y, test_X, test_Y
