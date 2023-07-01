import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import glob

def prepare_data(look_back, look_forward):
    folder_path = '../Data/NGSIM/chain/'
    file_paths = glob.glob(folder_path + '*.csv')
    X = []
    Y = []
    rows = []  # 保存行数信息
    A_hat_list = []
    A2_list = []
    k = 0
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['pre_V', 'A_hat'], how='all')
        veh_list = df['chain_id'].unique()
        for veh in veh_list:
            veh_df = df[df['chain_id'] == veh]
            data_size = len(veh_df)

            Y1 = veh_df['pre_Y'].values.reshape(-1, 1)
            Y2 = veh_df['Local_Y'].values.reshape(-1, 1)
            Speed1 = veh_df['pre_V'].values.reshape(-1, 1)
            Speed2 = veh_df['V'].values.reshape(-1, 1)
            A2 = veh_df['A'].values.reshape(-1, 1)
            A_hat = veh_df['A_hat'].values.reshape(-1, 1)
            A_error = A2 - A_hat

            # Normalization
            scaler = MinMaxScaler()
            Y1_normalized = scaler.fit_transform(Y1)
            Y2_normalized = scaler.fit_transform(Y2)
            Speed1_normalized = scaler.fit_transform(Speed1)
            Speed2_normalized = scaler.fit_transform(Speed2)
            A2_normalized = scaler.fit_transform(A2)
            A_error_normalized = scaler.fit_transform(A_error)
            a_error_min = min(A_error)
            a_error_max = max(A_error)

            # 构造特征和目标变量
            X_veh = []
            Y_veh = []
            for i in range(look_back, data_size - look_forward, look_back + look_forward):
                X_veh.append(np.concatenate((Y1_normalized[i - look_back:i, 0],
                                             Y2_normalized[i - look_back:i, 0],
                                             Speed1_normalized[i - look_back:i, 0],
                                             Speed2_normalized[i - look_back:i, 0],
                                             A2_normalized[i - look_back:i, 0]), axis=0))
                Y_veh.append(A_error_normalized[i:i + look_forward, 0])
                rows.append(k)  # k is the row for the predicted value
                k += 1

                A2_list.append(A2[i:i + look_forward, 0])
                A_hat_list.append(A_hat[i:i + look_forward, 0])

            X = X + X_veh
            Y = Y + Y_veh

    # 划分训练集和测试集
    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, y_train, y_test, rows_train, rows_test = train_test_split(X, Y, rows, test_size=0.3, random_state=42)

    A2_test_list = [A2_list[idx] for idx in rows_test]
    A_hat_test_list = [A_hat_list[idx] for idx in rows_test]

    return X_train, X_test, y_train, y_test, rows_test, a_error_min, a_error_max, A2_test_list, A_hat_test_list
