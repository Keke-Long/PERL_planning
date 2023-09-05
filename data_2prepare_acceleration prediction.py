'''
为原始数据加上预测值，
预测方法包括：IDM，PERL
结果存在 ‘./data/{DataName}_predicted/’ 中
'''

import os
import numpy as np
import pandas as pd
from IDM import *
from keras.models import load_model
import pandas as pd
import argparse
from datetime import datetime
import data as dt
import joblib


#DataName = "NGSIM_I80"
DataName = "NGSIM_US101"

os.makedirs(f'./data/{DataName}_predicted', exist_ok=True)

data_folder_path = f'./data/{DataName}/'
all_files = [f for f in os.listdir(data_folder_path) if f.endswith('.csv')]
for filename in all_files:
    print("Dealing with",filename)
    df = pd.read_csv(f'./data/{DataName}/{filename}')

    ### IDM ###
    df['a0_IDM'] = np.nan
    df['a0_IDM'].iloc[0] = df['a0'].iloc[0]
    for i in range(1, len(df)):
        vi = df.iloc[i - 1]['v0']
        delta_v = df.iloc[i - 1]['v0'] - df.iloc[i - 1]['v-1']
        delta_d = df.iloc[i - 1]['Y-1'] - df.iloc[i - 1]['Y0']
        arg = (22.2, 1.2, 2.0, 1.8, 1.9)
        df.loc[i, 'a0_IDM'] = IDM(arg, vi, delta_v, delta_d)
    df.to_csv(f'./data/{DataName}_predicted/{filename}', index=False)

    ### PERL ###
    df['a0_PERL'] = np.nan
    df['a0_PERL'].iloc[:30] = df['a0'].iloc[:30]
    df['a0_PERL'].iloc[-30:] = df['a0'].iloc[-30:]
    for i in range(30, len(df)-30):
        # load trained model
        model = load_model("./PERL model/NGSIM_US101_backward30_foreward1.h5")

        # prepare data
        delta_Y = df['Y-1']-df['Y0']
        V_1 = df['v-1']
        V = df['v0']
        A = df['a0']
        A_residual_IDM = df['a0_IDM'] - df['a0']

        delta_Y = delta_Y[i-30:i]
        V_1 = V_1[i-30:i]
        V = V[i-30:i]
        A = A[i-30:i]
        A_residual_IDM = A_residual_IDM[i-30:i]

        scaler_delta_y = joblib.load('./PERL model/scaler_delta_y.pkl')
        scaler_v_1 = joblib.load('./PERL model/scaler_v_1.pkl')
        scaler_v = joblib.load('./PERL model/scaler_v.pkl')
        scaler_a = joblib.load('./PERL model/scaler_a.pkl')
        scaler_a_residual_IDM = joblib.load('./PERL model/scaler_a_residual_IDM.pkl')

        delta_Y_normalized = scaler_delta_y.transform(delta_Y.values.reshape(-1, 1))
        V_1_normalized = scaler_v_1.transform(V_1.values.reshape(-1, 1))
        V_normalized = scaler_v.transform(V.values.reshape(-1, 1))
        A_normalized = scaler_a.transform(A.values.reshape(-1, 1))
        A_residual_IDM_normalized = scaler_a_residual_IDM.transform(A_residual_IDM.values.reshape(-1, 1))

        X = np.concatenate((delta_Y_normalized,
                            V_1_normalized,
                            V_normalized,
                            A_normalized,
                            A_residual_IDM_normalized), axis=0)
        # get predicted value
        X = X.reshape(1, 30, 5)
        A_residual_hat = model.predict(X)
        A_residual_hat = A_residual_hat.squeeze(axis=-1)
        A_residual_hat = scaler_a_residual_IDM.inverse_transform(A_residual_hat)
        df.loc[i, 'a0_PERL'] = df.loc[i, 'a0_IDM'] - A_residual_hat

    df.to_csv(f'./data/{DataName}_predicted/{filename}', index=False)
