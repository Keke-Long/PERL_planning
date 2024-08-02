'''
Add predicted values to the original data.
Prediction methods include: IDM, PERL
The results are stored in ‘./data/{DataName}_predicted/’
'''

import os
import numpy as np
from keras.models import load_model
import pandas as pd
from datetime import datetime
import data as dt
import joblib



def IDM(arg, vi, delta_v, delta_d):
    vf, A, b, s0, T = arg
    s_star = s0 + np.max([0, vi*T + (vi * delta_v) / (2 * (A*b) ** 0.5)], axis=0)
    #print('A*b',A*b)
    epsilon = 1e-5
    ahat = A*(1 - (vi/vf)**4 - (s_star/(delta_d+epsilon))**2)
    return ahat


def add_prediction(df):
    ### IDM ###
    df['a4_IDM'] = np.nan
    df['a4_IDM'].iloc[0] = df['a4'].iloc[0]
    for i in range(1, len(df)):
        vi = df.iloc[i - 1]['v4']
        delta_v = df.iloc[i - 1]['v4'] - df.iloc[i - 1]['v3']
        delta_d = df.iloc[i - 1]['Y3'] - df.iloc[i - 1]['Y4']
        arg = (22.2, 1.8, 2.0, 1.8, 1.9)
        df.loc[i, 'a4_IDM'] = IDM(arg, vi, delta_v, delta_d)

    ### PERL ###
    df['a4_PERL'] = np.nan
    df['a4_PERL'].iloc[:30] = df['a4'].iloc[:30]
    df['a4_PERL'].iloc[-30:] = df['a4'].iloc[-30:]
    for i in range(30, len(df)):
        # load trained model
        model = load_model("./PERL model/NGSIM_US101_backward30_foreward1.h5")

        # prepare data
        delta_Y = df['Y3'] - df['Y4']
        V_1 = df['v3']
        V = df['v4']
        A = df['a4']
        A_residual_IDM = df['a4_IDM'] - df['a4']

        delta_Y = delta_Y[i - 30:i]
        V_1 = V_1[i - 30:i]
        V = V[i - 30:i]
        A = A[i - 30:i]
        A_residual_IDM = A_residual_IDM[i - 30:i]

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
        df.loc[i, 'a4_PERL'] = df.loc[i, 'a4_IDM'] - A_residual_hat
    return df


DataName = "NGSIM_I80"

# Set path for data and output
data_folder = '/home/ubuntu/Documents/PERL_planning/data/NGSIM_I80'
output_folder = './data/NGSIM_I80_predicted'
os.makedirs(output_folder, exist_ok=True)

# Process each CSV file in the data folder
for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        output_file_path = os.path.join(output_folder, filename)
        if not os.path.exists(output_file_path):
            file_path = os.path.join(data_folder, filename)
            df = pd.read_csv(file_path)
            df = add_prediction(df)
            df.to_csv(os.path.join(output_folder, filename), index=False)
            print(f'Processed and saved: {filename}')
        else:
            print(f'Skipped as already processed: {filename}')
