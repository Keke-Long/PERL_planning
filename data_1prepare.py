'''
读取这个数据/home/ubuntu/Documents/NGSIM-Haotian Shi/NGSIM_Cleaned_Dataset-main/I-80_download/EMD_Reconstructed(i80 0400-0415).csv
这个数据包含以下这些列：
Vehicle_ID	Frame_ID	Total_Frames	Global_Time	Local_X	Local_Y	Global_X	Global_Y	v_Length	v_Width	v_Class	v_Vel	v_Acc	Lane_ID	Preceding	Following	Space_Headway	Time_Headway
我需要筛选连续三辆车的数据，也就是说在连续100条数据内，这三辆车保持之间的跟驰关系，将数据存为csv，格式为:
t, veh_ID1, Y1, v1, a1, Preceding1,	Following1, veh_ID2, Y2, v2, a2, veh_ID3, Y3, v3, a3
未完成：把数据都改成只有4位
数据存储在 './data/{DataName}/‘ 中
'''

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# from planning_funs import *

def smooth_acceleration(df, column, threshold, window_size):
    # 计算diff
    diff = df[column].diff().abs()
    # 标记异常值
    outlier_mask = diff > threshold
    # 使用前后两个有效值的平均值替代异常值
    df[column + '_filtered'] = df[column]
    df.loc[outlier_mask, column + '_filtered'] = (df[column].shift() + df[column].shift(-1)) / 2
    # 对filtered列应用滑动平均滤波
    df[column + '_filtered'] = df[column + '_filtered'].rolling(window_size, min_periods=1).mean()
    # 使用前一有效值填充第一个值的异常值
    df[column + '_filtered'] = df[column + '_filtered'].fillna(method='ffill')
    df[column] = df[column + '_filtered']
    df = df.drop(column + '_filtered', axis=1)
    return df


#DataName = "NGSIM_I80"
DataName = "NGSIM_US101"

if DataName == "NGSIM_I80":
    df = pd.read_csv('/home/ubuntu/Documents/NGSIM-Haotian Shi/NGSIM_Cleaned_Dataset-main/I-80_download/EMD_Reconstructed(i80 0400-0415).csv')
elif DataName == "NGSIM_US101":
    df = pd.read_csv('/home/ubuntu/Documents/NGSIM-Haotian Shi/NGSIM_Cleaned_Dataset-main/US-101_download/EMD_Reconstructed(us-101 0820am-0835am).csv')

df.sort_values(['Frame_ID', 'Local_Y'], ascending=[True, False], inplace=True)

# 单位转换：ft -> m, ft/s -> m/s, ft/s^2 -> m/s^2
df['Local_X'] *= 0.3048
df['Local_Y'] *= 0.3048
df['v_Vel'] *= 0.3048
df['v_Acc'] *= 0.3048
df['Space_Headway'] *= 0.3048
df['Global_Time'] *= 0.001

Lane_num = 2
df_lane = df[df['Lane_ID'] == Lane_num]

os.makedirs(f'./data/{DataName}', exist_ok=True)

output_dfs = []
vehicle_ids = df_lane['Vehicle_ID'].unique()
for veh_id in vehicle_ids:
    veh0_df = df_lane[df_lane['Vehicle_ID'] == veh_id]

    frame_count = 0

    temp_df = pd.DataFrame(columns=['t', 'veh_ID0', 'Y0', 'v0', 'a0',
                                         'veh_ID1', 'Y1', 'v1', 'a1',
                                         'veh_ID2', 'Y2', 'v2', 'a2'])

    for _, row in veh0_df.iterrows():
        frame_id = row['Frame_ID']

        veh_df = df_lane[(df_lane['Vehicle_ID'] == row['Preceding']) & (df_lane['Frame_ID'] == frame_id)]
        if veh_df.empty:
            break

        veh1_df = df_lane[(df_lane['Vehicle_ID'] == row['Following']) & (df_lane['Frame_ID'] == frame_id)]
        if veh1_df.empty:
            break

        veh2_df = df_lane[(df_lane['Vehicle_ID'] == veh1_df.iloc[0]['Following']) & (df_lane['Frame_ID'] == frame_id)]
        if veh2_df.empty:
            break

        new_row = {
            't': row['Global_Time'],
            'veh_ID0': row['Vehicle_ID'],
            'Y0': row['Local_Y'],
            'v0': row['v_Vel'],
            'a0': row['v_Acc'],

            #为了IDM用，需要前车信息
            #PINN目前也只用一辆前车的信息，
            'veh_ID-1': row['Preceding'],
            'Y-1': veh_df.iloc[0]['Local_Y'],
            'v-1': veh_df.iloc[0]['v_Vel'],
            'a-1': veh_df.iloc[0]['v_Acc'],

            'veh_ID1': veh1_df.iloc[0]['Vehicle_ID'],
            'Y1': veh1_df.iloc[0]['Local_Y'],
            'v1': veh1_df.iloc[0]['v_Vel'],
            'a1': veh1_df.iloc[0]['v_Acc'],

            'veh_ID2': veh2_df.iloc[0]['Vehicle_ID'],
            'Y2': veh2_df.iloc[0]['Local_Y'],
            'v2': veh2_df.iloc[0]['v_Vel'],
            'a2': veh2_df.iloc[0]['v_Acc'],
        }

        # Append the row to the temporary dataframe
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
        # 位置突变限制
        delta_Y_limit = 10
        if len(temp_df) > 0:
            diff_Y0 = abs(temp_df['Y0'].diff())
            diff_Y1 = abs(temp_df['Y1'].diff())
            diff_Y2 = abs(temp_df['Y2'].diff())
            if (diff_Y0 > delta_Y_limit).any() or (diff_Y1 > delta_Y_limit).any() or (diff_Y2 > delta_Y_limit).any():
                break

        frame_count += 1

        # If 100 frames are reached, stop processing this vehicle
        if frame_count >= 250:
            veh_ID0 = int(veh0_df.iloc[0]["Vehicle_ID"])
            veh_ID1 = int(veh1_df.iloc[0]["Vehicle_ID"])
            veh_ID2 = int(veh2_df.iloc[0]["Vehicle_ID"])

            # fillter
            # temp_df = smooth_acceleration(temp_df, 'a0', threshold=0.4, window_size=5)
            # temp_df = smooth_acceleration(temp_df, 'a1', threshold=0.4, window_size=5)
            # temp_df = smooth_acceleration(temp_df, 'a2', threshold=0.4, window_size=5)

            filename = f"lane{Lane_num}_veh{veh_ID0}_{veh_ID1}_{veh_ID2}"
            temp_df.to_csv(f'./data/{DataName}/{filename}.csv', index=False)
            break
