# 读取这个数据/home/ubuntu/Documents/NGSIM-Haotian Shi/NGSIM_Cleaned_Dataset-main/I-80_download/EMD_Reconstructed(i80 0400-0415).csv
# 这个数据包含以下这些列：
# Vehicle_ID	Frame_ID	Total_Frames	Global_Time	Local_X	Local_Y	Global_X	Global_Y	v_Length	v_Width	v_Class	v_Vel	v_Acc	Lane_ID	Preceding	Following	Space_Headway	Time_Headway
# 我需要筛选连续三辆车的数据，也就是说在连续100条数据内，这三辆车保持之间的跟驰关系，将数据存为csv，格式为:
# t, veh_ID1, Y1, v1, a1, Preceding1,	Following1, veh_ID2, Y2, v2, a2, veh_ID3, Y3, v3, a3

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/ubuntu/Documents/NGSIM-Haotian Shi/NGSIM_Cleaned_Dataset-main/I-80_download/EMD_Reconstructed(i80 0400-0415).csv')
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


output_dfs = []
vehicle_ids = df_lane['Vehicle_ID'].unique()
for veh_id in vehicle_ids:
    veh1_df = df_lane[df_lane['Vehicle_ID'] == veh_id]

    foll_veh_id = veh_id

    frame_count = 0

    # Initialize a temporary dataframe for the current vehicle chain
    temp_df = pd.DataFrame(columns=['t', 'veh_ID1', 'Y1', 'v1', 'a1', 'Following1',
                                    'veh_ID2', 'Y2', 'v2', 'a2', 'Following2', 'veh_ID3', 'Y3', 'v3', 'a3'])

    # Iterate over frames
    for _, row in veh1_df.iterrows():
        frame_id = row['Frame_ID']

        # Get data for the following vehicle in the same frame
        veh2_df = df_lane[(df_lane['Vehicle_ID'] == row['Following']) & (df_lane['Frame_ID'] == frame_id)]
        if veh2_df.empty:
            break

        # Get data for the following vehicle of veh2 in the same frame
        veh3_df = df_lane[(df_lane['Vehicle_ID'] == veh2_df.iloc[0]['Following']) & (df_lane['Frame_ID'] == frame_id)]
        if veh3_df.empty:
            break

        new_row = {
            't': row['Global_Time'],
            'veh_ID1': row['Vehicle_ID'],
            'Y1': row['Local_Y'],
            'v1': row['v_Vel'],
            'a1': row['v_Acc'],
            'Following1': row['Following'],

            'veh_ID2': veh2_df.iloc[0]['Vehicle_ID'],
            'Y2': veh2_df.iloc[0]['Local_Y'],
            'v2': veh2_df.iloc[0]['v_Vel'],
            'a2': veh2_df.iloc[0]['v_Acc'],
            'Following2': veh2_df.iloc[0]['Following'],

            'veh_ID3': veh3_df.iloc[0]['Vehicle_ID'],
            'Y3': veh3_df.iloc[0]['Local_Y'],
            'v3': veh3_df.iloc[0]['v_Vel'],
            'a3': veh3_df.iloc[0]['v_Acc'],
        }

        # Append the row to the temporary dataframe
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

        # Update frame count
        frame_count += 1

        # If 100 frames are reached, stop processing this vehicle
        if frame_count >= 200:
            veh_ID1 = int(row["Vehicle_ID"])
            veh_ID2 = int(row["Following"])
            veh_ID3 = int(veh3_df.iloc[0]["Vehicle_ID"])
            filename = f'./NGSIMdata/lane{Lane_num}_veh{veh_ID1}_{veh_ID2}_{veh_ID3}.csv'
            temp_df.to_csv(filename, index=False)

            # Create a figure with 3 subplots
            fig, axs = plt.subplots(3)

            # Add data to the subplots
            axs[0].plot(temp_df['t'], temp_df['Y1'], label='veh1_Y')
            axs[0].plot(temp_df['t'], temp_df['Y2'], label='veh2_Y')
            axs[0].plot(temp_df['t'], temp_df['Y3'], label='veh3_Y')
            axs[0].set_ylabel('Y (m)')
            axs[0].legend()

            axs[1].plot(temp_df['t'], temp_df['v1'], label='veh1_v')
            axs[1].plot(temp_df['t'], temp_df['v2'], label='veh2_v')
            axs[1].plot(temp_df['t'], temp_df['v3'], label='veh3_v')
            axs[1].set_ylabel('v (m/s)')
            axs[1].legend()

            axs[2].plot(temp_df['t'], temp_df['a1'], label='veh1_a')
            axs[2].plot(temp_df['t'], temp_df['a2'], label='veh2_a')
            axs[2].plot(temp_df['t'], temp_df['a3'], label='veh3_a')
            axs[2].set_ylabel('a (m/s²)')
            axs[2].set_xlabel('t')
            axs[2].legend()

            # Save the figure as a png file
            fig.savefig(f'./NGSIMdata/lane{Lane_num}_veh{veh_ID1}_{veh_ID2}_{veh_ID3}.png')

            break


