'''
Choose the 5th veh as subject veh be predicted
Calculate  v,a for 5th veh and 4th veh
'''

import pandas as pd


df = pd.read_csv('../Data/ASta_050719_platoon1.csv', skiprows=range(5))
new_df = df.loc[:, ['Time', 'Speed1', 'E1', 'N1', 'Speed2', 'E2', 'N2', 'IVS1']]

# Calculate acceleration
time_diff = new_df['Time'].diff()
speed_diff4 = new_df['Speed1'].diff()
speed_diff5 = new_df['Speed2'].diff()
new_df['A1'] = speed_diff4 / time_diff  # 计算加速度
new_df['A2'] = speed_diff5 / time_diff

new_df.to_csv('../Data/new_file.csv', index=False)
