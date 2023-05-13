import pandas as pd
import matplotlib.pyplot as plt

# Read in the csv file and store as a dataframe, ignoring the first five rows
df = pd.read_csv('../Data/ASta_050719_platoon1.csv', skiprows=range(5))

# # Plot the trajectories of all five cars
# fig, ax = plt.subplots()
# for car_id in range(1, 6):
#     east_col = f'E{car_id}'
#     north_col = f'N{car_id}'
#     east_vals = df[east_col].tolist()
#     north_vals = df[north_col].tolist()
#     ax.plot(east_vals, north_vals, label=f'Car {car_id}')
# ax.set_xlabel('East (m)')
# ax.set_ylabel('North (m)')
# ax.set_title('Vehicle Trajectories')
# ax.legend()
# plt.show()


# # Plot the trajectories of all five cars for the first 5 minutes (300 seconds) of data
# fig, ax = plt.subplots()
# for car_id in range(1, 6):
#     east_col = f'E{car_id}'
#     north_col = f'N{car_id}'
#     east_vals = df[east_col].tolist()[:400]
#     north_vals = df[north_col].tolist()[:400]
#     ax.plot(east_vals, north_vals, label=f'Car {car_id}')
# ax.set_xlabel('East (m)')
# ax.set_ylabel('North (m)')
# ax.set_title('Vehicle Trajectories from 0-5 Minutes')
# ax.legend()
# plt.show()


# 创建一个新的 Matplotlib 图形并添加两条线
fig, ax = plt.subplots()
ax.plot(df['Time'], df['Speed1'], label='Speed1')
ax.plot(df['Time'], df['Speed2'], label='Speed2')

# 添加图例和标签
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Speed')
ax.set_title('Speed vs Time')
plt.show()





