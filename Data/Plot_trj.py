import pandas as pd
import matplotlib.pyplot as plt

# Read in the csv file and store as a dataframe, ignoring the first five rows
df = pd.read_csv('../Data/ASta_040719_platoon3_new.csv')

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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 绘制速度
ax1.plot(df['Time'], df['Speed1'], label='Speed1')
ax1.plot(df['Time'], df['Speed2'], label='Speed2')
ax1.legend()
ax1.set_xlabel('Time')
ax1.set_ylabel('Speed')
ax1.set_title('Speed vs Time')

# 绘制加速度
ax2.plot(df['Time'], df['A1'], label='A')
ax2.plot(df['Time'], df['A2'], label='A2')
ax2.legend()
ax2.set_xlabel('Time')
ax2.set_ylabel('Acceleration')
ax2.set_title('Acceleration vs Time')

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.4)

# 保存图像并显示
plt.savefig('Plot_trj for platoon 3.jpg')
plt.show()




