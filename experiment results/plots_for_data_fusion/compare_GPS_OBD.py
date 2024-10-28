import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data0 = pd.read_csv('./5-23/1/1.csv', header=None)[20:].reset_index(drop=True)

data1 = pd.read_csv('./5-23/1/11speedOBD.csv', header=None).values.flatten() *0.47
data1 = data1[30:]
data2 = pd.read_csv('./5-23/1/12speedOBD.csv', header=None).values.flatten() *0.47
data2 = np.concatenate((np.zeros(34), data2))
data3 = pd.read_csv('./5-23/1/13fspeedOBD.csv', header=None).values.flatten() *0.47
data3 = np.concatenate((np.zeros(37), data3))

data4 = pd.read_csv('./5-23/1/aligned_real_speeds1.csv', header=None).values.flatten()
data5 = pd.read_csv('./5-23/1/aligned_real_speeds2.csv', header=None).values.flatten()
data6 = pd.read_csv('./5-23/1/aligned_real_speeds3.csv', header=None).values.flatten()


# 创建图形
plt.figure(figsize=(5, 2.4))
plt.plot(np.arange(len(data0))/10, data0, label='Planned trajectory', linestyle='-', color='black')

plt.plot(np.arange(len(data1))/10, data1, label='Trajectory1 from OBD', linestyle='--', color='orange')
plt.plot(np.arange(len(data4))/10, data4, label='Trajectory1 from GPS', linestyle=':', color='blue')

# plt.plot(np.arange(len(data2))/10, data2, label='Trajectory2 from OBD', linestyle='--', color='orange')
# plt.plot(np.arange(len(data5))/10, data5, label='Trajectory2 from GPS', linestyle=':', color='blue')
#
# plt.plot(np.arange(len(data3))/10, data3, label='Trajectory3 from OBD', linestyle='--', color='orange')
# plt.plot(np.arange(len(data6))/10, data6, label='Trajectory3 from GPS', linestyle=':', color='blue')

plt.legend()

# set title
plt.xlabel('Time (s)')
plt.ylabel('Vehicle Speed (m/s)')
plt.ylim([4, 14])
plt.xlim([4, 24])
plt.xticks(range(0, 25, 2)) 
plt.tight_layout()
plt.savefig('compare_data_from_GPS_OBD横坐标.png', dpi=300)

