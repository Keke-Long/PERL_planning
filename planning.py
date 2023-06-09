import random
import math
import numpy as np
from tqdm import tqdm
from Physical_model.IDM import IDM
import matplotlib.pyplot as plt

class Vehicle:
    def __init__(self, position, velocity, acceleration):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration

def generate_hv_refer_trajectory():
    refer_trajectory = [(0,15,0)]
    velocity = 15
    position = 0
    for i in range(50):
        if 20 < i < 30:
            acceleration = -3
        elif 30 <= i < 45:
            acceleration = 3
        else:
            acceleration = 0
        velocity += acceleration * 0.1
        position += velocity * 0.1
        refer_trajectory.append((position, velocity, acceleration))
    return np.array(refer_trajectory)

def generate_CAV_trajectory(trajectory, step_num):
    acceleration = random.uniform(-5, 5)  # 随机生成加速度
    for i in range(1, step_num):
        velocity = trajectory[-1].velocity + acceleration * 0.1  # 使用前一时刻的速度和随机生成的加速度计算当前时刻的速度
        position = trajectory[-1].position + trajectory[-1].velocity * 0.1 + 0.5 * acceleration * (0.1 ** 2)  # 使用前一时刻的位置、速度和随机生成的加速度计算当前时刻的位置
        veh = Vehicle(position, velocity, acceleration)
        trajectory.append(veh)
    
    return trajectory

def calculate_hv_trajectory(CAV_trajectory, hv_trajectory, step_num):
    n = len(hv_trajectory)
    m = len(hv_trajectory) + step_num
    for i in range(n, m):
        pre_veh = CAV_trajectory[i-1]
        ego_veh = hv_trajectory[i-1]
    
        delta_v = ego_veh.velocity - pre_veh.velocity
        delta_d = pre_veh.position - ego_veh.position
    
        arg = (23.5617, 1.0033, 3.2735, 2.9015, 2.6154)
        ahat = IDM(arg, ego_veh.velocity, delta_v, delta_d)
    
        # 限制加速度在范围内
        hv_acceleration = max(-3, min(3, ahat))
        hv_velocity = ego_veh.velocity + ego_veh.acceleration * 0.1
        hv_position = ego_veh.position + ego_veh.velocity * 0.1 + 0.5 * ego_veh.acceleration * (0.1 ** 2)
        hv = Vehicle(hv_position, hv_velocity, hv_acceleration)
        hv_trajectory.append(hv)

    return hv_trajectory


def evaluate(CAV_trajectory, hv_trajectory, hv_refer_trajectory):
    Cost = abs(hv_trajectory[-1].position - hv_refer_trajectory[-1])  # 第二辆车实际轨迹的位置与reference轨迹的差
    acceleration_sum = sum([abs(veh.acceleration) ** 2 for veh in hv_trajectory])  # 所有车辆的每时刻的加速度的平方的累计和
    return Cost


if __name__ == "__main__":
    random.seed(21)
    CAV_trajectory_record = None
    HV_trajectory_record = None
    
    best_Cost = float('inf')
    hv_refer_trajectory = generate_hv_refer_trajectory()
    
    CAV_trajectory = [Vehicle(30, 16, 0)] # CAV初始状态
    hv_trajectory = [Vehicle(-10, 15, 0)]  # HV初始状态

    # 从t=0开始，
    for t in tqdm(range(1,len(hv_refer_trajectory))):
        min_Cost = float('inf')
        min_trajectory = None
        # print('len(CAV_trajectory), len(hv_trajectory) = ', len(CAV_trajectory), len(hv_trajectory))

        # 循环多次找到合理控制量
        for _ in range(5000):
            CAV_trajectory_try = CAV_trajectory.copy()
            hv_trajectory_try = hv_trajectory.copy()

            # CAV预测step_num个步长
            step_num = 10
            CAV_trajectory_try = generate_CAV_trajectory(CAV_trajectory, step_num)

            # HV这里同样更新step_num个步长
            hv_trajectory_try = calculate_hv_trajectory(CAV_trajectory_try, hv_trajectory_try, step_num)
            Cost = evaluate(CAV_trajectory_try, hv_trajectory_try, hv_refer_trajectory[:, 0])

            if Cost < min_Cost:
                min_Cost = Cost
                best_cav_trajectory = CAV_trajectory_try
                best_hv_trajectory = hv_trajectory_try

        CAV_trajectory = best_cav_trajectory[:t+1]
        hv_trajectory = best_hv_trajectory[:t+1]

    print(hv_refer_trajectory)
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    time = np.arange(0, len(CAV_trajectory))
    plt.plot(time, [veh.position for veh in CAV_trajectory], label="CAV")
    plt.plot(time, [veh.position for veh in hv_trajectory], label="HV")
    plt.plot(time, hv_refer_trajectory[:, 0], label="Reference")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Vehicle Trajectories")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    time = np.arange(0, len(CAV_trajectory))
    plt.plot(time, [veh.velocity for veh in CAV_trajectory], label="CAV")
    plt.plot(time, [veh.velocity for veh in hv_trajectory], label="HV")
    plt.plot(time, hv_refer_trajectory[:, 1], label="Reference")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Vehicle Velocity")
    plt.legend()
    plt.grid(True)




    plt.show()
