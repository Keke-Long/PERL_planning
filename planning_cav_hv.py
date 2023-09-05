import numpy as np
import pandas as pd
from scipy.linalg import block_diag, tril
import cvxpy as cp
from planning_funs import *
from IDM import *


class Origin_trj:
    def __init__(self, t, a_0_origin, a_1_origin, a_2_origin, a_0_IDM,
                          v_0_origin, v_1_origin, v_2_origin,
                          x_0_origin, x_1_origin, x_2_origin):
        self.t = t
        self.a_0_origin = a_0_origin
        self.a_1_origin = a_1_origin
        self.a_2_origin = a_2_origin
        self.v_0_origin = v_0_origin
        self.v_1_origin = v_1_origin
        self.v_2_origin = v_2_origin
        self.x_0_origin = x_0_origin
        self.x_1_origin = x_1_origin
        self.x_2_origin = x_2_origin
        self.a_0_IDM = a_0_IDM

class MPC_cav_trj:
    def __init__(self, t, a_1, a_2, v_1, v_2, x_1, x_2):
        self.t = t
        self.a_1 = a_1
        self.a_2 = a_2
        self.v_1 = v_1
        self.v_2 = v_2
        self.x_1 = x_1
        self.x_2 = x_2


## parameter setting
L = 5      # 车辆长度
d0 = 5     # 最小安全间距
h = 1.9
tao = 0.4  # 机械时延
ts = 0.1   # 仿真步长
Np = 6  # 预测步长
Nc = 5  # 控制步长
Nx = 5     # 状态量数目
Ny = 5     # 输出量数目
Nu = 1     # 控制量数目
T =  h    # T in IDM model 是IDM稳定时的headway
s0 = 2


## load data
scenario = "IDM_prediction"
scenario = "PERL_prediction"

#DataName = "NGSIM_I80"
DataName = "NGSIM_US101"
os.makedirs(f'./results/{DataName}', exist_ok=True)

# filename = "lane3_veh1195_1198_1204.csv"
# filename = "lane2_veh313_322_328.csv"
filename = "lane2_veh817_822_827.csv"

def process_file(filename, DataName, scenario):
    df = pd.read_csv(f'./data/{DataName}_predicted/{filename}')
    n = len(df)
    Y0_start = min(df['Y0'])
    x_0_origin = df['Y0'] - Y0_start
    v_0_origin = df['v0']
    a_0_origin = df['a0']
    a_0_IDM = df['a0_IDM']
    a_0_PERL= df['a0_PERL']

    x_1_origin = df['Y1'] - Y0_start
    v_1_origin = df['v1']
    a_1_origin = df['a1']

    x_2_origin = df['Y2'] - Y0_start
    v_2_origin = df['v2']
    a_2_origin = df['a2']

    ref = np.zeros((n, 5))
    ref[:, 2] = a_0_origin

    ## 系统状态方程
    # Continuous system state equation Dx = A*x+B1*u+B2*w; C = C*x+D*u
    A = ts*np.array([[0, 1, -h, 0, 0],
                     [0, 0, -1, 0, 0],
                     [0, 0, -1/tao, 0, 0],
                     [0, 0, 0, 0, 1],
                     [0, 0, 1, 0, 0]]) + np.eye(5)
    B1 = ts*np.array([[0], [0], [1/tao], [0], [0]])
    B2 = ts*np.array([[0, 0],
                      [1, 0],
                      [0, 0],
                      [0, -T],
                      [0, -1]])
    C = np.eye(5)

    # 构造权重矩阵Q, R
    r = 0.1  #控制增量权重
    q = np.diag([2, 1, 10, 1, 1])
    Q = np.kron(np.eye(Np), q)
    R = r * np.eye(Nc)

    # Discrete system state equation coefficients
    Mx_cell = [None]*Np
    Mu_cell = [None]*Np
    Mw_cell = [None]*Np
    for i in range(Np):
        Mx_cell[i] = C @ np.linalg.matrix_power(A, i + 1)
        Mu_cell[i] = np.zeros((C.shape[0], B1.shape[1]))
        Mw_cell[i] = np.zeros((C.shape[0], B2.shape[1]))
        for j in range(i + 1):
            Mu_cell[i] += C @ np.linalg.matrix_power(A, j) @ B1
            Mw_cell[i] += C @ np.linalg.matrix_power(A, j) @ B2
    Mx = np.concatenate(Mx_cell, axis=0)
    Mu = np.concatenate(Mu_cell, axis=0)
    Mw = np.concatenate(Mw_cell, axis=0)

    Mdu_cell = [[None]*Nc for _ in range(Np)]
    for i in range(Np):
        for j in range(Nc):
            Mdu_cell[i][j] = np.zeros((C.shape[0], B1.shape[1]))
            if j <= i:
                for k in range(j, i+1):
                    Mdu_cell[i][j] += C @ np.linalg.matrix_power(A, i-k) @ B1
    Mdu = np.block(Mdu_cell)

    ## Constraints setting
    # Constraints of u
    umin = -3
    umax = 2.5
    U_min = np.array([umin] * Nc)
    U_max = np.array([umax] * Nc)
    # Constrain of delta u
    du_min = -3
    du_max = 3
    delta_umax = [du_max] * Nc
    delta_umin = [du_min] * Nc
    Row = 0.1              #松弛系数
    M = 5                  #松弛变量上界
    lb = delta_umin + [0]  #（求解方程）状态量下界，包含控制时域内控制增量和松弛因子
    lb = np.array(lb)
    ub = delta_umax + [M]  #（求解方程）状态量上界，包含控制时域内控制增量和松弛因子
    ub = np.array(ub)
    # 输出状态约束
    es_min = -1
    es_max = 3
    ev_min = -2
    ev_max = 2
    a_min = -3
    a_max = 2.5
    v_min = 0
    v_max = 30
    y_min = np.array([es_min, ev_min, a_min, es_min, ev_min])
    y_max = np.array([es_max, ev_max, a_max, es_max, ev_max])
    Y_min = np.kron(np.ones(Np), y_min)
    Y_max = np.kron(np.ones(Np), y_max)

    ## 松弛约束设置
    # 控制量松弛约束
    vdu_min = 0
    vdu_max = 0
    Vdu_min = np.kron(np.ones(Nc), vdu_min)
    Vdu_max = np.kron(np.ones(Nc), vdu_max)
    # 控制增量松弛约束
    vu_min = -0.01
    vu_max = 0.01
    Vu_min = np.kron(np.ones(Nc), vu_min)
    Vu_max = np.kron(np.ones(Nc), vu_max)
    # 输出状态松弛约束
    vy_min = np.array([0, -1, -0.1, 0, -1])
    vy_max = np.array([1,  1,  0.1, 1,  1])
    VY_min = np.kron(np.ones(Np), vy_min)
    VY_max = np.kron(np.ones(Np), vy_max)

    ## 初始值设置和预定义
    X = np.zeros((n+1, 5))
    U = np.zeros((n+1, 1))

    Veh0_init_v = df.iloc[0]['v0']
    Veh0_init_x = df.iloc[0]['Y0']
    Veh1_init_a = df.iloc[0]['a1']
    Veh1_init_v = df.iloc[0]['v1']
    Veh1_init_x = df.iloc[0]['Y1']
    Veh2_init_v = df.iloc[0]['v2']
    Veh2_init_x = df.iloc[0]['Y2']

    delta_x1 = Veh0_init_x - Veh1_init_x -L - h * Veh1_init_v
    delta_v1 = Veh0_init_v - Veh1_init_v
    delta_x2 = Veh1_init_x - Veh2_init_x -L - T * Veh2_init_v
    delta_v2 = Veh1_init_v - Veh2_init_v

    X[0, :] = [delta_x1, delta_v1, Veh1_init_a, delta_x2, delta_v2]
    #print('X[0, :]=', X[0, :])
    U[0, :] = Veh1_init_a
    A_I = np.kron(tril(np.ones((Nc, Nc))), np.eye(Nu))
    PSI = np.kron(np.ones(Nc), np.eye(Nu))

    a_2 = np.zeros((n, 1))

    ## MPC
    for k in range(200):
        print("k=", k)
        vi = v_0_origin[k] - X[k, 1] - X[k, 4] # HV speed
        delta_v = -X[k, 4]
        delta_d = X[k, 3] + s0 + vi * T
        arg = (22.2, 1.2, 2.0, 1.8, 1.9)
        a_22 = IDM(arg, vi, delta_v, delta_d)
        a_2[k] = a_22

        if scenario == "No_prediction":
            w = np.array([a_0_origin[k], a_22])
        elif scenario == "IDM_prediction":
            w = np.array([a_0_IDM[k], a_22])
        elif scenario == "PERL_prediction":
            w = np.array([a_0_PERL[k], a_22])

        H_cell = [[2 * (Mdu.T @ Q @ Mdu + R), np.zeros((Nu * Nc, 1))],
                  [np.zeros((1, Nu * Nc)), Row]]
        H = np.block(H_cell)
        H = (H + H.T) / 2

        E = np.kron(np.ones(Np), ref[k, :]) - Mx @ X[k, :].T - Mu @ U[k, :].T - Mw @ w
        f = np.concatenate([-2 * (Mdu.T @ Q @ E), [0]])

        Vu_max_2D = np.expand_dims(Vu_max, axis=-1)  # reshape from (5,) to (5,1)
        Vu_min_2D = np.expand_dims(Vu_min, axis=-1)
        VY_max_2D = np.expand_dims(VY_max, axis=-1)  # reshape from (30,) to (30,1)
        VY_min_2D = np.expand_dims(VY_min, axis=-1)
        A_cons_cell = np.vstack([
            np.hstack([A_I, -Vu_max_2D]),
            np.hstack([-A_I, Vu_min_2D]),
            np.hstack([Mdu, -VY_max_2D]),
            np.hstack([-Mdu, VY_min_2D])
        ])

        X_k_T = X[k, :].reshape(-1, 1)  # 这将使它成为一个列向量
        U_k_T = U[k, :].reshape(-1, 1)  # 这将使它成为一个列向量
        w = w.reshape(-1, 1)  # 如果 w 是一维数组，那么也需要进行同样的转换
        PSI_T = PSI.T

        U_max_2D = U_max.reshape(-1, 1)
        U_min_2D = U_min.reshape(-1, 1)
        Y_max_2D = Y_max.reshape(-1, 1)
        Y_min_2D = Y_min.reshape(-1, 1)

        b_cons_cell = [
            U_max_2D - PSI.T @ U[k, :].reshape(-1, 1),
            -U_min_2D + PSI.T @ U[k, :].reshape(-1, 1),
            Y_max_2D - Mx @ X_k_T - Mu @ U_k_T - Mw @ w,
            -Y_min_2D + Mx @ X_k_T + Mu @ U_k_T + Mw @ w
        ]

        A_cons = np.vstack(A_cons_cell)
        b_cons = np.vstack(b_cons_cell)
        b_cons = np.squeeze(b_cons)

        du = cp.Variable(Nc+1)
        objective = cp.Minimize(0.5 * cp.quad_form(du, H) + f.T @ du)
        constraints = [A_cons @ du <= b_cons, lb <= du, du <= ub]
        problem = cp.Problem(objective, constraints)

        result = problem.solve()
        res = du.value

        # print("problem.status =", problem.status)
        # return problem.status == 'optimal'

        if problem.status == 'optimal':
            U[k+1, :] = U[k, :] + res[:Nu] # 更新控制量
            # print('U[k+1, :]', U[k+1, :])
            X[k+1, :] = A @ X[k, :].T + B1 @ U[k + 1, :].T + (B2 @ w).T # 更新状态量
            # print('X[k+1, 2] = a[k+1]=', X[k+1, 2])
        else:
            return

    # results processing
    a_1 = X[:n, 2]   # 不是a_1 = U[:-1]
    v_1 = v_0_origin - X[:n, 1]
    x_1 = x_0_origin - X[:n, 0] - L - h * v_1
    v_2 = v_1 - X[:n, 4]
    x_2 = x_1 - X[:n, 3] - L - T * v_2

    ## Store results in instances of Origin_trj and MPC_cav_trj
    origin_trj = Origin_trj(t=df['t'], a_0_origin=a_0_origin, a_1_origin=a_1_origin, a_2_origin=a_2_origin, a_0_IDM=a_0_IDM,
                                       v_0_origin=v_0_origin, v_1_origin=v_1_origin, v_2_origin=v_2_origin,
                                       x_0_origin=x_0_origin, x_1_origin=x_1_origin, x_2_origin=x_2_origin)
    mpc_trj = MPC_cav_trj(t=df['t'], a_1=a_1, a_2=a_2, v_1=v_1, v_2=v_2, x_1=x_1, x_2=x_2)

    #plot
    plot_results(origin_trj, mpc_trj, DataName, filename, scenario, n)
    #save result trajectories
    save_trj(origin_trj, mpc_trj, DataName, filename, scenario)
    # compare the origin trajectory and the MPC trajectory
    save_measures(origin_trj, mpc_trj, DataName, filename, scenario)

# process_file(filename=filename, DataName=DataName, scenario="IDM_prediction")
# process_file(filename=filename, DataName=DataName, scenario="PERL_prediction")

# 遍历文件 寻找可以求解的轨迹
data_folder_path = f'./data/{DataName}_predicted/'
all_files = [f for f in os.listdir(data_folder_path) if f.endswith('.csv')]
for filename in all_files:
    #print(f"Processing {filename}...")
    process_file(filename=filename, DataName=DataName, scenario="IDM_prediction")
    process_file(filename=filename, DataName=DataName, scenario="PERL_prediction")




