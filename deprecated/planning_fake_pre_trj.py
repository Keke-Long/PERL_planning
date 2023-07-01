import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag, tril
from scipy.optimize import minimize
from scipy.signal import cont2discrete
import cvxpy as cp


def IDM(vi, delta_v, delta_d):
    vf= 30
    A = 3
    b = 3.2735
    s0= 2
    T = 2.2
    s_star = s0 + max(0, (vi*T + (vi * delta_v) / (2 * np.sqrt(A*b))) )
    epsilon = 1e-20
    ahat = A*(1 - (vi/vf)**4 - (s_star/(delta_d + epsilon))**2)
    return ahat

## parameter setting
L = 5      # 车辆长度
d0 = 5     # 最小安全间距
h = 0.6    # 发动机时延
tao = 0.4  # 机械时延
ts = 0.1   # 仿真步长
Np = 6     # 预测步长
Nc = 5     # 控制步长
Nx = 5     # 状态量数目
Ny = 5     # 输出量数目
Nu = 1     # 控制量数目
T = 2.2    # T in IDM model 是IDM稳定时的headway
T_sim = 40

## Trajectory of preceding vehicle
t = np.arange(0, T_sim+ts, ts)          # 仿真时间域
n = t.shape[0]

ref = np.zeros((n, 5))
a_preceding = np.zeros((n,)) # 前车加速度
v_preceding = np.zeros((n,)) # 前车速度
x_preceding = np.zeros((n,)) # 前车位置
for i in range(n):
    if t[i] <= 20:
        a_preceding[i] = 1
        v_preceding[i] = a_preceding[i]*t[i]
    elif 20 < t[i] <= 30:
        v_preceding[i] = 20
    elif 30 < t[i] <= 40:
        a_preceding[i] = -1.5
        v_preceding[i] = 20 + a_preceding[i]*(t[i] - 30)
    if i == 0:
        x_preceding[i] = 0
    else:
        x_preceding[i] = x_preceding[i-1] + v_preceding[i]*ts + 0.5*a_preceding[i]*ts**2

ref[:, 2] = a_preceding

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
q = np.diag([10, 8, 3, 10, 1])
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

#Mdu = np.concatenate([np.concatenate(row, axis=1) for row in Mdu_cell], axis=0)
Mdu = np.block(Mdu_cell)

## Constraints setting
# Constraints of u
umin = -4.5
umax = 2.5
U_min = np.array([umin] * Nc)
U_max = np.array([umax] * Nc)
# Constrain of delta u
du_min = -3
du_max = 3
delta_umax = [du_max] * Nc
delta_umin = [du_min] * Nc
Row = 3                #松弛系数
M = 10                 #松弛变量上界
lb = delta_umin + [0]  #（求解方程）状态量下界，包含控制时域内控制增量和松弛因子
lb = np.array(lb)
ub = delta_umax + [M]  #（求解方程）状态量上界，包含控制时域内控制增量和松弛因子
ub = np.array(ub)
# 输出状态约束
es_min = -1
es_max = 3
ev_min = -2
ev_max = 2
a_min = -4.5
a_max = 2.5
v_min = 0
v_max = 40
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
vy_max = np.array([1, 1, 0.1, 1, 1])
VY_min = np.kron(np.ones(Np), vy_min)
VY_max = np.kron(np.ones(Np), vy_max)

## 初始值设置和预定义
X = np.zeros((n+1, 5))
U = np.zeros((n+1, 1))
X[0, :] = [0, 0, 0, 0, 0]
U[0, :] = 0
A_I = np.kron(tril(np.ones((Nc, Nc))), np.eye(Nu))
PSI = np.kron(np.ones(Nc), np.eye(Nu))

a_2_list = np.zeros((n, 1))

## MPC
s0 = 2
for k in range(n):
    print(k)
    vi = v_preceding[k] - X[k, 1] - X[k, 4]
    delta_v = -X[k, 4]
    delta_d = X[k, 3] + s0 + vi * T
    a_2 = IDM(vi, delta_v, delta_d)
    a_2_list[k] = a_2

    w = np.array([a_preceding[k], a_2])

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

    # constraints = [{'type': 'ineq', 'fun': lambda du: b_cons - A_cons @ du}]
    # #result = minimize(objective, np.zeros(Nu * Nc + 1), constraints=constraints, bounds=bounds)
    # res = minimize(lambda du: 0.5* du.T @ H @ du + f @ du,
    #                x0=np.zeros(Np),
    #                #constraints={'type': 'ineq', 'fun': lambda du: b_cons - A_cons @ du},
    #                constraints=[{'type': 'ineq', 'fun': lambda du: (b_cons - A_cons @ du).flatten()}],
    #                bounds=np.vstack((lb, ub)).T)
    du = cp.Variable(Np)
    objective = cp.Minimize(0.5 * cp.quad_form(du, H) + f.T @ du)

    print("A_cons shape:", A_cons.shape)
    print("du shape:", du.shape)
    print("b_cons shape:", b_cons.shape)
    print("lb shape:", lb.shape)
    print("ub shape:", ub.shape)

    constraints = [A_cons @ du <= b_cons, lb <= du, du <= ub]
    problem = cp.Problem(objective, constraints)
    result = problem.solve()

    res = du.value

    # 更新控制量
    U[k+1, :] = U[k, :] + res[:Nu]
    # 更新状态量
    X[k + 1, :] = A @ X[k, :].T + B1 @ U[k + 1, :].T + (B2 @ w).T

## plot
# 在 Python 中实现
v_cav = v_preceding - X[:n, 1]
v_hv  = v_preceding - X[:n, 1] - X[:n, 4]
x_cav = x_preceding - X[:n, 0] - L - h * v_cav
x_hv  = x_cav - X[:n, 3] - L - T * v_hv

# Create a figure and subplot
fig, axs = plt.subplots(3, 1, figsize=(6, 8))

axs[0].plot(t, a_preceding, label='preceding')
axs[0].plot(t, X[:n, 2], label='CAV')
axs[0].plot(t, a_2_list, label='HV')
axs[0].set_ylim([-2, 2.5])
axs[0].set_xlabel('Sampling time T')
axs[0].set_ylabel('Acceleration a')
axs[0].legend(loc='upper right')
axs[0].grid(True)

axs[1].plot(t, v_preceding, label='preceding')
axs[1].plot(t, v_cav, label='CAV')
axs[1].plot(t, v_hv, label='HV')
axs[1].set_xlabel('Sampling time T')
axs[1].set_ylabel('Speed v')
axs[1].legend(loc='upper right')
axs[1].grid(True)

axs[2].plot(t, x_preceding, label='preceding')
axs[2].plot(t, x_cav, label='CAV')
axs[2].plot(t, x_hv, label='HV')
axs[2].set_xlabel('Sampling time T')
axs[2].set_ylabel('Position x')
axs[2].legend(loc='upper right')
axs[2].grid(True)

# Adjust space between subplots
plt.tight_layout()

# Save the figure
filename = f"MPC_cav_hv_Np{Np}_Nc{Nc}.png"
plt.savefig(filename, dpi=300)

plt.show()