import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


scenario = '_proposed'
Np = 17  # Prediction horizon, requires multiple tuning
Nc = 3   # Control horizon, requires multiple tuning
q = np.diag([1, 2, 2, 0])  # Output variable weights


# Load data
file_name, d0, h = '59_79_87_93_97_116.csv', 1.3, 1.3
data = pd.read_csv(f'./data/NGSIM_I80_results/{file_name}')

# Ensure 'a4_PERL' column exists
if 'a4_PERL' not in data.columns:
    print("a4_PERL column not found, using example values.")

# Calculate preceding vehicle's speed and position based on 'a4_PERL' cumulative sum
ts = 0.1  # Simulation time step, adjust based on actual conditions
v_initial = data[data['t1'] == 0]['v4'][0]
data['v4_PERL'] = np.cumsum(data['a4_PERL'] * ts) + v_initial
p_initial = data[data['t1'] == 0]['Y4'][0]
data['Y4_PERL'] = np.cumsum(data['v4_PERL'] * ts) + p_initial

# Extract 'a4_PERL', 'v4_PERL' for MPC
a_preceding = data['a4_PERL'].values
v_preceding = data['v4_PERL'].values
p_preceding = data['Y4_PERL'].values

# Time settings
t = np.arange(0, len(a_preceding) * ts, ts)
n = len(t)


# Parameter settings
L = 4     # Vehicle length
tao_h = 0     # Engine delay
tao = 0.1 # Mechanical delay
ts = 0.1  # Simulation time step
Nx = 4    # Number of state variables [e_s, e_v, a_t, v_t]
Ny = 4    # Number of output variables
Nu = 1    # Number of control variables


# Check for abnormal accelerations and handle them
a_preceding = np.diff(v_preceding, prepend=v_preceding[0]) / ts
mask = np.abs(a_preceding) > 2
v_preceding[mask] = np.nan  # Set speed to NaN for abnormal accelerations
v_preceding_filled = pd.Series(v_preceding).interpolate().values  # Interpolate to fill missing values

a_preceding = np.diff(v_preceding_filled, prepend=v_preceding_filled[0]) / ts

# Reference trajectory
ref = np.zeros((n, 4))
ref[:, 0] = v_preceding_filled*h
ref[:, 2] = a_preceding  # 前车加速度
ref[:, 3] = v_preceding_filled  # 前车速度

# System state
# dx = A*x+B1*u+B2*w; y = C*x+D*u
A = ts * np.array([[0, 1, -tao_h, 0],
                   [0, 0, -1, 0],
                   [0, 0, -1/tao, 0],
                   [0, 0, 1, 0]]) + np.eye(4)
B1 = ts * np.array([[0], [0], [1/tao], [0]])
B2 = ts * np.array([[0], [1], [0], [0]])
C = np.eye(4)
r = 0.1

# Output state constraints
a_min, a_max = -4, 4
v_min, v_max = 0, 15

# Initialize and predefine values
X = np.zeros((n+1, Nx))
U = np.zeros((n+1, Nu))
es_t0 = data[data['t1'] == 0]['Y4'][0] - data[data['t1'] == 0]['Y5'][0] - L - d0
ev_t0 = -data[data['t1'] == 0]['v5'][0] + data[data['t1'] == 0]['v4'][0]
a5t0 = data[data['t1'] == 0]['a5'][0]
v5t0 = data[data['t1'] == 0]['v5'][0]
X[0, :] = [es_t0, ev_t0, a5t0, v5t0]
U[0, :] = a5t0

# MPC framework
for k in range(n-Np):
    print("Current timestep:", k)
    J = 0
    U_var = cp.Variable(Nc)
    constraints = [U_var >= a_min, U_var <= a_max]  # Acceleration constraints

    # Initialize current state as the state at this time step
    current_X_k = X[k, :].reshape(-1, 1)

    for i in range(Np):
        # Calculate control input: use the last control input value for i beyond Nc range
        U_i = U_var[i] if i < Nc else U_var[Nc-1]
        # Calculate the next state
        w_i = a_preceding[k + i] if k + i < n else a_preceding[-1]
        X_next = A @ current_X_k + B1 * U_i + B2 * w_i
        # Calculate error
        error = X_next - ref[k + i, :].reshape(-1, 1)
        # Objective function
        J += cp.quad_form(error, q)
        # Update current state as the predicted next state
        current_X_k = X_next

    # Define optimization problem and solve
    objective = cp.Minimize(J)
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver=cp.ECOS)

    # Update control and state based on optimization results
    for i in range(min(Nc, n - k)):
        # Update control
        U[k + 1 + i, 0] = U_var.value[i]
    # Update state
    if k + 1 < n:
        w_i = a_preceding[k + 1] if k + 1 < n else a_preceding[-1]
        X[k + 1, :] = (A @ X[k, :].reshape(-1, 1) + B1 * U[k + 1, 0] + B2 * w_i).flatten()


# Recalculate the number of NaNs to fill to ensure consistent length
nan_fill_count = len(data) - (n - Np)
# Calculate the fifth vehicle's acceleration, speed, and position
data[f'a5{scenario}'] = np.concatenate([U[:n-Np, 0], [np.nan] * nan_fill_count])
data[f'v5{scenario}'] = np.concatenate([X[:n-Np, 3], [np.nan] * nan_fill_count])
# Use cumsum to calculate position, assuming initial position as v5t0
initial_position = data.loc[data['t1'] == 0, 'Y5'].values[0]

data[f'Y5{scenario}'] = np.cumsum(data[f'v5{scenario}'].fillna(method='ffill') * ts) + initial_position

data[f'Y5{scenario}'][n-Np:] = np.nan

data.to_csv(f'./data/NGSIM_I80_results/{file_name}', index=False)
print(f"CSV file has been updated with new columns: a5{scenario}, v5{scenario}, Y5{scenario}")




def plot_results(t, X, data):

    Y5_MPC_initial = data[data['t1'] == 0]['Y5'][0]
    Y5_MPC = np.cumsum(X[:n, 3] * ts) + Y5_MPC_initial

    # Set up plots
    fig, ax = plt.subplots(5, 1, figsize=(5, 10), sharex=True)

    colors = ['#8391a5', '#5c6f89', '#354f6d', '#003153', 'red', '#0d151f']
    linestyles = [(0, (5, 7)), (0, (4, 5)), (0, (3, 3)), (0, (2, 1)), '-', (0, (1, 1))]
    linewidths = [0.6, 0.7, 0.8, 1, 1.3, 1.5]

    for i in range(4):
        ax[0].plot(data['t1'], data[f'a{i + 1}'], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i],
                   label=f'Vehicle {i + 1}')
        ax[1].plot(data['t1'], data[f'v{i + 1}'], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i],
                   label=f'Vehicle {i + 1}')
        ax[2].plot(data['t1'], data[f'Y{i + 1}'], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i],
                   label=f'Vehicle {i + 1}')

    i = 4
    # acceleration
    ax[0].plot(data['t1'], X[:n, 2], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i],
               label=f'Vehicle {i + 1}')
    # speed
    ax[1].plot(data['t1'], X[:n, 3], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i],
               label=f'Vehicle {i + 1}')
    # position
    ax[2].plot(data['t1'], Y5_MPC[:len(t)], color=colors[i], linestyle=linestyles[5], linewidth=linewidths[i],
               label=f'Vehicle {i + 1}')

    ax[0].set_ylabel('Acceleration (m/s²)')
    ax[1].set_ylabel('Velocity (m/s)')
    ax[2].set_ylabel('Position (m)')
    ax[2].set_xlabel('Time (s)')

    i = 5
    ax[3].plot(data['t1'], X[:n, 0], label = 'real')
    ax[3].plot(data['t1'],ref[:, 0], label = 'reference')
    ax[3].set_ylabel('Headway_Distance (m)')
    ax[3].grid = True
    ax[3].legend()
    ax[4].plot(data['t1'], X[:n, 1])
    ax[4].set_ylabel('Velocity_error (m)')

    plt.tight_layout()
    plt.show()

plot_results(t, X, data)

