import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

scenario = '_mpc_base2'  # Only considering CAV, multi-step prediction
Np = 17   # Prediction horizon, requires multiple tunings
Nc = 3    # Control horizon, requires multiple tunings
q = np.diag([1, 2, 2, 0])  # Output variable weights

scenario = '_mpc_base3'  # Considering following vehicle, single-step prediction
Np = 4   # Prediction horizon, requires multiple tunings
Nc = 2    # Control horizon, requires multiple tunings
q = np.diag([1, 2, 0.5, 0])  # Output variable weights


### Load data  # d0: minimum safe distance; h: desired headway
# Acceleration cases
# file_name, d0, h = '43_61_67_77_94_102.csv' , 2, 2.25
# file_name, d0, h = '267_277_282_293_299_311.csv', 1.2, 1.2

# Deceleration cases
# file_name, d0, h = '350_355_361_364_372_379.csv' , 1, 0.9
# file_name, d0, h = '660_683_681_694_695_711.csv' , 1.5, 1.4

# Constant speed or mixed acceleration and deceleration
# file_name, d0, h = '1_11_24_44_55_54.csv' , 1.22, 1.5
# file_name, d0, h = '59_79_87_93_97_116.csv' , 1.3, 1.3
# file_name, d0, h = '79_87_93_97_116_117.csv' , 1.5, 1.8
# file_name, d0, h = '293_299_311_329_348_343.csv', 1.5, 1.8
# file_name, d0, h = '305_314_313_326_339_343.csv', 1.5, 1.8
# file_name, d0, h = '124_136_140_146_162_164.csv', 1.3, 1.2
# file_name, d0, h = '650_656_662_666_706_705.csv', 1.3, 1.2
file_name, d0, h = '903_907_914_926_920_939.csv', 1, 1.2

#data = pd.read_csv(f'./data/NGSIM_I80_predicted/{file_name}')
data = pd.read_csv(f'./data/NGSIM_I80_results/{file_name}')

# Check if 'a4_PERL' column exists
if 'a4_PERL' not in data.columns:
    print("a4_PERL column not found, using example values.")

# Calculate preceding vehicle's speed and position (based on cumulative sum of a4_PERL)
ts = 0.1  # Simulation timestep, adjust based on actual needs
v_initial = data[data['t1'] == 0]['v4'][0]
data['v4_PERL'] = np.cumsum(data['a4_PERL'] * ts) + v_initial
p_initial = data[data['t1'] == 0]['Y4'][0]
data['Y4_PERL'] = np.cumsum(data['v4_PERL'] * ts) + p_initial

# Extract a4_PERL and v4_PERL for MPC
a_preceding = data['a4_PERL'].values
v_preceding = data['v4_PERL'].values
p_preceding = data['Y4_PERL'].values

# Time settings
t = np.arange(0, len(a_preceding) * ts, ts)
n = len(t)


# Step 0: Parameter settings
L = 4     # Vehicle length
tao_h = 0     # Engine delay
tao = 0.1  # Mechanical delay
ts = 0.1  # Simulation timestep
# Np = 19   # Prediction horizon, requires multiple tunings
# Nc = 3    # Control horizon, requires multiple tunings
Nx = 4    # Number of state variables [e_s, e_v, a_t, v_t]
Ny = 4    # Number of output variables
Nu = 1    # Number of control variables


window_size = 5
# Use moving average to smooth speed
v_preceding_smooth = pd.Series(v_preceding).rolling(window=window_size, min_periods=1, center=True).mean().values

# Check for abnormal acceleration and handle it
a_preceding = np.diff(v_preceding, prepend=v_preceding_smooth[0]) / ts
mask = np.abs(a_preceding) > 2
v_preceding[mask] = np.nan  # Set speed corresponding to abnormal acceleration to NaN
v_preceding_filled = pd.Series(v_preceding_smooth).interpolate().values  # Use interpolation to fill

a_preceding = np.diff(v_preceding_filled, prepend=v_preceding_filled[0]) / ts

# Step 1: Reference trajectory
ref = np.zeros((n, 4))
ref[:, 0] = v_preceding_filled*h
#ref[:, 2] = a_preceding  # Preceding vehicle's acceleration
ref[:, 3] = v_preceding_filled  # Preceding vehicle's speed

# Step 2: System state equation
# (1) Continuous system state equation: dx = A*x+B1*u+B2*w; y = C*x+D*u
A = ts * np.array([[0, 1, -tao_h, 0],
                   [0, 0, -1, 0],
                   [0, 0, -1/tao, 0],
                   [0, 0, 1, 0]]) + np.eye(4)
B1 = ts * np.array([[0], [0], [1/tao], [0]])
B2 = ts * np.array([[0], [1], [0], [0]])
C = np.eye(4)
r = 0.1  # Control increment weight


# Output state constraints
a_min, a_max = -4, 4
v_min, v_max = 0, 20

# Initial value settings and predefined arrays
X = np.zeros((n+1, Nx))
U = np.zeros((n+1, Nu))
es_t0 = data[data['t1'] == 0]['Y4'][0] - data[data['t1'] == 0]['Y5'][0] - L - d0
ev_t0 = -data[data['t1'] == 0]['v5'][0] + data[data['t1'] == 0]['v4'][0]
a5t0 = data[data['t1'] == 0]['a5'][0]
v5t0 = data[data['t1'] == 0]['v5'][0]
X[0, :] = [es_t0, ev_t0, a5t0, v5t0]
U[0, :] = a5t0

# Step 7: Main MPC framework
for k in range(n-Np):
    print("Current timestep:", k)
    J = 0  # Initialize objective function value
    U_var = cp.Variable(Nc)  # Define control variable as acceleration
    constraints = [U_var >= a_min, U_var <= a_max]  # Acceleration constraints

    # Initialize current state as the state of the current time step
    current_X_k = X[k, :].reshape(-1, 1)

    for i in range(Np):
        # Compute control input: for i beyond Nc, use the last control input value
        U_i = U_var[i] if i < Nc else U_var[Nc-1]

        # Compute the next state
        w_i = a_preceding[k + i] if k + i < n else a_preceding[-1]
        X_next = A @ current_X_k + B1 * U_i + B2 * w_i  # Use * for scalar multiplication
        # Compute the error
        error = X_next - ref[k + i, :].reshape(-1, 1)
        # Objective function
        J += cp.quad_form(error, q)
        # Update current state to the predicted next state
        current_X_k = X_next

    # Define and solve the optimization problem
    objective = cp.Minimize(J)
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver=cp.ECOS)

    # Update control input and state based on the optimization result
    for i in range(min(Nc, n - k)):
        # Update control input: here, update the actual control sequence U
        U[k + 1 + i, 0] = U_var.value[i]
    # Update state
    if k + 1 <
